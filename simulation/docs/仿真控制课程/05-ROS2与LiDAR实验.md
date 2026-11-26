# Lite3 的 ROS2 与 LiDAR 实验

## 学习目标

完成本单元后，学生应能够：

1. 搭建 stage4 的 `/cmd_vel → /JOINTS_CMD → /JOINTS_DATA` ROS2 控制链。
2. 说明键盘、步态转换节点和 MuJoCo 仿真节点各自的职责与发布频率。
3. 正确启动 stage4 的两个替代控制器之一，避免多发布者争抢同一话题。
4. 识别 stage5 发布的 `/lidar_points`、`/ground_truth_odom`、TF 和 `/dog_body`。
5. 说明 stage5 的点云生成方式、坐标系和发布周期。
6. 明确 stage5 只是 LiDAR/SLAM 数据前端，发布仿真真值，不执行建图、定位或回环。
7. 根据话题列表、订阅关系、TF 和日志排查通信故障。

## 建议学时

建议 **6 学时**：教师演示 2 学时，学生实操 4 学时。

| 活动 | 学时 |
|---|---:|
| ROS2 控制链与节点频率 | 1 |
| stage4 键盘/控制器/仿真联调 | 1 |
| stage5 数据前端观察 | 1.5 |
| 话题、TF 与边界验收 | 1.5 |
| 故障排查与课堂总结 | 1 |

## 先修知识

- 了解 ROS2 节点、话题、发布者、订阅者、消息类型和 TF。
- 能在每个新终端中 source ROS2 及 `drdds` 消息包所在工作空间。
- 理解 `/cmd_vel` 的 `linear.x`、`angular.z` 含义。
- 理解 stage3 的 12 关节指令、PD/MIT 力矩与 0.5 秒超时。
- 了解 PointCloud2、Odometry、TransformStamped 和 Marker 的基本作用。
- 具备基本 Linux 终端操作能力，stage4 键盘和 stage5 都需要交互式 TTY。

## 核心概念

### stage4 控制链

推荐的控制链为：

```text
键盘节点 stage4_ros2_keyboard.py
    │ Twist, /cmd_vel
    ▼
步态转换节点（walk_sim 或 controller 二选一）
    │ JointsDataCmd, /JOINTS_CMD
    ▼
MuJoCo 仿真 stage3_mujoco_ros2_sim.py
    │ JointsData, /JOINTS_DATA
    ▼
状态观察/反馈
```

每个节点有独立职责：

- **键盘节点**只生成速度命令；
- **步态转换节点**把速度命令转换为 12 个关节的 MIT 参数；
- **仿真节点**校验并执行关节指令，同时发布关节状态。

### stage4 的频率与停止语义

| 组件 | 周期 | 注意事项 |
|---|---:|---|
| `stage4_ros2_keyboard.py` | `0.05 s`，20 Hz | W/S/A/D/空格；停止按键输入 1 秒后自动把速度命令清零 |
| `stage4_ros2_walk_sim.py` | `0.02 s`，50 Hz | 无 MuJoCo 窗口；停止时把内部步态时间 `t` 清零 |
| `stage4_ros2_controller.py` | `0.002 s`，500 Hz | 无 MuJoCo 窗口；时间持续递增，停止时只把相位设为 0 |
| `stage3_mujoco_ros2_sim.py` | 每次循环约 `0.002 s` | 每 5 步发布一次关节状态；0.5 秒无有效关节命令则零力矩 |

键盘的 1 秒自动停车与仿真器的 0.5 秒超时是两个不同机制：

- 前者让键盘在无输入后持续发布零 `/cmd_vel`；
- 后者让仿真器在没有新的有效 `/JOINTS_CMD` 时输出零力矩。

如果直接杀掉键盘进程，步态转换节点可能保留最后收到的 `/cmd_vel` 并继续发布关节命令。因此安全停止应先按空格，再结束进程；不要只关闭键盘终端。

### 两个 stage4 转换节点只能二选一

`stage4_ros2_walk_sim.py` 与 `stage4_ros2_controller.py` 都发布 `/JOINTS_CMD`。同时启动会产生两个发布者，机器人会收到交错的两套步态命令。

| 行为 | `stage4_ros2_walk_sim.py` | `stage4_ros2_controller.py` |
|---|---|---|
| 停止时 | `t=0`、相位归零 | 相位归零，但 `t` 继续增加 |
| 转向 HipY 修正 | 左右腿分别加 `±0.1*wz` | 所有 HipY 统一加 `+0.1` 或 `-0.1` |
| 抬膝半周期 | 根据 `vx` 正负选择 `sin(p)` 或 `-sin(p)` | 固定使用 `max(0,-sin(p))` |
| 发布频率 | 50 Hz | 500 Hz |

两者都使用基础站姿 `HipY=-0.985`、`Knee=2.084`、左右 HipX 为 `±0.1`，并下发：

```text
kp = 40
kd = 2
position = target
torque = 0
```

这是脚本式步态，不是 stage3 的 ONNX 强化学习策略。节点名中的“无窗口”只表示转换节点自身不打开 MuJoCo 查看器，仿真节点仍会打开查看器。

### ROS2 关节指令校验回顾

`stage3_mujoco_ros2_sim.py` 只接受：

1. 恰好 `model.nu` 个关节，当前为 12；
2. 能解析为数值的 `[kp,kd,position,velocity,torque]`；
3. 所有数值均有限。

有效命令的力矩为：

```text
tau = kp*(position-q) + kd*(velocity-dq) + torque
```

无效消息被拒绝且不更新时间戳。超过 0.5 秒没有新的有效消息时，仿真器输出 12 维全零力矩。

### stage5 的节点边界

`stage5_lidar_frontend_sim.py` 是独立的 MuJoCo + ROS2 数据前端：

- 它**不订阅** `/cmd_vel` 或 `/JOINTS_CMD`；
- 键盘 `w/s/a/d/空格` 在节点内部直接设置 `vx/wz`；
- 它自行计算 12 关节 PD 力矩，`kp=40`、`kd=2`；
- 它不调用 ONNX，也不运行 stage4 的步态转换节点。

因此，在 stage5 中启动键盘遥控或 stage4 控制器不会形成预期控制链。stage5 的键盘速度大小为 `0.3`，与 stage4 键盘的 `0.5` 也不同。

### stage5 发布内容

| 输出 | 类型 | 坐标系/内容 |
|---|---|---|
| `/lidar_points` | `sensor_msgs/PointCloud2` | `world` 系中的命中点，字段为 `x,y,z` |
| `/ground_truth_odom` | `nav_msgs/Odometry` | `world` 系机身真值位姿 |
| TF | `TransformStamped` | `world → base_link` 真值变换 |
| `/dog_body` | `visualization_msgs/Marker` | `world` 系红色立方体，跟随机身位姿 |

里程计和 TF 都直接读取 `data.qpos[0:7]`，不是通过传感器估计得到的。当前 Odometry 主要填写位姿，没有主动填充速度 twist 和协方差。

### 理想 LiDAR 点云如何生成

每次发布时，节点：

1. 以机身位置上方 `0.2 m` 为雷达原点；
2. 用机身四元数把局部射线方向旋转到世界系；
3. 在 4 个垂直角 `[-15°,-5°,5°,15°]` 与 36 个水平角 `[0°,10°,…,350°]` 上发射射线；
4. 使用 `mujoco.mj_ray` 求首个命中距离；
5. 只保留 `0 < distance < 15 m` 的命中点；
6. 打包为 `world` 系的 PointCloud2。

理想情况下最多有 `4×36=144` 条射线方向，但只有发生有效命中的方向才产生点。若当前没有点，节点不会发布空点云消息。噪声代码处于注释状态，默认没有模拟运动畸变或随机噪声。

点云表达为世界系坐标，点云消息本身不携带雷达安装外参。若接入期望 `lidar` 或 `sensor_frame` 坐标的外部系统，需要额外提供坐标变换，这已超出当前脚本范围。

### 发布节奏与课程边界

主循环每次睡眠约 `0.002 s`，每 50 步发布一次点云、里程计、TF 和 Marker。若模型时间步为 `0.002 s`，名义频率约为 10 Hz，但实际间隔还受运行负载影响。

必须明确：

> stage5 不是 SLAM。它只模拟 LiDAR 数据前端，并发布仿真真值里程计和 TF；不建立地图、不做定位、不做扫描匹配、不构建位姿图、不做回环，也不发布 SLAM 估计轨迹。

真值数据可用于外部 SLAM 的输入、可视化和误差评测，但不能把 `/ground_truth_odom` 当作算法输出。

## 教师演示

### 演示 1：启动三节点控制链

以下命令中的 ROS2 setup 路径需按机房实际环境替换。每个新终端都要重新 source：

```bash
source /opt/ros/<发行版>/setup.bash
source <工作空间>/install/setup.bash
cd ~/lite3_lab/simulation
source .venv/bin/activate
```

终端 1：

```bash
python3 stage3_mujoco_ros2_sim.py
```

终端 2（选择一个，不要同时运行）：

```bash
python3 stage4_ros2_walk_sim.py
```

或：

```bash
python3 stage4_ros2_controller.py
```

终端 3：

```bash
python3 stage4_ros2_keyboard.py
```

### 演示 2：观察键盘与停止行为

1. 键盘终端不需要回车，按 `w/s/a/d` 观察 `/cmd_vel` 日志。
2. 按空格，确认线速度和角速度清零。
3. 松开所有键，观察 1 秒后键盘自动发布停车命令。
4. 结束步态转换节点，再按键盘；解释最后命令可能被转换节点保留。
5. 重新启动转换节点，验证 `/JOINTS_DATA` 反馈恢复。

### 演示 3：话题与 TF 检查

在已 source 的第 4 个终端执行：

```bash
ros2 topic list
ros2 topic info /cmd_vel --verbose
ros2 topic info /JOINTS_CMD --verbose
ros2 topic echo /JOINTS_DATA --once
ros2 run tf2_ros tf2_echo world base_link
```

让学生根据发布/订阅关系判断链路断在哪一段，而不是只根据“机器人不动”猜原因。

### 演示 4：stage5 数据前端

先停止上述三个节点，避免资源与关节控制冲突。然后运行：

```bash
python3 stage5_lidar_frontend_sim.py
```

操作 `w/s/a/d/空格`，并在另一个已 source 的终端检查：

```bash
ros2 topic list
ros2 topic hz /lidar_points
ros2 topic echo /ground_truth_odom --once
ros2 topic echo /lidar_points --once --field width
ros2 run tf2_ros tf2_echo world base_link
```

若机房安装了 RViz2，可把 `/lidar_points`、`/dog_body` 和 `world → base_link` 加入同一场景；没有 RViz2 不影响课程验收，因为话题和 TF 命令已足以检查数据。

## 学生实操

### 任务 1：绘制并验证控制图

画出三个节点的名称、输入话题、输出话题和周期。使用 `ros2 topic info` 验证每条边，特别检查：

- `/cmd_vel` 是否只有一个步态转换订阅者；
- `/JOINTS_CMD` 是否只有一个预期发布者；
- `/JOINTS_DATA` 是否持续产生。

### 任务 2：比较两个 stage4 转换节点

在相同 `/cmd_vel` 下分别运行 `walk_sim` 与 `controller`，记录：

| 对比项 | walk_sim | controller |
|---|---|---|
| 定时器周期 |  |  |
| 停止时 `t` 的处理 |  |  |
| 转向 HipY 修正 |  |  |
| 抬膝规则 |  |  |
| 日志中位置目标的变化 |  |  |

测试时必须串行运行，不能同时启动两个发布者。

### 任务 3：停止与超时复现

1. 先按空格，确认机器人收到停车命令。
2. 结束步态转换节点但保持仿真运行，观察 0.5 秒后进入零力矩。
3. 重启转换节点，确认新的有效命令刷新时间戳。
4. 记录“键盘停车”和“仿真超时”两条日志链路，说明二者不能互相替代。

### 任务 4：stage5 数据检查

| 检查项 | 预期结果 | 实际记录 |
|---|---|---|
| `/lidar_points` | 有发布者；运动时可出现点云 |  |
| 点云坐标 | `x,y,z` 为 `world` 系坐标 |  |
| `/ground_truth_odom` | `world` 系位姿随仿真变化 |  |
| TF | `world → base_link` 与机身真值一致 |  |
| `/dog_body` | 红色 Marker 与机身位姿一致 |  |
| 地图/定位结果 | 当前脚本没有发布 |  |

### 任务 5：边界说明

用三句话向同学解释：

1. stage5 接收什么输入；
2. stage5 发布什么数据；
3. 外部 SLAM 还需要额外完成什么。

验收时不能只说“stage5 做了 SLAM 前端”，必须具体说出真值来源、点云坐标系，以及未实现的建图/定位/回环。

## 思考题

1. 为什么 stage4 的两个转换节点不能同时运行？
2. 键盘的 1 秒自动停车和仿真器的 0.5 秒超时有何区别？
3. 如果 `/cmd_vel` 有发布者但没有订阅者，问题更可能在哪个节点？
4. 如果 `/JOINTS_CMD` 有数据但 `/JOINTS_DATA` 没有，应该检查哪些运行条件？
5. 为什么 stage5 不订阅 `/cmd_vel`，却能响应 `w/s/a/d`？
6. 为什么点云消息可能暂时不出现，而不是发布一个空点云？
7. 将点云放在 `world` 系对外部 SLAM 有什么便利和限制？
8. `/ground_truth_odom` 为什么不能作为 SLAM 算法的定位结果？
9. 当前脚本没有发布 `/clock` 或雷达安装 TF，对外部系统集成可能有什么影响？

## 验收标准

| 验收项 | 合格标准 |
|---|---|
| 节点职责 | 正确说明键盘、步态转换、仿真执行的职责边界 |
| 话题链 | `/cmd_vel` 与 `/JOINTS_CMD` 发布/订阅关系正确，`/JOINTS_DATA` 有反馈 |
| 启动方式 | 每个终端都 source 正确环境，键盘与 stage5 在交互式 TTY 运行 |
| 单发布者 | 同一时间只运行一个 stage4 关节命令转换节点 |
| 停止机制 | 能分别演示键盘空格、1 秒自动停车、仿真 0.5 秒超时 |
| stage5 输出 | 识别点云、真值里程计、TF 和 Marker 的坐标系及来源 |
| 点云理解 | 知道 4×36 条射线、15 m 有效距离和“无命中不发布”的规则 |
| 课程边界 | 明确 stage5 不建图、不定位、不回环，不发布 SLAM 估计 |
| 排查能力 | 能用话题列表、话题信息和 TF 命令定位断链位置 |

## 常见错误

- **忘记在每个终端 source ROS2**：新终端不会自动继承另一个终端的环境。
- **把 `requirements.txt` 当作 ROS2 消息包安装方式**：`drdds` 等需要工作空间构建与 source。
- **同时启动两个 stage4 转换节点**：同一 `/JOINTS_CMD` 出现多发布者，步态交错。
- **在 stage4 键盘输入后按回车等待**：键盘使用单字符读取，不需要回车。
- **非交互式终端启动键盘或 stage5**：`isatty`、`termios` 或 `tty` 可能失败。
- **只关闭键盘终端就以为停车**：转换节点可能保留最后速度；应先按空格。
- **期望 stage5 接收 `/cmd_vel`**：stage5 使用自己的内部键盘状态和 PD 步态。
- **把真值 TF 当作 SLAM 估计轨迹**：它直接来自 MuJoCo `qpos`。
- **把 stage5 称为完整 SLAM**：当前没有地图、定位、扫描匹配或回环模块。
- **点云没有消息就认定雷达故障**：可能只是所有射线都未获得 `0<dist<15 m` 的有效命中。
- **忽略点云为 `world` 系**：外部 SLAM 若使用其他传感器系，需要额外坐标适配。
- **忽略缺少 `/clock` 与雷达安装变换**：当前脚本没有提供这些集成条件，不能假设外部系统可直接工作。
