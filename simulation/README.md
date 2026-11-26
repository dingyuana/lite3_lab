# 仿真 stage 系列

本目录使用项目根目录中的 `Lite3_rl_deploy/` 模型与 ONNX 策略。`project_paths.py` 自动定位资源，不依赖 `/home/<用户名>` 或个人绝对路径。

## 安装

```bash
cd simulation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ROS2 实验还需安装 ROS2、`drdds` 消息包及对应 `sensor_msgs_py`、`tf2_ros` 等 ROS2 依赖，并先 source ROS2 工作空间。不同 ROS2 发行版的包安装方式不同，本目录不通过 pip 冒充完整 ROS2 环境。

## 阶段顺序

| 阶段 | 入口 | 内容 |
|---|---|---|
| stage1 | `stage1_pybullet.py` | PyBullet 模型、站立/蹲伏/trot、方向键与推力 |
| stage2 | `stage2_mujoco_lab.py` | MuJoCo 站立、安全蹲伏、摩擦/重力切换与扰动 |
| stage3 | `stage3_mujoco_rl.py` | ONNX RL 策略、45 维观测、PD 力矩伺服 |
| stage3 ROS2 | `stage3_mujoco_ros2_sim.py` | `/JOINTS_CMD` 与 `/JOINTS_DATA`、MIT 力矩 |
| stage4 | `stage4_ros2_controller.py` + `stage4_ros2_keyboard.py` + `stage4_ros2_walk_sim.py` | `/cmd_vel` 到关节指令、键盘与无窗口运行 |
| stage5 | `stage5_lidar_frontend_sim.py` | 3D LiDAR 点云、真值里程计/TF、Marker |

```bash
python3 stage1_pybullet.py
python3 stage2_mujoco_lab.py
python3 stage3_mujoco_rl.py
```

`stage3_mujoco_rl.py` 的终端控制：`w/s` 前后、`a/d` 转向、`x` 停止、`p` 侧向 150 N 推力、`r` 复位、`q` 退出。`stage3_ros2_node.py` 的有效代码处于注释状态，仅作历史草稿，不是运行入口。

## stage3 关键实现

- 投影重力使用 `R^T * (0,0,-1)`。
- 默认关节角按部署策略使用 `(0, -0.65, 1.3)` 循环。
- 观测顺序为角速度(3)、投影重力(3)、速度指令(3)、关节位置偏差(12)、关节速度(12)、上一帧动作(12)。
- 电机执行器下发力矩：`30*(target-qpos) + 1*(0-qvel)`，并按执行器控制范围限幅。
- ROS2 仿真拒绝关节数量错误或非有限值指令；0.5 s 无新 `/JOINTS_CMD` 时输出零力矩。

## stage5 的边界

stage5 **不是 SLAM 算法实现**。它生成 LiDAR 点云并发布 `/ground_truth_odom` 与真值 TF，作为外部 SLAM 系统的数据前端。课程中应明确区分“仿真真值”和“SLAM 估计”。

## 测试与工具

```bash
python3 -m unittest discover -s tests -v
python3 tools/lite3_check.py
python3 tools/lite3_pybullet_lab.py
```

单元测试只依赖 NumPy，覆盖重力投影、默认关节角、PD 力矩和观测构造，不会打开仿真窗口。`tools/smoke_test.py` 是 PyBullet 通用 GUI 冒烟脚本，不是 Lite3 运动实验。

## 资源

- `../Lite3_rl_deploy/third_party/deep_robotics_model/Lite3/`：URDF/MJCF 模型。
- `../Lite3_rl_deploy/policy/ppo/policy.onnx`：部署策略。
- `assets/generated/`：历史生成的 MuJoCo 模型文件，仅作保存，不参与课程运行。

## 课程文档

- 文档索引：[`docs/README.md`](docs/README.md)
- 离线 HTML 课程：[`docs/html/index.html`](docs/html/index.html)
- Markdown 讲义：`docs/仿真控制课程/`
