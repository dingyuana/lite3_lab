# Lite3 四足机器人实训项目

面向大学生实训的教学仓库，分为**真机 task 系列**与**仿真 stage 系列**。两部分可独立开课，也可通过公共基础、坐标系、安全规范和综合项目联合使用。

> 真机运动具有跌倒风险。首次起立、转向和前进实验必须在悬挂或防跌倒条件下进行，手边准备软急停，并确保前后至少 2 m 空间。

## 目录

| 路径 | 内容 | 独立使用 |
|---|---|---|
| [`real_robot/`](real_robot/README.md) | task01~task10、task12；UDP、状态帧、GUI、视觉跟随、回归测试 | 是 |
| [`simulation/`](simulation/README.md) | stage1~stage5；PyBullet、MuJoCo、ONNX、ROS2、LiDAR 数据前端 | 是 |
| [`shared/`](shared/README.md) | 两类实验共同的基础、安全、坐标与排障资料 | 是 |
| [`docs/`](docs/README.md) | 课程地图、教师指南、评价量规、HTML 课程入口 | 是 |
| `Lite3_rl_deploy/` | 上游 RL 部署工程、模型、策略与第三方依赖 | 仿真依赖 |
| `sdk_deploy/` | 上游 SDK 与 DDS/UDP/ROS2 工程 | 扩展阅读 |

`real_robot/legacy/task11_follow.py` 仅作为历史实现保留，不再作为课程入口；task12 已包含其跟踪功能并修复安全问题。

## 快速开始

```bash
cd ~/lite3_lab
python3 -m venv .venv
source .venv/bin/activate

# 只做真机课程
pip install -r real_robot/requirements.txt

# 只做仿真课程
pip install -r simulation/requirements.txt

# 两部分都做
pip install -r requirements.txt
```

Ubuntu 的 Tkinter 支持通常由 `python3-tk` 提供；ROS2 课程需另外安装 ROS2 与 `drdds` 消息包，不能仅靠 pip 安装。

```bash
# 真机：先做只读链路检查
cd real_robot
python3 task01_heartbeat.py
python3 task02_state_receiver.py --joints

# 仿真：先验证 PyBullet 环境
cd ../simulation
python3 stage1_pybullet.py
```

## 课程入口

- 真机课程说明与实验顺序：[`real_robot/README.md`](real_robot/README.md)
- 仿真课程说明与阶段顺序：[`simulation/README.md`](simulation/README.md)
- 公共基础：[`shared/README.md`](shared/README.md)
- 课程地图与评价：[`docs/README.md`](docs/README.md)
- 可离线打印的单文件 HTML 课程：`docs/html/`、`real_robot/docs/html/`、`simulation/docs/html/`

## 默认真机参数

| 项目 | 默认值 |
|---|---|
| 运动主机 IP | `192.168.1.120` |
| 命令 UDP | `43893` |
| 状态监听 UDP | `43897` |
| RTSP | `rtsp://192.168.1.120:8554/test?tcp` |
| 心跳 / 轴流 | 5 Hz / 30 Hz，轴流必须 ≥20 Hz |
| 轴死区 | 前进 `6553`、侧向 `12553`、转向 `9553` |

## 测试

```bash
# 在项目根目录
source .venv/bin/activate
(cd real_robot && python3 -m unittest discover -s tests -v)
(cd simulation && python3 -m unittest discover -s tests -v)
```

测试不连接真机、不启动图形仿真器。真机 GUI 与仿真器仍需在具备显示环境的终端中人工验收。

## 安全要点

1. 状态 `98` 只表示初始待机；电量 ≤20% 时禁止首次起立。
2. 起立成功判据是 `(6,0,0)`、四膝 60°~90° 并稳定 3 秒。
3. 轴值归零后保持 ≥20 Hz 零值轴流，等待机器人停稳，避免 250 ms 指令超时。
4. `H_dummy.npy` 只能用于脱机测试，真机模式会拒绝占位单应矩阵。
5. task12 使用真实帧龄、独立看门狗和原始状态帧龄；链路异常立即停车。
6. stage5 是 LiDAR/SLAM **数据前端**，发布真值里程计和点云，不执行建图、定位或回环。
