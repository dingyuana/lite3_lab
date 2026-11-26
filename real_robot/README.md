# 真机 task 系列

本目录可单独复制使用，但视觉跟随需要 `assets/` 中的标定文件和 YOLO 权重。所有脚本默认机器人 IP 为 `192.168.1.120`。

## 安全准备

- 使用悬挂、护架或人工扶稳完成首次起立与方向验证。
- 清理前后 2 m 范围，确认侧面空间；侧移没有超声波保护。
- 手边执行软急停：

```bash
python3 -c "import lite3_common as l3,time; c=l3.Lite3Controller('192.168.1.120'); c.start_heartbeat(5); c.send_simple(l3.CMD_SOFT_ESTOP); time.sleep(1); c.close()"
```

## 安装

```bash
cd real_robot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo apt install python3-tk
```

## 实验顺序

> 历史编号中没有 task04，课程直接从 task03 进入 task05。

| 任务 | 入口 | 目标 |
|---|---|---|
| task01 | `task01_heartbeat.py` | 心跳保活、UDP 双向链路与统计 |
| task02 | `task02_state_receiver.py` | 只读 `0x0901/0x0902` 状态与关节帧 |
| task03 | `task03_stand_to_force.py` | 起立、力控站立、趴下、归零 |
| task05 | `task05_move_mode_low_gait.py` | 移动模式与低速步态，不发运动轴 |
| task06 | `task06_axis_tap.py` | 前进点动与死区验证 |
| task07 | `task07_axis_stream_forward.py` | 持续前进、可配置停车观察窗口 |
| task08 | `task08_ultrasonic_guard_forward.py` | 超声波保护与停车 |
| task09 | `task09_axis_directions.py` | 前后左右平移符号验证，不含偏航转向 |
| task10 | `task10_gui.py` | Tkinter 键盘、视频、状态与控制面板 |
| task12 | `task12_follow.py` | 在 task10 界面中集成 YOLO 人体跟随 |

```bash
python3 task01_heartbeat.py
python3 task02_state_receiver.py --joints
python3 task03_stand_to_force.py
python3 task05_move_mode_low_gait.py
python3 task06_axis_tap.py --value 8000
python3 task07_axis_stream_forward.py --duration 3 --cooldown 3
python3 task08_ultrasonic_guard_forward.py
python3 task09_axis_directions.py
python3 task10_gui.py
```

## task10 / task12

GUI 按键：

- 方向键：按住移动，松开归零。
- `Space`：起立/趴下。
- `T`：开始/停止人体跟踪。
- `Esc` 或窗口失焦：停止运动并退出跟踪。
- “回零”按钮：先趴下，再发送 `0x21010C05`。

```bash
# 不依赖相机、机器人和标定文件的 PID 数值自检
python3 task12_follow.py --selfcheck

# 脱机视觉链路；不会发送运动指令
python3 task12_follow.py --offline

# 没有真实标定时，仅脱机使用占位矩阵
python3 task12_follow.py --offline --dummy-h

# 交互式标定，默认写入 assets/H.npy
python3 task12_follow.py --calibrate

# 真机 GUI，按 T 开始跟踪
python3 task12_follow.py --auto-follow
```

默认资源：

- `assets/H.npy`：真实四点单应标定。
- `assets/H_dummy.npy`：占位矩阵，仅脱机。
- `assets/yolov8n.pt`：YOLOv8 nano 权重。

标定点顺序为**近左 → 近右 → 远右 → 远左**。默认地面坐标为 `(-1,1) (1,1) (1,3) (-1,3)`，原点位于机器人地面投影，X 向右、Y 向前。`--dummy-h` 与 dummy 文件在真机模式会被拒绝。

## 关键安全逻辑

- task07 的 `--cooldown` 会传入安全停车函数；未停稳不会报告成功。
- task12 使用真实视频帧时间戳，帧龄超过 0.2 s 立即停车。
- 独立看门狗在 1.0 s 无控制更新时停车、退出跟踪状态并恢复手动控制。
- 超声波超时按原始 `0x0901` 状态帧龄判断，不用“读取时刻”刷新数据年龄。
- 后退与转向轴值会越过固件死区；零值始终保持零。
- 手动与跟踪模式互斥，跟踪停止后恢复手动控制。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

覆盖协议状态、启动门控、task07 停车参数、task10 防撞与互斥、task12 帧龄/看门狗/死区/dummy 拦截。

## 文档与历史代码

- GUI 详细说明：[`docs/TASK10_README.md`](docs/TASK10_README.md)
- 课程讲义：[`docs/`](docs/README.md)
- `legacy/task11_follow.py` 只用于阅读历史实现，不作为安装、测试或教学运行入口。
