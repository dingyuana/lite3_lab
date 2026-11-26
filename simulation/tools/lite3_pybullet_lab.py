import pybullet as p
import pybullet_data
import numpy as np
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = str(
    PROJECT_ROOT
    / "Lite3_rl_deploy"
    / "third_party"
    / "deep_robotics_model"
    / "Lite3"
    / "Lite3_urdf"
    / "urdf"
    / "Lite3.urdf"
)

GRAVITY_DEFAULT = -9.81
FRICTION_DEFAULT = 0.8
DT = 1.0 / 240.0

STAND_JOINT_ANGLES = {
    "HipX": 0.0, "HipY": 0.65, "Knee": -1.3
}
# 站立姿态：髋外展 0，髋屈曲约 37°，膝约 -75°

GAIT_FREQ = 2.0          # trot 步态频率 Hz
GAIT_SWING = 0.25        # 摆动幅度 rad
GAIT_KNEE_LIFT = 0.25    # 摆动相抬膝幅度 rad

LEG_NAMES = ["FL", "FR", "HL", "HR"]


# ============================================================
# 1. 连接 GUI、加载场景
# ============================================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setTimeStep(DT)
p.setGravity(0, 0, GRAVITY_DEFAULT)

plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=1.0)

# 加载 Lite3，开启自碰撞（碰撞实验需要）
robot_id = p.loadURDF(
    URDF_PATH,
    basePosition=[0, 0, 0.45],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    flags=p.URDF_USE_SELF_COLLISION,
    useFixedBase=False
)

# 识别 12 个可动关节，并按腿分组
joint_indices = []
joint_names = []
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    if info[2] != p.JOINT_FIXED:
        joint_indices.append(j)
        joint_names.append(info[1].decode())

leg_joints = {leg: [] for leg in LEG_NAMES}
for j, name in zip(joint_indices, joint_names):
    for leg in LEG_NAMES:
        if name.startswith(leg):
            leg_joints[leg].append(j)
            break

# 识别足端 link（用于摩擦实验）
foot_links = []
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    link_name = info[12].decode().lower()
    if "foot" in link_name or "toe" in link_name:
        foot_links.append(j)
if not foot_links:
    # 若名称中没有 foot/toe，退化为每条腿最后一个关节的 child link
    foot_links = [leg_joints[leg][-1] for leg in LEG_NAMES]

print("关节列表：", joint_names)
print("足端 link index：", foot_links)


# ============================================================
# 2. GUI 滑块：物理参数 + 操控指令
# ============================================================
slider_gravity = p.addUserDebugParameter("重力 Z (m/s²)", -20.0, 0.0, GRAVITY_DEFAULT)
slider_friction = p.addUserDebugParameter("足端滑动摩擦 μ", 0.01, 1.5, FRICTION_DEFAULT)
slider_mode = p.addUserDebugParameter(
    "控制模式 0站立 1行走 2趴下 3阻尼", 0, 3, 0
)
slider_fwd = p.addUserDebugParameter("前进速度", -0.5, 0.5, 0.0)
slider_yaw = p.addUserDebugParameter("转向角速度", -1.0, 1.0, 0.0)
slider_gait_freq = p.addUserDebugParameter("步态频率 Hz", 0.5, 4.0, GAIT_FREQ)
slider_swing = p.addUserDebugParameter("摆动幅度", 0.05, 0.5, GAIT_SWING)


# ============================================================
# 3. 键盘操控（与 Lite3_rl_deploy 风格一致）
# ============================================================
# z: 站立  c: 行走  x: 趴下  r: 阻尼
# wasd: 前后左右  qe: 旋转
keyboard_state = {"z": False, "c": False, "x": False, "r": False,
                  "w": False, "a": False, "s": False, "d": False,
                  "q": False, "e": False}


def on_key(event):
    key = event.key.lower()
    if key in keyboard_state:
        if event.type == p.KEY_WAS_TRIGGERED:
            keyboard_state[key] = True
        elif event.type == p.KEY_WAS_RELEASED:
            keyboard_state[key] = False


p.setKeyboardCallback(on_key)


def get_keyboard_command():
    if keyboard_state["x"]:
        return 2
    if keyboard_state["r"]:
        return 3
    if keyboard_state["c"] or any(keyboard_state[k] for k in "wasdqe"):
        return 1
    if keyboard_state["z"]:
        return 0
    return None


# ============================================================
# 4. 站立姿态与 trot 步态生成
# ============================================================
def get_standing_targets():
    """返回 12 个关节的站立目标角度（按 joint_indices 顺序）"""
    targets = []
    for j in joint_indices:
        name = p.getJointInfo(robot_id, j)[1].decode()
        if "HipX" in name:
            targets.append(STAND_JOINT_ANGLES["HipX"])
        elif "HipY" in name:
            targets.append(STAND_JOINT_ANGLES["HipY"])
        elif "Knee" in name:
            targets.append(STAND_JOINT_ANGLES["Knee"])
        else:
            targets.append(0.0)
    return targets


def get_trot_targets(t, freq, swing_amp, knee_lift):
    """
    简单 trot 步态：对角腿同相，相位差 π。
    FL + HR 同相，FR + HL 同相。
    在 HipY 上叠加正弦摆动，Knee 在摆动相加抬膝。
    """
    phase_fl_hr = 2 * np.pi * freq * t
    phase_fr_hl = phase_fl_hr + np.pi

    targets = []
    for j in joint_indices:
        name = p.getJointInfo(robot_id, j)[1].decode()
        leg = name[:2]
        if leg in ("FL", "HR"):
            phase = phase_fl_hr
        else:
            phase = phase_fr_hl

        if "HipX" in name:
            targets.append(0.0)
        elif "HipY" in name:
            base = STAND_JOINT_ANGLES["HipY"]
            targets.append(base + swing_amp * np.sin(phase))
        elif "Knee" in name:
            base = STAND_JOINT_ANGLES["Knee"]
            # 摆动相（sin(phase) < 0）时抬膝
            lift = knee_lift * max(0.0, -np.sin(phase))
            targets.append(base - lift)
        else:
            targets.append(0.0)
    return targets


def apply_position_control(targets, kp=0.8, kd=0.05, max_force=25.0):
    for j, target in zip(joint_indices, targets):
        p.setJointMotorControl2(
            robot_id, j, p.POSITION_CONTROL,
            targetPosition=target,
            positionGain=kp,
            velocityGain=kd,
            force=max_force
        )


def apply_damping():
    for j in joint_indices:
        p.setJointMotorControl2(
            robot_id, j, p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=1.0
        )


# ============================================================
# 5. 实验辅助函数
# ============================================================
def set_gravity(g):
    p.setGravity(0, 0, g)


def set_foot_friction(mu):
    for link in foot_links:
        p.changeDynamics(robot_id, link, lateralFriction=mu)


def add_obstacle():
    """在机器人前方添加一个静态障碍物，用于外部碰撞实验"""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08, 0.25, 0.25])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.25, 0.25],
                              rgbaColor=[1, 0.2, 0.2, 1])
    obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=[0.9, 0, 0.25])
    return obs_id


obstacle_id = None


def print_base_state(step):
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    vel, _ = p.getBaseVelocity(robot_id)
    print(f"[step {step:05d}] base z={pos[2]:.3f}  "
          f"vx={vel[0]:.3f}  vy={vel[1]:.3f}")


# ============================================================
# 6. 主循环
# ============================================================
p.resetDebugVisualizerCamera(
    cameraDistance=1.5, cameraYaw=45, cameraPitch=-25,
    cameraTargetPosition=[0, 0, 0.3]
)

mode = 0
step = 0
start_time = time.time()

print("\n=== Lite3 PyBullet 实验平台 ===")
print("键盘：z 站立  c 行走  x 趴下  r 阻尼  wasd 移动  qe 旋转")
print("滑块：可实时调节重力、足端摩擦、控制模式、速度、步态参数")
print("碰撞实验：按 'o' 添加/移除障碍物（或使用滑块模式 1 + 前进速度撞向障碍物）\n")

while True:
    # ---- 读取滑块 ----
    g = p.readUserDebugParameter(slider_gravity)
    mu = p.readUserDebugParameter(slider_friction)
    mode_slider = int(p.readUserDebugParameter(slider_mode))
    fwd = p.readUserDebugParameter(slider_fwd)
    yaw_rate = p.readUserDebugParameter(slider_yaw)
    freq = p.readUserDebugParameter(slider_gait_freq)
    swing = p.readUserDebugParameter(slider_swing)

    set_gravity(g)
    set_foot_friction(mu)

    # ---- 键盘优先覆盖滑块模式 ----
    kb_mode = get_keyboard_command()
    mode = kb_mode if kb_mode is not None else mode_slider

    t = time.time() - start_time

    # ---- 控制逻辑 ----
    if mode == 0:  # 站立
        targets = get_standing_targets()
        apply_position_control(targets)
    elif mode == 1:  # 行走 / trot
        targets = get_trot_targets(t, freq, swing, GAIT_KNEE_LIFT)
        apply_position_control(targets)

        # 差速转向：左右腿摆动幅度微调
        if abs(yaw_rate) > 0.01:
            for j, target in zip(joint_indices, targets):
                name = p.getJointInfo(robot_id, j)[1].decode()
                leg = name[:2]
                if "HipY" in name:
                    side = 1.0 if leg in ("FL", "HL") else -1.0
                    p.setJointMotorControl2(
                        robot_id, j, p.POSITION_CONTROL,
                        targetPosition=target + side * yaw_rate * 0.1,
                        force=25.0
                    )

        # 前进速度：通过 base 速度约束或直接施加小推力
        if abs(fwd) > 0.01:
            p.applyExternalForce(
                robot_id, -1,
                forceObj=[fwd * 15.0, 0, 0],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME
            )
    elif mode == 2:  # 趴下
        targets = [0.0] * len(joint_indices)
        apply_position_control(targets)
    elif mode == 3:  # 阻尼
        apply_damping()

    # ---- 碰撞实验：键盘 'o' 切换障碍物 ----
    # 简单起见，用滑块模式 1 + 前进速度撞向固定障碍物；
    # 也可在 GUI 外手动调用 add_obstacle()

    p.stepSimulation()
    step += 1

    if step % 240 == 0:  # 每秒打印一次
        print_base_state(step)

    time.sleep(DT)