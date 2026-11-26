import pybullet as p
import pybullet_data
import numpy as np
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(
    SCRIPT_DIR,
    "Lite3_rl_deploy/third_party/deep_robotics_model/Lite3/Lite3_urdf/urdf/Lite3.urdf"
)

DT = 1.0 / 240.0
STAND_HIPX, STAND_HIPY, STAND_KNEE = 0.1, -0.985, 2.084
GAIT_KNEE_LIFT = 0.30

p.connect(p.GUI, options="--opengl2") 
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setTimeStep(DT)
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=1.0)

obs_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.3, 0.2])
obs_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.3, 0.2], rgbaColor=[1, 0, 0, 1])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obs_col, baseVisualShapeIndex=obs_vis, basePosition=[1.2, 0, 0.2])

robot_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0.28], flags=p.URDF_USE_SELF_COLLISION)

joint_indices, joint_names = [], []
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    if info[2] != p.JOINT_FIXED:
        joint_indices.append(j)
        joint_names.append(info[1].decode())

foot_links = []
for j in range(p.getNumJoints(robot_id)):
    link_name = p.getJointInfo(robot_id, j)[12].decode().lower()
    if "foot" in link_name or "toe" in link_name:
        foot_links.append(j)

def standing_targets():
    t = []
    for name in joint_names:
        leg = name[:2]
        if "HipX" in name:
            t.append(STAND_HIPX if leg in ("FL", "HL") else -STAND_HIPX)
        elif "HipY" in name:
            t.append(STAND_HIPY)
        elif "Knee" in name:
            t.append(STAND_KNEE)
        else:
            t.append(0.0)
    return t

def crouch_targets():
    t = []
    for name in joint_names:
        leg = name[:2]
        if "HipX" in name:
            t.append(STAND_HIPX if leg in ("FL", "HL") else -STAND_HIPX)
        elif "HipY" in name:
            t.append(-1.5)
        elif "Knee" in name:
            t.append(2.6)
        else:
            t.append(0.0)
    return t

def trot_targets(t, freq, vx):
    phase_fl_hr = 2 * np.pi * freq * t
    phase_fr_hl = phase_fl_hr + np.pi
    targets = []
    for name in joint_names:
        leg = name[:2]
        phase = phase_fl_hr if leg in ("FL", "HR") else phase_fr_hl

        if "HipX" in name:
            targets.append(STAND_HIPX if leg in ("FL", "HL") else -STAND_HIPX)
        elif "HipY" in name:
            targets.append(STAND_HIPY + 0.25 * np.sin(phase) + vx * 0.2)
        elif "Knee" in name:
            lift = GAIT_KNEE_LIFT * max(0.0, -np.sin(phase))
            targets.append(STAND_KNEE + lift)
        else:
            targets.append(0.0)
    return targets

def reset_robot():
    global current_mu, current_g, current_mode, push_force
    current_mu = 0.8
    current_g = -9.81
    current_mode = 0
    push_force = 0.0
    
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.28], [0, 0, 0, 1])
    p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
    for j, tg in zip(joint_indices, standing_targets()):
        p.resetJointState(robot_id, j, tg, targetVelocity=0)
    print("✨ 机器人已复位到初始状态！")

reset_robot()
p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0.6, 0, 0.2])

step = 0
start_time = time.time()
move_speed = 0.0

while True:
    keys = p.getKeyboardEvents()
    t = time.time() - start_time

    if p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN):
        move_speed = 1.0
    elif p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN):
        move_speed = -1.0
    else:
        move_speed = 0.0

    if ord('1') in keys and (keys[ord('1')] & p.KEY_WAS_TRIGGERED):
        current_mu = 0.01 if current_mu > 0.5 else 0.8
        print(f"⚙️ 摩擦系数改变为: {current_mu}")
        
    if ord('2') in keys and (keys[ord('2')] & p.KEY_WAS_TRIGGERED):
        current_g = 0.0 if current_g < -5 else -9.81
        print(f"⚙️ 重力改变为: {current_g}")
        
    if ord('3') in keys and (keys[ord('3')] & p.KEY_WAS_TRIGGERED):
        current_mode = (current_mode + 1) % 3
        mode_names = {0: "站立", 1: "阻尼", 2: "安全蹲伏"}
        print(f"⚙️ 模式切换为: {current_mode} ({mode_names[current_mode]})")
        
    # 【修改点 1】：将推力提高到 150N
    if ord(' ') in keys and (keys[ord(' ')] & p.KEY_WAS_TRIGGERED):
        push_force = 150.0
        print("💥 施加猛烈瞬间推力！")

    if ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED):
        reset_robot()

    p.setGravity(0, 0, current_g)
    for link in foot_links:
        p.changeDynamics(robot_id, link, lateralFriction=current_mu)

    if current_mode == 0:
        if abs(move_speed) > 0.1:
            targets = trot_targets(t, freq=2.0, vx=move_speed)
            p.applyExternalForce(robot_id, -1, forceObj=[move_speed * 30.0, 0, 0], posObj=[0,0,0], flags=p.LINK_FRAME)
        else:
            targets = standing_targets()
            
        for j, tg in zip(joint_indices, targets):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=tg, force=60.0)
            
    elif current_mode == 1:
        for j in joint_indices:
            p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0.1)
    elif current_mode == 2:
        for j, tg in zip(joint_indices, crouch_targets()):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=tg, force=30.0)

    # 【修改点 2】和【修改点 3】：施加力，作用于偏上方产生力矩，衰减变慢
    if abs(push_force) > 0.1:
        # 施加在 z=0.1 的高度（底盘偏上），产生向后的翻倒力矩
        p.applyExternalForce(robot_id, -1, forceObj=[push_force, 0, 0], posObj=[0, 0, 0.1], flags=p.LINK_FRAME)
        push_force *= 0.95 

    if step % 120 == 0 and current_mode == 0:
        contacts = p.getContactPoints(robot_id, plane_id)
        print(f"[step {step}] μ={current_mu:.2f} | g={current_g:.2f} | 接触对数={len(contacts)} | 速度={move_speed}")

    p.stepSimulation()
    step += 1
    time.sleep(DT)
    