import numpy as np

from project_paths import LITE3_XML, POLICY_ONNX

XML_PATH = str(LITE3_XML)
ONNX_PATH = str(POLICY_ONNX)

NUM_JOINTS = 12
OBS_DIM = 45
ACTION_SCALE = 0.25
OMEGA_SCALE = 0.25
DOF_VEL_SCALE = 0.05
GRAVITY_DIRECTION = np.array([0.0, 0.0, -1.0])
DEFAULT_JOINT_PATTERN = np.array([0.0, -0.65, 1.3])  # HipX, HipY, Knee


def projected_gravity(base_quat):
    """Return R^T * (0, 0, -1) for a MuJoCo (w, x, y, z) base quaternion."""
    qw, qx, qy, qz = np.asarray(base_quat, dtype=np.float64).reshape(4)
    # Body-to-world rotation matrix, matching MuJoCo's quaternion convention.
    rotation = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])
    return rotation.T @ GRAVITY_DIRECTION


def default_joint_positions():
    """Return the deploy policy's default joint positions in RL joint order."""
    return np.tile(DEFAULT_JOINT_PATTERN, 4).astype(np.float64)


def build_observation(base_angular_velocity, gravity_projected, command_velocity,
                      qpos, qvel, last_action):
    """Build the 45-dimensional observation without requiring MuJoCo."""
    default_qpos = default_joint_positions()
    components = [
        np.asarray(base_angular_velocity, dtype=np.float64).reshape(-1) * OMEGA_SCALE,
        np.asarray(gravity_projected, dtype=np.float64).reshape(-1),
        np.asarray(command_velocity, dtype=np.float64).reshape(-1),
        np.asarray(qpos, dtype=np.float64).reshape(-1) - default_qpos,
        np.asarray(qvel, dtype=np.float64).reshape(-1) * DOF_VEL_SCALE,
        np.asarray(last_action, dtype=np.float64).reshape(-1),
    ]
    expected_sizes = (3, 3, 3, NUM_JOINTS, NUM_JOINTS, NUM_JOINTS)
    for index, (component, expected_size) in enumerate(zip(components, expected_sizes)):
        if component.size != expected_size:
            raise ValueError(f"Observation component {index} must have {expected_size} values")
    return np.concatenate(components).astype(np.float32, copy=False)


def pd_torque_control(target_qpos, qpos, qvel, ctrlrange=None, kp=30.0, kd=1.0):
    """Compute motor torque, optionally clipped to per-actuator ctrlrange."""
    target_qpos = np.asarray(target_qpos, dtype=np.float64)
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    torque = kp * (target_qpos - qpos) + kd * (0.0 - qvel)
    if ctrlrange is not None:
        limits = np.asarray(ctrlrange, dtype=np.float64)
        torque = np.clip(torque, limits[:, 0], limits[:, 1])
    return torque


def main():
    # Heavy simulator/inference imports stay out of module import time so the
    # NumPy-only control helpers can be tested without opening MuJoCo.
    import mujoco
    import mujoco.viewer
    import onnxruntime as ort
    import threading
    import time

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("正在加载 AI 策略...")
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    obs_dim = ort_session.get_inputs()[0].shape[1]
    if obs_dim != OBS_DIM:
        raise ValueError(f"策略需要 {obs_dim} 维观测，当前实现要求 {OBS_DIM} 维")
    print(f"✅ AI 加载成功，需要观测维度: {obs_dim}")

    # 获取 12 个关节在 qpos 和 qvel 中的索引
    joint_ids_qpos = []
    joint_ids_qvel = []
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        joint_ids_qpos.append(model.jnt_qposadr[jid])
        joint_ids_qvel.append(model.jnt_dofadr[jid])

    default_qpos = default_joint_positions()
    action = np.zeros(NUM_JOINTS, dtype=np.float32)
    cmd_vx, cmd_vy, cmd_vyaw = 0.0, 0.0, 0.0
    apply_push = False
    exit_flag = False

    def reset_sim():
        nonlocal action, apply_push
        mujoco.mj_resetData(model, data)
        for i, qadr in enumerate(joint_ids_qpos):
            data.qpos[qadr] = default_qpos[i]
        data.qpos[2] = 0.35
        mujoco.mj_forward(model, data)
        action = np.zeros(NUM_JOINTS, dtype=np.float32)
        apply_push = False
        print("✨ 机器人已复位！")

    reset_sim()

    # 终端输入监听 (控制移动与扰动)
    def input_listener():
        nonlocal cmd_vx, cmd_vy, cmd_vyaw, apply_push, exit_flag
        while not exit_flag:
            try:
                cmd = input().strip().lower()
                if cmd == 'w': cmd_vx = 0.6; print("👉 指令: 前进")
                elif cmd == 's': cmd_vx = -0.4; print("👉 指令: 后退")
                elif cmd == 'a': cmd_vyaw = 0.6; print("👉 指令: 左转")
                elif cmd == 'd': cmd_vyaw = -0.6; print("👉 指令: 右转")
                elif cmd == 'x': cmd_vx, cmd_vyaw = 0.0, 0.0; print("👉 指令: 停止")
                elif cmd == 'p': apply_push = True; print("💥 施加 150N 侧向推力！")
                elif cmd == 'r': reset_sim()
                elif cmd == 'q': exit_flag = True; break
            except EOFError:
                break

    threading.Thread(target=input_listener, daemon=True).start()

    print("\n==================================================")
    print("🎮 终端操作指南（输入字母并回车）：")
    print("输入 'w' 前进 / 's' 后退 / 'a' 左转 / 'd' 右转")
    print("输入 'x' 停止 / 'p' 施加 150N 推力！ / 'r' 复位 / 'q' 退出")
    print("==================================================\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running() and not exit_flag:
            qpos = data.qpos[joint_ids_qpos]
            qvel = data.qvel[joint_ids_qvel]
            gravity = projected_gravity(data.qpos[3:7])

            # 角速度(3), 投影重力(3), 指令(3), 关节位置偏差(12),
            # 关节速度(12), 上一帧动作(12)
            obs = build_observation(
                data.qvel[0:3], gravity, [cmd_vx, cmd_vy, cmd_vyaw],
                qpos, qvel, action,
            )

            ai_action = ort_session.run(None, {input_name: obs.reshape(1, -1)})[0][0]
            action = np.asarray(ai_action, dtype=np.float32).reshape(-1)

            target_qpos = default_qpos + action * ACTION_SCALE
            # Lite3.xml 的执行器是电机，需要下发力矩而不是目标关节位置。
            data.ctrl[:] = pd_torque_control(
                target_qpos, qpos, qvel, model.actuator_ctrlrange,
            )

            if apply_push:
                data.xfrc_applied[1, 0] = 150.0
                apply_push = False
            else:
                data.xfrc_applied[1, 0] *= 0.9

            mujoco.mj_step(model, data)
            viewer.sync()

            if step % 120 == 0:
                print(f"[step {step}] Z高度={data.qpos[2]:.2f} | VX={cmd_vx:.1f}")

            step += 1
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
