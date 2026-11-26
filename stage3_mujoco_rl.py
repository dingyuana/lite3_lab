import mujoco
import mujoco.viewer
import numpy as np
import time
import onnxruntime as ort
import threading

# ==========================================
# 1. 路径与模型配置
# ==========================================
XML_PATH = "/home/dingyan/lite3_lab/Lite3_rl_deploy/third_party/deep_robotics_model/Lite3/Lite3_mjcf/mjcf/Lite3.xml"
ONNX_PATH = "/home/dingyan/lite3_lab/Lite3_rl_deploy/policy/ppo/policy.onnx"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("正在加载 AI 策略...")
# 使用 CPU 推理
ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name
obs_dim = ort_session.get_inputs()[0].shape[1]
print(f"✅ AI 加载成功，需要观测维度: {obs_dim}")

# 获取 12 个关节在 qpos 和 qvel 中的索引
joint_ids_qpos = []
joint_ids_qvel = []
for i in range(model.nu):
    jid = model.actuator_trnid[i, 0]
    joint_ids_qpos.append(model.jnt_qposadr[jid])
    joint_ids_qvel.append(model.jnt_dofadr[jid])

# ==========================================
# 2. 初始化与控制变量
# ==========================================
# Lite3 默认站立关节角
default_qpos = np.zeros(12)
for i in range(12):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if "HipX" in name: default_qpos[i] = 0.1 if "FL" in name or "HL" in name else -0.1
    elif "HipY" in name: default_qpos[i] = -0.985
    elif "Knee" in name: default_qpos[i] = 2.084

action = np.zeros(12)       # 记录上一帧动作
cmd_vx, cmd_vy, cmd_vyaw = 0.0, 0.0, 0.0  # 速度指令
apply_push = False          # 是否施加外力
exit_flag = False

def reset_sim():
    global action, apply_push
    mujoco.mj_resetData(model, data)
    for i, qadr in enumerate(joint_ids_qpos):
        data.qpos[qadr] = default_qpos[i]
    data.qpos[2] = 0.35  # 稍微抬高，自然落地
    mujoco.mj_forward(model, data)
    action = np.zeros(12)
    apply_push = False
    print("✨ 机器人已复位！")

reset_sim()

# ==========================================
# 3. 终端输入监听 (控制移动与扰动)
# ==========================================
def input_listener():
    global cmd_vx, cmd_vy, cmd_vyaw, apply_push, exit_flag
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

# ==========================================
# 4. 主循环 (修复观测向量)
# ==========================================
print("\n==================================================")
print("🎮 终端操作指南（输入字母并回车）：")
print("输入 'w' 前进 / 's' 后退 / 'a' 左转 / 'd' 右转")
print("输入 'x' 停止 / 'p' 施加 150N 推力！ / 'r' 复位 / 'q' 退出")
print("==================================================\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running() and not exit_flag:
        # --- 1. 获取物理状态 ---
        qpos = np.array([data.qpos[qadr] for qadr in joint_ids_qpos])
        qvel = np.array([data.qvel[qadr] for qadr in joint_ids_qvel])
        
        # 获取底盘姿态 (四元数 qpos[3:7])
        quat = data.qpos[3:7]
        
        # 将四元数转为重力在机体坐标系的投影 (projected gravity)
        qw, qx, qy, qz = quat
        gravity = np.array([
            2 * (qx*qz - qw*qy),
            2 * (qy*qz + qw*qx),
            1 - 2 * (qx*qx + qy*qy)
        ])

        # --- 2. 拼装 45 维观测向量 (严格按照 legged_gym 规范) ---
        obs = np.zeros(obs_dim, dtype=np.float32)
        
        # 前 9 维通常是：角速度(3), 重力(3), 指令(3)
        # 底盘角速度 (通常带缩放 0.25)
        obs[0:3] = data.qvel[0:3] * 0.25   
        obs[3:6] = gravity                 
        obs[6:9] = [cmd_vx, cmd_vy, cmd_vyaw]
        
        # 后续 36 维：关节位置偏差(12), 关节速度缩放(12), 上一帧动作(12)
        obs[9:21]  = qpos - default_qpos   
        obs[21:33] = qvel * 0.05           
        obs[33:45] = action                

        # --- 3. AI 推理 ---
        ai_action = ort_session.run(None, {input_name: obs.reshape(1, -1)})[0][0]
        action = ai_action
        
        # --- 4. 下发控制指令 ---
        # 策略输出的是目标关节角的增量，缩放系数通常为 0.25
        target_qpos = default_qpos + ai_action * 0.25
        data.ctrl[:] = target_qpos

        # --- 5. 施加外力 (碰撞实验) ---
        if apply_push:
            data.xfrc_applied[1, 0] = 150.0
            apply_push = False
        else:
            data.xfrc_applied[1, 0] *= 0.9

        # --- 6. 物理步进 ---
        mujoco.mj_step(model, data)
        viewer.sync()
        
        if step % 120 == 0:
            print(f"[step {step}] Z高度={data.qpos[2]:.2f} | VX={cmd_vx:.1f}")
            
        step += 1
        time.sleep(model.opt.timestep)