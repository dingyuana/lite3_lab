import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import queue

from project_paths import LITE3_XML

XML_PATH = str(LITE3_XML)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

STAND_HIPX, STAND_HIPY, STAND_KNEE = 0.1, -0.7, 1.2
POSITION_KP, POSITION_KD = 40.0, 2.0
PUSH_FORCE = 150.0
PUSH_DURATION = 0.15
TORSO_BODY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "TORSO")

print(f"✅ MuJoCo 模型加载成功！执行器数量: {model.nu}")

# ==========================================
# 2. 状态变量与控制函数
# ==========================================
current_mu = 0.8
current_g = -9.81
current_mode = 0  # 0站立, 1安全蹲伏
push_steps_remaining = 0
exit_flag = False
command_queue = queue.Queue()

def joint_position_ctrl(hip_y, knee):
    ctrl = np.zeros(model.nu)
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not act_name: continue
        if "FL_HipX" in act_name or "HL_HipX" in act_name:
            target = STAND_HIPX
        elif "FR_HipX" in act_name or "HR_HipX" in act_name:
            target = -STAND_HIPX
        elif "HipY" in act_name:
            target = hip_y
        elif "Knee" in act_name:
            target = knee
        else:
            continue

        joint_id = model.actuator_trnid[i, 0]
        qpos_addr = model.jnt_qposadr[joint_id]
        dof_addr = model.jnt_dofadr[joint_id]
        ctrl[i] = POSITION_KP * (target - data.qpos[qpos_addr]) - POSITION_KD * data.qvel[dof_addr]
    return ctrl

def standing_ctrl():
    return joint_position_ctrl(STAND_HIPY, STAND_KNEE)

def crouch_ctrl():
    return joint_position_ctrl(-1.5, 2.6)

def reset_sim():
    global current_mu, current_g, current_mode, push_steps_remaining
    current_mu = 0.8
    current_g = -9.81
    current_mode = 0
    push_steps_remaining = 0
    
    for g in range(model.ngeom):
        model.geom_friction[g, 0] = current_mu
    model.opt.gravity[2] = current_g
    
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.35
    mujoco.mj_forward(model, data)
    print("✨ 机器人已复位！")

# ==========================================
# 3. 终端输入监听线程
# ==========================================
def input_listener():
    print("\n等待终端输入...")
    while not exit_flag:
        try:
            cmd = input().strip().lower()
            if cmd:
                command_queue.put(cmd)
        except EOFError:
            break

def viewer_key_callback(key):
    if 65 <= key <= 90:
        command_queue.put(chr(key).lower())

# 启动终端输入线程
thread = threading.Thread(target=input_listener, daemon=True)
thread.start()

reset_sim()

# ==========================================
# 4. 主循环
# ==========================================
print("\n==================================================")
print("🎮 操作指南（在查看器中按键，或在终端输入字母并回车）：")
print("'f'：切换摩擦 | 'g'：切换重力 | 'v'：站立 / 安全蹲伏")
print("'p'：施加 150N 推力 | 'r'：一键复位 | 'q'：退出程序")
print("==================================================\n")

with mujoco.viewer.launch_passive(model, data, key_callback=viewer_key_callback) as viewer:
    step = 0
    while viewer.is_running() and not exit_flag:
        try:
            cmd = command_queue.get_nowait()
        except queue.Empty:
            cmd = None

        if cmd == 'f':
            current_mu = 0.02 if current_mu > 0.5 else 0.8
            for geom_id in range(model.ngeom):
                model.geom_friction[geom_id, 0] = current_mu
            print(f"⚙️ 摩擦系数改变为: {current_mu}")
        elif cmd == 'g':
            current_g = 0.0 if current_g < -5 else -9.81
            model.opt.gravity[2] = current_g
            print(f"⚙️ 重力改变为: {current_g}")
        elif cmd == 'v':
            current_mode = (current_mode + 1) % 2
            mode_names = {0: "站立", 1: "安全蹲伏"}
            print(f"⚙️ 模式切换为: {current_mode} ({mode_names[current_mode]})")
        elif cmd == 'p':
            push_steps_remaining = max(1, round(PUSH_DURATION / model.opt.timestep))
            print(f"💥 施加 {PUSH_FORCE:.0f}N 推力，持续 {PUSH_DURATION:.2f} 秒！")
        elif cmd == 'r':
            reset_sim()
        elif cmd == 'q':
            exit_flag = True
            print("退出中...")
            break

        if current_mode == 0:
            data.ctrl[:] = standing_ctrl()
        elif current_mode == 1:
            data.ctrl[:] = crouch_ctrl()

        if push_steps_remaining > 0:
            data.xfrc_applied[TORSO_BODY_ID, 0] = PUSH_FORCE
            push_steps_remaining -= 1
        else:
            data.xfrc_applied[TORSO_BODY_ID, :] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()
        
        if step % 120 == 0:
            print(f"[step {step}] μ={current_mu:.2f} | g={current_g:.2f} | 接触对={data.ncon} | 模式={current_mode}")
            
        step += 1
        time.sleep(model.opt.timestep)