import pybullet as p
import time

URDF_PATH = "/home/dingyan/deep_robotics_model/Lite3/Lite3_urdf/urdf/Lite3.urdf"
# ↑ 改成你的实际绝对路径

p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

plane = p.loadURDF("plane.urdf")

robot = p.loadURDF(
    URDF_PATH,
    basePosition=[0, 0, 0.45],
    flags=p.URDF_USE_SELF_COLLISION
)

num_joints = p.getNumJoints(robot)
print(f"Lite3 加载成功，关节数：{num_joints}")

# 打印关节名称，确认命名规范
for j in range(num_joints):
    info = p.getJointInfo(robot, j)
    print(f"  joint {j}: {info[1].decode()}")

for _ in range(2400):
    p.stepSimulation()
    time.sleep(1. / 240.)

p.disconnect()