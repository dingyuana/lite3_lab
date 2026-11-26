# import rclpy
# from rclpy.node import Node
# from drdds.msg import JointsDataCmd, JointsData, ImuData
# import mujoco
# import mujoco.viewer
# import numpy as np
# import time
# import threading

# XML_PATH = "/home/dingyan/lite3_lab/Lite3_rl_deploy/third_party/deep_robotics_model/Lite3/Lite3_mjcf/mjcf/Lite3.xml"

# class MujocoSimNode(Node):
#     def __init__(self):
#         super().__init__('mujoco_simulation')
#         self.model = mujoco.MjModel.from_xml_path(XML_PATH)
#         self.data = mujoco.MjData(self.model)
        
#         self.cmd_sub = self.create_subscription(JointsDataCmd, '/JOINTS_CMD', self.cmd_cb, 10)
#         self.joint_pub = self.create_publisher(JointsData, '/JOINTS_DATA', 10)
        
#         # 初始化控制参数 (MIT模式)
#         self.kp_cmd = np.zeros(12)
#         self.kd_cmd = np.zeros(12)
#         self.pos_cmd = np.zeros(12)
#         self.vel_cmd = np.zeros(12)
#         self.tau_ff = np.zeros(12)
        
#         self.get_logger().info("✅ 自定义 MuJoCo 仿真节点已启动！")

#     def cmd_cb(self, msg):
#         # 提取 12 个关节的控制指令
#         for i in range(12):
#             self.kp_cmd[i] = msg.data.joints_data[i].kp
#             self.kd_cmd[i] = msg.data.joints_data[i].kd
#             self.pos_cmd[i] = msg.data.joints_data[i].position
#             self.vel_cmd[i] = msg.data.joints_data[i].velocity
#             self.tau_ff[i] = msg.data.joints_data[i].torque

#     def publish_state(self):
#         msg = JointsData()
#         # 构造标准的 feedback 数据
#         for i in range(12):
#             joint_data = msg.data.joints_data[i]
#             joint_data.position = self.data.qpos[7 + i]
#             joint_data.velocity = self.data.qvel[6 + i]
#             # 仿真中读取不到真实力矩，用 0 代替
#             joint_data.torque = 0.0 
#         self.joint_pub.publish(msg)

# def ros2_spin_thread(node):
#     rclpy.spin(node)

# def main():
#     rclpy.init()
#     sim_node = MujocoSimNode()
    
#     # 启动一个独立的线程，专门去接收 ROS2 的指令，绝不被渲染卡死
#     thread = threading.Thread(target=ros2_spin_thread, args=(sim_node,), daemon=True)
#     thread.start()

#     # 主线程专门跑 MuJoCo 渲染和物理计算
#     with mujoco.viewer.launch_passive(sim_node.model, sim_node.data) as viewer:
#         step = 0
#         while viewer.is_running():
#             q = sim_node.data.qpos[7:19]
#             dq = sim_node.data.qvel[6:18]
            
#             # 根据收到的指令计算力矩
#             tau = sim_node.kp_cmd * (sim_node.pos_cmd - q) + \
#                   sim_node.kd_cmd * (sim_node.vel_cmd - dq) + \
#                   sim_node.tau_ff
                  
#             sim_node.data.ctrl[:] = tau
#             mujoco.mj_step(sim_node.model, sim_node.data)
            
#             if step % 5 == 0:
#                 sim_node.publish_state()
                
#             viewer.sync()
#             step += 1
#             time.sleep(0.002)
            
#     sim_node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from drdds.msg import JointsDataCmd, JointsData
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import os

XML_PATH = os.path.expanduser("~/lite3_lab/Lite3_rl_deploy/third_party/deep_robotics_model/Lite3/Lite3_mjcf/mjcf/Lite3.xml")

class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_simulation')
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        self.cmd_sub = self.create_subscription(JointsDataCmd, '/JOINTS_CMD', self.cmd_cb, 10)
        self.joint_pub = self.create_publisher(JointsData, '/JOINTS_DATA', 10)

        self.kp_cmd = np.zeros(12)
        self.kd_cmd = np.zeros(12)
        self.pos_cmd = np.zeros(12)
        self.vel_cmd = np.zeros(12)
        self.tau_ff = np.zeros(12)
        self.get_logger().info("✅ 自定义 MuJoCo 仿真节点已启动！")

    def cmd_cb(self, msg):
        for i in range(12):
            self.kp_cmd[i] = msg.data.joints_data[i].kp
            self.kd_cmd[i] = msg.data.joints_data[i].kd
            self.pos_cmd[i] = msg.data.joints_data[i].position
            self.vel_cmd[i] = msg.data.joints_data[i].velocity
            self.tau_ff[i] = msg.data.joints_data[i].torque

    def publish_state(self):
        msg = JointsData()
        for i in range(12):
            msg.data.joints_data[i].position = self.data.qpos[7 + i]
            msg.data.joints_data[i].velocity = self.data.qvel[6 + i]
        self.joint_pub.publish(msg)

def ros2_spin_thread(node):
    rclpy.spin(node)

def main():
    rclpy.init()
    sim_node = MujocoSimNode()
    threading.Thread(target=ros2_spin_thread, args=(sim_node,), daemon=True).start()

    with mujoco.viewer.launch_passive(sim_node.model, sim_node.data) as viewer:
        step = 0
        while viewer.is_running():
            q = sim_node.data.qpos[7:19]
            dq = sim_node.data.qvel[6:18]

            # 核心公式：MIT 模式下的力矩计算
            tau = sim_node.kp_cmd * (sim_node.pos_cmd - q) + \
                  sim_node.kd_cmd * (sim_node.vel_cmd - dq) + \
                  sim_node.tau_ff

            sim_node.data.ctrl[:] = tau
            mujoco.mj_step(sim_node.model, sim_node.data)

            if step % 5 == 0:
                sim_node.publish_state()

            viewer.sync()
            step += 1
            time.sleep(0.002)

    sim_node.destroy_node()
    if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()