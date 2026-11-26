# import rclpy
# from rclpy.node import Node
# from drdds.msg import JointsDataCmd, JointsData
# import numpy as np

# class TorqueController(Node):
#     def __init__(self):
#         super().__init__('torque_experiment_node')
#         self.cmd_pub = self.create_publisher(JointsDataCmd, '/JOINTS_CMD', 10)
#         self.data_sub = self.create_subscription(JointsData, '/JOINTS_DATA', self.data_cb, 10)
#         self.timer = self.create_timer(0.005, self.control_loop)
#         self.step = 0
#         self.latest_data = None
#         self.get_logger().info("✅ 纯力矩控制节点已启动！(修正版)")

#     def data_cb(self, msg):
#         self.latest_data = msg

#     def control_loop(self):
#         cmd = JointsDataCmd()
#         self.step += 1
#         joints_list = cmd.data.joints_data
        
#         # 获取当前时间，用于生成正弦波
#         t = self.step * 0.005 
        
#         for i in range(12):
#             # 彻底关闭位置环和速度环
#             joints_list[i].kp = 0.0
#             joints_list[i].kd = 0.0
#             joints_list[i].position = 0.0
#             joints_list[i].velocity = 0.0
            
#             if self.step > 200:
#                 # 把固定力矩改成正弦波，制造持续的“触电抽搐”
#                 if i % 3 == 1:   joints_list[i].torque = -10.0 + 15.0 * np.sin(t * 10)
#                 elif i % 3 == 2: joints_list[i].torque = 15.0 + 20.0 * np.sin(t * 10)
#             else:
#                 joints_list[i].torque = 0.0
                
#         self.cmd_pub.publish(cmd)
        
#         # 正确的日志打印
#         if self.step % 200 == 0 and self.latest_data:
#             real_pos = self.latest_data.data.joints_data[1].position
#             self.get_logger().info(f"Step: {self.step} | HipY 实际角度: {real_pos:.3f}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = TorqueController()
#     try: 
#         rclpy.spin(node)
#     except KeyboardInterrupt: 
#         pass
#     finally:
#         node.destroy_node()
#         if rclpy.ok(): 
#             rclpy.shutdown()
            
# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from drdds.msg import JointsDataCmd, JointsData
import numpy as np

class TorqueController(Node):
    def __init__(self):
        super().__init__('torque_experiment_node')
        self.cmd_pub = self.create_publisher(JointsDataCmd, '/JOINTS_CMD', 10)
        self.data_sub = self.create_subscription(JointsData, '/JOINTS_DATA', self.data_cb, 10)
        self.timer = self.create_timer(0.005, self.control_loop)
        self.step = 0
        self.latest_data = None
        self.get_logger().info("✅ 纯力矩控制节点已启动！(修正版)")

    def data_cb(self, msg):
        self.latest_data = msg

    def control_loop(self):
        cmd = JointsDataCmd()
        self.step += 1
        joints_list = cmd.data.joints_data

        # 获取当前时间，用于生成正弦波
        t = self.step * 0.005

        for i in range(12):
            # 彻底关闭位置环和速度环
            joints_list[i].kp = 0.0
            joints_list[i].kd = 0.0
            joints_list[i].position = 0.0
            joints_list[i].velocity = 0.0

            if self.step > 200:
                # 把固定力矩改成正弦波，制造持续的“触电抽搐”
                if i % 3 == 1:   joints_list[i].torque = -10.0 + 15.0 * np.sin(t * 10)
                elif i % 3 == 2: joints_list[i].torque = 15.0 + 20.0 * np.sin(t * 10)
            else:
                joints_list[i].torque = 0.0

        self.cmd_pub.publish(cmd)

        # 日志打印
        if self.step % 200 == 0 and self.latest_data:
            real_pos = self.latest_data.data.joints_data[1].position
            self.get_logger().info(f"Step: {self.step} | HipY 实际角度: {real_pos:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = TorqueController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()