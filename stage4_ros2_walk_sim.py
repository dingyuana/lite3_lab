import rclpy
from rclpy.node import Node
from drdds.msg import JointsDataCmd
from geometry_msgs.msg import Twist
import numpy as np

CONTROL_PERIOD = 0.002


class CmdVelToJointCommand(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_joint_command')
        self.cmd_pub = self.create_publisher(JointsDataCmd, '/JOINTS_CMD', 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)

        self.vx = 0.0
        self.wz = 0.0
        self.t = 0.0
        self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)
        self.get_logger().info(
            "✅ /cmd_vel 到 /JOINTS_CMD 转换节点已启动（无 MuJoCo 窗口）。"
        )

    def cmd_vel_cb(self, msg):
        self.vx = msg.linear.x
        self.wz = msg.angular.z

    def control_loop(self):
        self.t += CONTROL_PERIOD
        cmd = JointsDataCmd()
        joints_list = cmd.data.joints_data

        # 基础站立姿态
        stand_hipy, stand_knee = -0.985, 2.084
        freq = 2.0
        swing_amp = 0.25

        # 如果速度指令不为 0，则开始计算步态相位
        if abs(self.vx) > 0.01 or abs(self.wz) > 0.01:
            phase = 2 * np.pi * freq * self.t
        else:
            phase = 0.0

        for i in range(12):
            leg = i // 3 # 0:FL, 1:FR, 2:HL, 3:HR
            joint_type = i % 3 # 0:HipX, 1:HipY, 2:Knee

            # 对角步态相位差
            p = phase if leg in (0, 3) else phase + np.pi

            if joint_type == 1: # HipY
                target_pos = stand_hipy + swing_amp * np.sin(p) + self.vx * 0.2
                if self.wz > 0: target_pos += 0.1 
                elif self.wz < 0: target_pos -= 0.1
            elif joint_type == 2: # Knee
                target_pos = stand_knee + 0.3 * max(0.0, -np.sin(p))
            else: # HipX
                target_pos = 0.1 if leg in (0, 2) else -0.1

            # 下发 MIT 控制参数
            joints_list[i].kp = 40.0
            joints_list[i].kd = 2.0
            joints_list[i].position = target_pos
            joints_list[i].torque = 0.0

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = CmdVelToJointCommand()
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