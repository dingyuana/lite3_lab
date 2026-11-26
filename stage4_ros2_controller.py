import rclpy
from rclpy.node import Node
from drdds.msg import JointsDataCmd
from geometry_msgs.msg import Twist
import numpy as np

class GaitController(Node):
    def __init__(self):
        super().__init__('gait_controller')
        self.cmd_pub = self.create_publisher(JointsDataCmd, '/JOINTS_CMD', 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)
        
        self.vx = 0.0
        self.wz = 0.0
        self.t = 0.0
        self._last_logged_twist = None
        self._log_ticks = 0
        
        # 50Hz 的频率更新关节指令（足够驱动仿真）
        self.timer = self.create_timer(0.02, self.timer_cb)
        self.get_logger().info("✅ 步态控制器已启动，正在监听 /cmd_vel...")

    def cmd_vel_cb(self, msg):
        self.vx = msg.linear.x
        self.wz = msg.angular.z
        command = (self.vx, self.wz)
        if command != self._last_logged_twist:
            self.get_logger().info(
                f"收到 /cmd_vel: linear.x={self.vx:.2f}, "
                f"angular.z={self.wz:.2f}"
            )
            self._last_logged_twist = command

    def timer_cb(self):
        self.t += 0.02
        cmd = JointsDataCmd()
        joints_list = cmd.data.joints_data
        
        # Lite3 基础站立姿态
        stand_hipy, stand_knee = -0.985, 2.084
        freq = 2.0
        swing_amp = 0.25
        
        # 停止时回到基础站姿；有运动指令时才运行对角步态。
        if abs(self.vx) < 0.01 and abs(self.wz) < 0.01:
            self.t = 0.0
            phase = 0.0
        else:
            phase = 2 * np.pi * freq * self.t

        for i in range(12):
            leg = i // 3      # 0:FL, 1:FR, 2:HL, 3:HR
            joint_type = i % 3 # 0:HipX, 1:HipY, 2:Knee
            
            # 对角步态：左前(0)和右后(3)同相，右前(1)和左后(2)反相
            p = phase if leg in (0, 3) else phase + np.pi

            if joint_type == 1: # HipY (大腿)
                target_pos = stand_hipy + swing_amp * np.sin(p) + self.vx * 0.2
                if self.wz != 0.0:
                    side = 1.0 if leg in (0, 2) else -1.0
                    target_pos += side * self.wz * 0.1
            elif joint_type == 2: # Knee (小腿)
                # The model's positive HipY direction is opposite to forward
                # motion, so lift the legs on the opposite half-cycle to walk forward.
                lift_phase = np.sin(p) if self.vx >= 0.0 else -np.sin(p)
                target_pos = stand_knee + 0.3 * max(0.0, lift_phase)
            else: # HipX (侧摆)
                target_pos = 0.1 if leg in (0, 2) else -0.1

            # 下发 MIT 参数，用位置控制（kp=40）跟踪目标
            joints_list[i].kp = 40.0
            joints_list[i].kd = 2.0
            joints_list[i].position = target_pos
            joints_list[i].torque = 0.0
            
        self.cmd_pub.publish(cmd)
        self._log_ticks += 1
        if self._log_ticks >= 50:
            positions = ', '.join(
                f'{joint.position:.2f}' for joint in joints_list[:12]
            )
            self.get_logger().info(
                f"发送 /JOINTS_CMD: 前12个关节 position=[{positions}], "
                "kp=40.0, kd=2.0"
            )
            self._log_ticks = 0

def main(args=None):
    rclpy.init(args=args)
    node = GaitController()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()