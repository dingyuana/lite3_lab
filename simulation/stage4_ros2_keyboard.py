import os
import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

KEY_TIMEOUT = 1.0
PUBLISH_PERIOD = 0.05


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vx = 0.0
        self.wz = 0.0
        self.last_key_time = 0.0
        self._warned_no_subscriber = False
        self._reported_subscriber = False
        self._last_logged_command = None
        self.timer = self.create_timer(PUBLISH_PERIOD, self.timer_cb)
        self.get_logger().info(
            "键盘遥控已启动：W/S 前后，A/D 转向，空格停止。"
            "终端键盘自动重复期间持续运动，停止输入 1 秒后自动停车。"
        )

    def read_keys(self):
        while select.select([sys.stdin], [], [], 0)[0]:
            char = os.read(sys.stdin.fileno(), 1).decode(errors='ignore').lower()
            if not char:
                continue
            self.last_key_time = time.monotonic()
            if char == 'w':
                self.vx = 0.5
            elif char == 's':
                self.vx = -0.5
            elif char == 'a':
                self.wz = 0.5
            elif char == 'd':
                self.wz = -0.5
            elif char == ' ':
                self.vx, self.wz = 0.0, 0.0
                self.last_key_time = 0.0

    def timer_cb(self):
        self.read_keys()
        if self.last_key_time and time.monotonic() - self.last_key_time > KEY_TIMEOUT:
            self.vx, self.wz = 0.0, 0.0

        msg = Twist()
        msg.linear.x = self.vx
        msg.angular.z = self.wz
        self.pub.publish(msg)

        command = (msg.linear.x, msg.angular.z)
        if command != self._last_logged_command:
            self.get_logger().info(
                f"发送 /cmd_vel: linear.x={msg.linear.x:.2f}, "
                f"angular.z={msg.angular.z:.2f}"
            )
            self._last_logged_command = command

        has_subscriber = self.pub.get_subscription_count() > 0
        if not has_subscriber and not self._warned_no_subscriber:
            self.get_logger().warning(
                "没有 /cmd_vel 订阅者；请启动一个步态控制器。"
            )
            self._warned_no_subscriber = True
        elif has_subscriber:
            self._warned_no_subscriber = False
            if not self._reported_subscriber:
                self.get_logger().info("已连接 /cmd_vel 步态控制器。")
                self._reported_subscriber = True


def main(args=None):
    if not sys.stdin.isatty():
        raise RuntimeError("键盘遥控需要在交互式终端中运行。")

    stdin_fd = sys.stdin.fileno()
    original_terminal = termios.tcgetattr(stdin_fd)
    tty.setcbreak(stdin_fd)
    node = None
    try:
        rclpy.init(args=args)
        node = KeyboardTeleop()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, original_terminal)


if __name__ == '__main__':
    main()
