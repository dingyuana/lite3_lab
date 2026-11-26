import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from drdds.msg import JointsData, JointsDataCmd
from rclpy.node import Node

from project_paths import LITE3_XML

XML_PATH = str(LITE3_XML)
COMMAND_TIMEOUT_S = 0.5


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__("mujoco_simulation")
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.cmd_sub = self.create_subscription(
            JointsDataCmd, "/JOINTS_CMD", self.cmd_cb, 10)
        self.joint_pub = self.create_publisher(JointsData, "/JOINTS_DATA", 10)
        self._command = np.zeros((self.model.nu, 5), dtype=np.float64)
        self._command_lock = threading.Lock()
        self._last_cmd_time = time.monotonic()
        self.get_logger().info(
            "MuJoCo 仿真节点已启动；无效指令将被拒绝，"
            f"{COMMAND_TIMEOUT_S:.1f}s 无新 /JOINTS_CMD 时输出零力矩。"
        )

    def cmd_cb(self, msg):
        joints_data = msg.data.joints_data
        if len(joints_data) != self.model.nu:
            self.get_logger().warning(
                f"拒绝关节指令：需要 {self.model.nu} 个关节，收到 {len(joints_data)} 个"
            )
            return
        try:
            command = np.array([
                [joint.kp, joint.kd, joint.position, joint.velocity, joint.torque]
                for joint in joints_data
            ], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"拒绝关节指令：无法解析数值 ({exc})")
            return
        if not np.all(np.isfinite(command)):
            self.get_logger().warning(
                "拒绝关节指令：增益、位置、速度或力矩包含非有限值"
            )
            return
        with self._command_lock:
            self._command = command
            self._last_cmd_time = time.monotonic()

    def publish_state(self):
        msg = JointsData()
        for i in range(self.model.nu):
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
            with sim_node._command_lock:
                command = sim_node._command.copy()
                last_cmd_time = sim_node._last_cmd_time

            if time.monotonic() - last_cmd_time > COMMAND_TIMEOUT_S:
                tau = np.zeros_like(q)
            else:
                kp_cmd, kd_cmd, pos_cmd, vel_cmd, tau_ff = command.T
                tau = kp_cmd * (pos_cmd - q) + kd_cmd * (vel_cmd - dq) + tau_ff

            sim_node.data.ctrl[:] = tau
            mujoco.mj_step(sim_node.model, sim_node.data)
            if step % 5 == 0:
                sim_node.publish_state()
            viewer.sync()
            step += 1
            time.sleep(0.002)

    sim_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
