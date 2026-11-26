"""LiDAR/SLAM-data-frontend simulation using ground-truth odometry.

Publishes simulated LiDAR points, ground-truth odometry/TF and a body marker.
This node supplies inputs for an external SLAM system; it performs no mapping,
localization or loop closure, and its odometry is not a SLAM estimate.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker  # 【新增】引入 Marker 用于显示狗的身体
import tf2_ros
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import math
import os
import select
import sys
import tty

from project_paths import LITE3_XML

XML_PATH = str(LITE3_XML)

class LidarFrontendSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_lidar_frontend')
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        
        self.cloud_pub = self.create_publisher(PointCloud2, '/lidar_points', 10)
        self.odom_pub = self.create_publisher(Odometry, '/ground_truth_odom', 10)
        
        # 【新增】发布代表机器狗身体的 Marker 话题
        self.marker_pub = self.create_publisher(Marker, '/dog_body', 10)
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        self.vx, self.wz = 0.0, 0.0
        self.gait_t = 0.0
        
        self.get_logger().info(
            "✅ LiDAR/SLAM 数据前端仿真节点已启动（使用真值里程计，不执行建图、定位或回环）！"
        )
        
    def process_keyboard(self, char):
        if char == 'w': self.vx = 0.3
        elif char == 's': self.vx = -0.3
        elif char == 'a': self.wz = 0.3
        elif char == 'd': self.wz = -0.3
        elif char == ' ': self.vx, self.wz = 0.0, 0.0

    def publish_lidar_and_odom(self):
        now = self.get_clock().now().to_msg()
        
        # 1. 发布真值里程计（SLAM 前端输入，不是定位估计）
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "world"
        odom.pose.pose.position.x = self.data.qpos[0]
        odom.pose.pose.position.y = self.data.qpos[1]
        odom.pose.pose.position.z = self.data.qpos[2]
        odom.pose.pose.orientation.w = self.data.qpos[3]
        odom.pose.pose.orientation.x = self.data.qpos[4]
        odom.pose.pose.orientation.y = self.data.qpos[5]
        odom.pose.pose.orientation.z = self.data.qpos[6]
        self.odom_pub.publish(odom)

        # 2. 发布真值 TF（用于可视化和对比，不是 SLAM 估计）
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.data.qpos[0]
        t.transform.translation.y = self.data.qpos[1]
        t.transform.translation.z = self.data.qpos[2]
        t.transform.rotation.w = self.data.qpos[3]
        t.transform.rotation.x = self.data.qpos[4]
        t.transform.rotation.y = self.data.qpos[5]
        t.transform.rotation.z = self.data.qpos[6]
        self.tf_broadcaster.sendTransform(t)

        # 3. 【新增】发布一个红色的方块代表狗的身体
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = now
        marker.ns = "dog"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        # 位置和姿态与底盘一致
        marker.pose.position.x = self.data.qpos[0]
        marker.pose.position.y = self.data.qpos[1]
        marker.pose.position.z = self.data.qpos[2]
        marker.pose.orientation.w = self.data.qpos[3]
        marker.pose.orientation.x = self.data.qpos[4]
        marker.pose.orientation.y = self.data.qpos[5]
        marker.pose.orientation.z = self.data.qpos[6]
        # 尺寸
        marker.scale.x = 0.4
        marker.scale.y = 0.2
        marker.scale.z = 0.1
        # 颜色：红色
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        self.marker_pub.publish(marker)

        # 4. 模拟 3D 雷达
        points = []
        lidar_origin = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2] + 0.2])
        
        # 【进阶实验】：取消注释以模拟运动畸变
        # lidar_origin += np.random.normal(0, 0.03, 3)
        
        quat = np.array([self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]])
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)

        for v_angle in [-15, -5, 5, 15]:
            for h_angle in range(0, 360, 10):
                rad_h, rad_v = math.radians(h_angle), math.radians(v_angle)
                local_dir = np.array([math.cos(rad_v)*math.cos(rad_h), math.cos(rad_v)*math.sin(rad_h), math.sin(rad_v)])
                world_dir = rot_mat.dot(local_dir)
                
                geom_id = np.zeros(1, dtype=np.int32)
                dist = mujoco.mj_ray(self.model, self.data, lidar_origin, world_dir, None, 1, -1, geom_id)
                
                if 0 < dist < 15.0:
                    hit_point = lidar_origin + world_dir * dist
                    points.append([hit_point[0], hit_point[1], hit_point[2]])

        # 5. 打包点云
        if points:
            header = Header()
            header.stamp = now
            header.frame_id = 'world'

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            cloud_msg = point_cloud2.create_cloud(header, fields, points)
            self.cloud_pub.publish(cloud_msg)

    def compute_gait(self, dt):
        self.gait_t += dt
        cmd = np.zeros(12)
        
        stand_hipy, stand_knee = -0.985, 2.084
        swing_amp = 0.25
        phase = 2 * np.pi * 2.0 * self.gait_t if (abs(self.vx) > 0.01 or abs(self.wz) > 0.01) else 0.0

        for i in range(12):
            leg, joint_type = i // 3, i % 3
            p = phase if leg in (0, 3) else phase + np.pi
            if joint_type == 1: # HipY
                pos = stand_hipy + swing_amp * np.sin(p) + self.vx * 0.2
                if self.wz != 0.0: pos += (1.0 if leg in (0,2) else -1.0) * self.wz * 0.1
            elif joint_type == 2: # Knee
                lift_phase = np.sin(p) if self.vx >= 0.0 else -np.sin(p)
                pos = stand_knee + 0.3 * max(0.0, lift_phase)
            else: # HipX
                pos = 0.1 if leg in (0, 2) else -0.1
            
            cmd[i] = 40.0 * (pos - self.data.qpos[7 + i]) - 2.0 * self.data.qvel[6 + i]
        return cmd

def ros2_spin_thread(node):
    rclpy.spin(node)

def main():
    rclpy.init()
    sim_node = LidarFrontendSimNode()
    threading.Thread(target=ros2_spin_thread, args=(sim_node,), daemon=True).start()

    tty.setcbreak(sys.stdin.fileno())

    with mujoco.viewer.launch_passive(sim_node.model, sim_node.data) as viewer:
        step = 0
        while viewer.is_running():
            if select.select([sys.stdin], [], [], 0)[0]:
                char = os.read(sys.stdin.fileno(), 1).decode(errors='ignore').lower()
                sim_node.process_keyboard(char)

            sim_node.data.ctrl[:] = sim_node.compute_gait(0.002)
            mujoco.mj_step(sim_node.model, sim_node.data)
            
            if step % 50 == 0: 
                sim_node.publish_lidar_and_odom()
                
            viewer.sync()
            step += 1
            time.sleep(0.002)
            
    sim_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()