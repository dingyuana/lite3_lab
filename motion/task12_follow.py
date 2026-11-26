#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained Lite3 task12: task10 dashboard with human tracking."""

import argparse
import math
import os
import threading
import time
import tkinter as tk

import cv2
import numpy as np

import lite3_common as l3
from task10_video import VideoStream
from ultralytics import YOLO


class ConfigurationError(Exception):
    pass


class TargetTracker:
    def __init__(self, model_path="yolov8n.pt", conf=0.5, lost_confirm=10):
        self.model = YOLO(model_path)
        self.conf = conf
        self.lost_confirm = lost_confirm
        self.lost_count = 0
        self.locked_id = None

    def update(self, frame):
        results = self.model.track(
            frame, persist=True, classes=[0], conf=self.conf, verbose=False)
        if not results or results[0].boxes.id is None:
            self.lost_count += 1
            if self.lost_count > self.lost_confirm:
                self.locked_id = None
            return None

        boxes = results[0].boxes
        if self.locked_id is None:
            candidates = []
            center_x = frame.shape[1] / 2.0
            for i in range(len(boxes.id)):
                x1, _, x2, _ = boxes.xyxy[i].cpu().numpy()
                candidates.append((abs((x1 + x2) / 2.0 - center_x),
                                   int(boxes.id[i])))
            if candidates:
                self.locked_id = min(candidates)[1]

        for i, tid in enumerate(boxes.id):
            if int(tid) == self.locked_id:
                self.lost_count = 0
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                if not all(np.isfinite((x1, y1, x2, y2))):
                    self.lost_count += 1
                    return None
                cx = (x1 + x2) / 2.0
                cy = y2
                return (cx, cy, float(x1), float(y1),
                        float(x2), float(y2))

        self.lost_count += 1
        if self.lost_count > self.lost_confirm:
            self.locked_id = None
        return None


def _quad_cross_values(pts):
    if np.asarray(pts).shape != (4, 2):
        return []
    crosses = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1])
                       - (b[1] - a[1]) * (c[0] - b[0]))
    return crosses


def _is_ordered_quad(pts):
    crosses = _quad_cross_values(pts)
    if len(crosses) != 4:
        return False
    return (all(v > 1e-6 for v in crosses)
            or all(v < -1e-6 for v in crosses))


def calibrate_interactive(frame, dst_pts=None, save_path="H.npy"):
    if dst_pts is None:
        dst_pts = np.array([[-1, 1], [1, 1], [1, 3], [-1, 3]],
                           dtype=np.float32)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)
    if dst_pts.shape != (4, 2) or not _is_ordered_quad(dst_pts):
        print("给定的 4 个地面坐标不是顺序一致的凸四边形，请检查 --dst")
        return None

    points = []
    base = frame.copy()
    labels = ["近左", "近右", "远右", "远左"]
    print("坐标定义：原点 = 机器人地面投影，X 右正，Y 前正（米）")
    print("请先用卷尺在地面放置 4 个标记，真实坐标为：")
    for i, (x, y) in enumerate(dst_pts):
        text = f"{i + 1}: {labels[i]} ({x:+.2f}, {y:+.2f})"
        print("   " + text)
        cv2.putText(base, text, (10, 24 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    print("按同样顺序左键点击 4 个标记；右键或 R 清除；Esc 中止。")

    def redraw():
        display = base.copy()
        for n, (px, py) in enumerate(points, 1):
            cv2.circle(display, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(display, str(n), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration", display)

    def on_click(event, x, y, _flags, _param):
        if event == cv2.EVENT_RBUTTONDOWN:
            points.clear()
            redraw()
        elif event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            redraw()

    cv2.imshow("Calibration", base.copy())
    cv2.setMouseCallback("Calibration", on_click)
    while len(points) < 4:
        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyAllWindows()
            return None
        if key in (ord("r"), ord("R")):
            points.clear()
            redraw()
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    src = np.array(points, dtype=np.float32)
    crosses = _quad_cross_values(src)
    if not (all(v > 1e-6 for v in crosses)
            or all(v < -1e-6 for v in crosses)):
        print("四点未构成顺序一致的凸四边形，已拒绝标定")
        print(f"转向叉积 = {np.round(crosses, 1)}")
        print("正确顺序：近左 → 近右 → 远右 → 远左（或整体反向）。")
        return None

    homography, _ = cv2.findHomography(src, dst_pts)
    if homography is None or not np.all(np.isfinite(homography)):
        print("单应矩阵求解失败，请重新标定")
        return None
    projected = cv2.perspectiveTransform(
        src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    residual = np.linalg.norm(projected - dst_pts, axis=1)
    print("重投影残差(m):", np.round(residual, 4))
    if residual.max() > 5e-3:
        print("残差异常，点选可能有误，请重新标定")
        return None

    np.save(save_path, homography)
    print(f"标定通过，H 已保存至 {save_path}\n{homography}")
    return homography


def pixel_to_ground(homography, u, v):
    point = np.array([u, v, 1.0])
    world = homography @ point
    if abs(world[2]) < 1e-9:
        return float("nan"), float("nan")
    world /= world[2]
    return float(world[0]), float(world[1])


def make_dummy_h(width=640, height=480, near=1.0, far=3.0):
    sx = 2.0 / float(width)
    sy = (far - near) / (height / 2.0)
    return np.array([
        [sx, 0.0, -sx * width / 2.0],
        [0.0, -sy, far + sy * height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


class SafetySupervisor:
    def __init__(self, require_ultrasonic=True):
        self.frame_age_max = 0.2
        self.lost_confirm = 10
        self.lost_frames = 0
        self.dist_stop, self.dist_resume = 1.0, 1.3
        self.dist_giveup, self.dist_back = 3.5, 3.0
        self.too_close = False
        self.too_far = False
        self.require_ultrasonic = require_ultrasonic
        self.us_min, self.us_resume = 1.0, 1.3
        self.us_timeout = 0.3
        self.us_value = None
        self.us_stamp = 0.0
        self.us_blocked = False
        self.link_timeout = 0.5
        self.last_perception_ok = time.monotonic()

    def update_ultrasonic(self, distance_m):
        self.us_value = float(distance_m)
        self.us_stamp = time.monotonic()

    def _check_ultrasonic(self):
        if not self.require_ultrasonic:
            return True, "OK(脱机)"
        if time.monotonic() - self.us_stamp > self.us_timeout:
            return False, "超声波数据超时"
        ultrasound = self.us_value
        if ultrasound is None or not np.isfinite(ultrasound) or ultrasound <= 0:
            return False, "超声波数据无效"
        if ultrasound < self.us_min:
            self.us_blocked = True
        elif ultrasound > self.us_resume:
            self.us_blocked = False
        if self.us_blocked:
            return False, f"前方障碍 {ultrasound:.2f}m"
        return True, "OK"

    def check(self, frame_age, target, distance):
        if frame_age is None or frame_age > self.frame_age_max:
            return False, "帧过期", "STOP"
        if time.monotonic() - self.last_perception_ok > self.link_timeout:
            return False, "感知链路超时", "STOP"
        if target is None:
            self.lost_frames += 1
            return False, f"目标丢失({self.lost_frames})", "STOP"
        self.lost_frames = 0
        if distance is None or not np.isfinite(distance):
            return False, "距离数据无效", "STOP"
        ok, reason = self._check_ultrasonic()
        if not ok:
            return False, reason, "STOP"
        if distance < self.dist_stop:
            self.too_close = True
        elif distance > self.dist_resume:
            self.too_close = False
        if distance > self.dist_giveup:
            self.too_far = True
        elif distance < self.dist_back:
            self.too_far = False
        if self.too_close:
            return True, f"目标过近 {distance:.2f}m", "ONLY_TURN"
        if self.too_far:
            return False, f"目标过远 {distance:.2f}m", "STOP"
        return True, "OK", "NORMAL"


class SafetyGate:
    def __init__(self):
        self.allow_forward = True
        self.allow_backward = True
        self.allow_rotation = True

    def apply(self, vx, wz, decision="NORMAL"):
        if decision == "STOP":
            return 0.0, 0.0
        if decision == "ONLY_TURN":
            return 0.0, wz if self.allow_rotation else 0.0
        if vx > 0 and not self.allow_forward:
            vx = 0.0
        if vx < 0 and not self.allow_backward:
            vx = 0.0
        if not self.allow_rotation:
            wz = 0.0
        return vx, wz


class DualLoopPID:
    WZ_SIGN = +1

    def __init__(self, target_distance=2.0,
                 kp_yaw=0.6, kd_yaw=0.25, tau=0.15,
                 kp_vx=0.5, ki_vx=0.05, max_integral=5.0,
                 vx_max=0.4, vx_min=-0.2, wz_max=0.25,
                 yaw_deadzone=0.06, dist_deadzone=0.10,
                 dvx_max=0.15, dwz_max=0.5):
        self.kp_yaw, self.kd_yaw, self.tau = kp_yaw, kd_yaw, tau
        self.error_filt = 0.0
        self.prev_error_filt = 0.0
        self.yaw_deadzone = yaw_deadzone
        self.kp_vx, self.ki_vx = kp_vx, ki_vx
        self.integral_dist = 0.0
        self.target_distance = target_distance
        self.max_integral = max_integral
        self.dist_deadzone = dist_deadzone
        self.vx_max, self.vx_min, self.wz_max = vx_max, vx_min, wz_max
        self.dvx_max = dvx_max
        self.dwz_max = dwz_max
        self.prev_vx = 0.0
        self.prev_wz = 0.0
        self._last_t = None

    def reset(self):
        self.error_filt = 0.0
        self.prev_error_filt = 0.0
        self.integral_dist = 0.0
        self.prev_vx = 0.0
        self.prev_wz = 0.0
        self._last_t = None

    def update(self, error_x, distance, active=True):
        if not active:
            self.reset()
            return 0.0, 0.0

        first_update = self._last_t is None
        now = time.monotonic()
        dt = 0.05 if first_update else now - self._last_t
        self._last_t = now
        dt = min(max(dt, 0.01), 0.5)

        alpha = min(dt / (self.tau + dt), 1.0)
        self.error_filt = alpha * error_x + (1 - alpha) * self.error_filt
        if abs(self.error_filt) < self.yaw_deadzone:
            wz = 0.0
        else:
            d_err = (0.0 if first_update else
                     (self.error_filt - self.prev_error_filt) / dt)
            wz = self.WZ_SIGN * (self.kp_yaw * self.error_filt
                                 + self.kd_yaw * d_err)
        self.prev_error_filt = self.error_filt
        wz = max(min(wz, self.wz_max), -self.wz_max)
        dwz = self.dwz_max * min(dt, 0.1)
        wz = max(min(wz, self.prev_wz + dwz), self.prev_wz - dwz)
        self.prev_wz = wz

        error_dist = distance - self.target_distance
        vx = 0.0
        if abs(error_dist) >= self.dist_deadzone:
            self.integral_dist = max(
                min(self.integral_dist + error_dist * dt, self.max_integral),
                -self.max_integral)
            vx = self.kp_vx * error_dist + self.ki_vx * self.integral_dist
            vx = max(min(vx, self.vx_max), self.vx_min)
        dvx = self.dvx_max * dt / 0.05
        vx = max(min(vx, self.prev_vx + dvx), self.prev_vx - dvx)
        self.prev_vx = vx
        return vx, wz


def pid_selfcheck():
    pid = DualLoopPID()
    vx, wz = pid.update(+0.3, 2.0)
    assert (wz < 0) == (pid.WZ_SIGN < 0), f"方向错误 WZ_SIGN={pid.WZ_SIGN}"
    print(f"[1] 目标在右  vx={vx:.2f} wz={wz:.2f} ✓")
    pid.reset()
    vx, _ = pid.update(0.0, 3.0)
    assert vx > 0, "目标偏远应前进"
    print(f"[2] 目标偏远  vx={vx:.2f} ✓")
    pid.reset()
    vx, _ = pid.update(0.0, 1.0)
    assert vx < 0, "目标偏近应后退"
    print(f"[3] 目标偏近  vx={vx:.2f} ✓")
    pid.reset()
    for _ in range(20):
        vx, wz = pid.update(0.02, 2.05)
    assert vx == 0 and wz == 0, "死区内应输出0"
    print("[4] 死区      输出=0 ✓")
    pid.reset()
    for _ in range(10):
        pid.update(0.5, 3.5)
    vx, wz = pid.update(0.5, 3.5, active=False)
    assert vx == 0 and wz == 0 and pid.integral_dist == 0.0
    print("[5] 安全停车  归零+积分清零 ✓")
    pid.reset()
    vx, _ = pid.update(0.0, 5.0)
    assert abs(vx) <= pid.dvx_max + 1e-6
    print(f"[6] 斜率限制  首周期 vx={vx:.2f} ✓")
    print("\n全部自检通过（WZ_SIGN 方向需悬挂测试真机确认）")


class FollowSession:
    def __init__(self, robot, video, h_path="H.npy", model_path="yolov8n.pt",
                 offline=False, dummy_h=False, fwd_scale=20000,
                 yaw_scale=65534):
        self.robot = robot
        self.video = video
        self.h_path = h_path
        self.model_path = model_path
        self.offline = offline
        self.dummy_h = dummy_h
        self.fwd_scale = fwd_scale
        self.yaw_scale = yaw_scale
        self.H = None
        self.tracker = None
        self.pid = DualLoopPID()
        self.gate = SafetyGate()
        self.supervisor = SafetySupervisor(require_ultrasonic=not offline)
        self.active = False
        self._stop = threading.Event()
        self._thread = None
        self._display_lock = threading.Lock()
        self._display_frame = None
        self._status = "未开始"

    def start(self):
        if self.active:
            return
        if self.dummy_h and not self.offline:
            raise ConfigurationError(
                "--dummy-h 只能用于脱机跟踪，真机模式请使用真实标定文件")
        if self.dummy_h:
            frame, _ = self.video.latest()
            if frame is None:
                height, width = 480, 640
            else:
                height, width = frame.shape[:2]
            self.H = make_dummy_h(width=width, height=height)
        else:
            if not os.path.exists(self.h_path):
                raise ConfigurationError(
                    f"未找到标定文件 {self.h_path}，请先运行：\n"
                    f"  python3 task12_follow.py --calibrate --h-file {self.h_path}")
            self.H = np.load(self.h_path)
        if self.H.shape != (3, 3) or not np.all(np.isfinite(self.H)):
            raise ConfigurationError(
                f"{self.h_path} 不是有效的 3x3 单应矩阵，请重新标定")

        self.tracker = TargetTracker(self.model_path)
        self.pid.reset()
        self.supervisor = SafetySupervisor(require_ultrasonic=not self.offline)
        if not self.offline:
            self.robot.start_tracking()
        self.active = True
        self._status = "正在搜索人体目标"
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _safe_stop(self, reason):
        self.pid.reset()
        if not self.offline:
            self.robot.set_tracking_axes(0, 0)
        return reason

    def _publish(self, frame, status):
        with self._display_lock:
            self._display_frame = frame.copy() if frame is not None else None
            self._status = status

    def latest(self):
        with self._display_lock:
            frame = (self._display_frame.copy()
                     if self._display_frame is not None else None)
            return frame, self._status

    @staticmethod
    def _annotate(frame, target, distance, status, offline):
        display = frame.copy()
        color = (0, 200, 0) if target is not None else (0, 165, 255)
        if target is not None:
            x1, y1, x2, y2 = map(int, target[2:6])
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.circle(display, (int(target[0]), int(target[1])), 5, color, -1)
        mode = "OFFLINE" if offline else "FOLLOW"
        cv2.putText(display, f"{mode}: {status}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        if distance is not None and math.isfinite(distance):
            cv2.putText(display, f"distance={distance:.2f}m", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return display

    def _run(self):
        while not self._stop.is_set():
            frame, video_error = self.video.latest()
            if frame is None:
                reason = self._safe_stop(video_error or "视频帧不可用")
                self._publish(None, reason)
                self._stop.wait(0.1)
                continue

            target = None
            distance = None
            try:
                target = self.tracker.update(frame)
                self.supervisor.last_perception_ok = time.monotonic()
                snapshot = self.robot.snapshot()
                if not self.offline and not snapshot.get("tracking", True):
                    self.active = False
                    self._stop.set()
                    self._publish(frame, "跟踪已停止：机器人状态不满足运动条件")
                    return
                if snapshot.get("front") is not None:
                    self.supervisor.update_ultrasonic(snapshot["front"])

                if target is None:
                    status = self._safe_stop("目标丢失，已停车")
                else:
                    lateral, forward = pixel_to_ground(
                        self.H, target[0], target[1])
                    if not (np.isfinite(lateral) and np.isfinite(forward)) \
                            or forward <= 0:
                        status = self._safe_stop("地面反投影无效，已停车")
                    else:
                        distance = forward
                        error_x = math.atan2(lateral, forward)
                        age = 0.0
                        ok, reason, decision = self.supervisor.check(
                            age, target, distance)
                        if not ok:
                            status = self._safe_stop(reason)
                        else:
                            vx, wz = self.pid.update(error_x, distance)
                            vx, wz = self.gate.apply(vx, wz, decision)
                            if not self.offline:
                                forward_axis = int(round(max(
                                    min(vx * self.fwd_scale, 32767), -32767)))
                                yaw_axis = int(round(max(
                                    min(wz * self.yaw_scale, 32767), -32767)))
                                self.robot.set_tracking_axes(
                                    forward_axis, yaw_axis)
                            if decision == "NORMAL":
                                status = "跟踪中"
                            else:
                                status = reason
                display = self._annotate(
                    frame, target, distance, status, self.offline)
                self._publish(display, status)
            except Exception as exc:
                status = self._safe_stop(f"感知异常：{exc}")
                display = self._annotate(
                    frame, target, distance, status, self.offline)
                self._publish(display, status)
            self._stop.wait(0.1)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.active = False
        if not self.offline:
            self.robot.stop_tracking()
        with self._display_lock:
            self._display_frame = None
            self._status = "已停止"

    close = stop


def _calibrate(video_url, h_file):
    video = VideoStream(video_url)
    video.start()
    frame = None
    deadline = time.monotonic() + 10.0
    try:
        while time.monotonic() < deadline:
            frame, _ = video.latest()
            if frame is not None:
                break
            time.sleep(0.1)
    finally:
        video.close()
    if frame is None:
        print("未获取到视频帧，请检查 RTSP 地址和网络")
        return
    calibrate_interactive(frame, save_path=h_file)


def main():
    parser = argparse.ArgumentParser(description="Lite3 task12 人体跟踪界面")
    l3.add_common_args(parser)
    parser.add_argument(
        "--video-url", default="rtsp://192.168.1.120:8554/test?tcp",
        help="机器人 RTSP 视频地址")
    parser.add_argument("--h-file", default="H.npy", help="单应矩阵文件")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 权重")
    parser.add_argument("--fwd-scale", type=int, default=20000,
                        help="前进速度到轴值的映射系数")
    parser.add_argument("--yaw-scale", type=int, default=65534,
                        help="转向速度到轴值的映射系数")
    parser.add_argument("--offline", action="store_true",
                        help="只显示跟踪结果，不发送运动指令")
    parser.add_argument("--dummy-h", action="store_true",
                        help="脱机模式使用占位单应矩阵")
    parser.add_argument("--calibrate", action="store_true", help="交互式标定")
    parser.add_argument("--selfcheck", action="store_true", help="PID 数值自检")
    parser.add_argument("--auto-follow", action="store_true",
                        help="界面启动后自动开始跟踪")
    args = parser.parse_args()

    if args.selfcheck:
        pid_selfcheck()
        return
    if args.dummy_h and args.calibrate:
        parser.error("--dummy-h 不能与 --calibrate 同时使用")
    if args.dummy_h and not args.offline:
        parser.error("--dummy-h 只能与 --offline 同时使用")
    if args.calibrate:
        _calibrate(args.video_url, args.h_file)
        return

    from task10_gui import Task10App

    root = tk.Tk()
    try:
        app = Task10App(
            root, args.ip, args.state_port, args.video_url,
            title="Lite3 task12 - 人体跟踪",
            follow_options={
                "h_path": args.h_file,
                "model_path": args.model,
                "offline": args.offline,
                "dummy_h": args.dummy_h,
                "fwd_scale": args.fwd_scale,
                "yaw_scale": args.yaw_scale,
            })
    except OSError as exc:
        root.destroy()
        parser.error(f"无法初始化机器人 UDP 状态端口：{exc}")
    if args.auto_follow:
        root.after(500, app.toggle_tracking)
    root.mainloop()


if __name__ == "__main__":
    main()
