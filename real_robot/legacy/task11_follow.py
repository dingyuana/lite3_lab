# -*- coding: utf-8 -*-
"""
Lite3 视觉跟随系统 —— 安全修复版
流水线：RTSP 采集 → YOLO+ByteTrack → Homography 反投影
        → 安全校验(含超声波) → 双环 PID → SafetyGate → 三轴指令发送

与上一版的关键修复：
  1. 距离环符号修正（远→前进、近→后退），--selfcheck 可通过
  2. 不再调用 lite3_common 中不存在的 API；使用本地三轴 AxisStream
  3. 补齐 起立 → CMD_MOVE_MODE → 轴流保活 → CMD_GAIT_LOW 的完整准备流程
  4. 所有异常/丢失/断流路径统一走 _safe_stop：持续发 (0,0,0) + PID reset
  5. 目标丢失不再残留旧速度；超声波阈值与轴值标定回归安全范围
  6. 多人检测不再触发张量布尔异常；标定增加点序校验

使用顺序：
   1. python3 task11_follow.py --calibrate    # 交互式 Homography 标定（生成 H.npy）
   2. python3 task11_follow.py --selfcheck    # 数值自检（PID 方向/死区/限幅）
   3. python3 task11_follow.py --offline      # 脱机验证：只看视觉，不发指令
   4. python3 task11_follow.py                # 真机跟随（先悬挂测试！）

没有 H.npy 时可用 --dummy-h 生成占位矩阵做脱机冒烟测试：
   python3 task11_follow.py --offline --dummy-h
注意：占位矩阵无物理意义，--dummy-h 严禁用于真机模式。
"""
import argparse
import math
import os
import socket
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np

import lite3_common as l3
from ultralytics import YOLO


class ConfigurationError(Exception):
    """启动前置条件不满足：由 main 捕获后给出明确提示，不抛 traceback。"""


# ======================================================================
# 模块 1：RTSP 低延迟采集线程
# ======================================================================
class RTSPGrabber:
    """只留最新帧 + 时间戳。max_age 超时的帧视为不可信。"""

    def __init__(self, url, max_age=0.2, connect_timeout=2.0):
        # OpenCV/FFmpeg 打开 RTSP 不可达地址会阻塞约 30s，先做 TCP 预检
        parsed = urlparse(url)
        if parsed.scheme == "rtsp" and parsed.hostname:
            port = parsed.port or 554
            try:
                with socket.create_connection(
                        (parsed.hostname, port), timeout=connect_timeout):
                    pass
            except OSError as e:
                raise ConfigurationError(
                    f"RTSP 主机不可达：{parsed.hostname}:{port}（{e}）\n"
                    f"视频地址：{url}\n"
                    "请检查：\n"
                    "  1) 机器人是否已上电、RTSP 服务是否已启动\n"
                    "  2) IP/端口是否正确，可用 --url 指定地址\n"
                    f"  3) 先用另一个程序确认画面："
                    f"python3 task10_gui.py --video-url {url}")
        self.cap = cv2.VideoCapture()
        if not self.cap.open(url, cv2.CAP_FFMPEG):
            raise ConfigurationError(
                f"无法打开视频流：{url}\n"
                "请检查：\n"
                "  1) 机器人是否已上电、RTSP 服务是否已启动\n"
                "  2) IP/端口是否正确，可用 --url 指定地址\n"
                f"  3) 先用另一个程序确认画面：python3 task10_gui.py --video-url {url}")
        self.max_age = max_age
        self._frame = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._stamp = time.monotonic()
            else:
                time.sleep(0.01)          # 流暂时无数据，避免空转

    def get_latest(self):
        """返回 (frame, age)。过期返回 (None, age)。"""
        with self._lock:
            frame, stamp = self._frame, self._stamp
        if frame is None:
            return None, 999.0
        age = time.monotonic() - stamp
        if age > self.max_age:
            return None, age
        return frame, age

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        self.cap.release()


# ======================================================================
# 模块 2：YOLOv8n 检测 + ByteTrack 跟踪
# ======================================================================
class TargetTracker:
    """锁定单一 person 的 track ID，输出脚底中心像素坐标。"""

    def __init__(self, model_path="yolov8n.pt", conf=0.5,
                 lost_confirm=10):
        self.model = YOLO(model_path)
        self.conf = conf
        self.lost_confirm = lost_confirm
        self.lost_count = 0
        self.locked_id = None

    def update(self, frame):
        """返回 (cx, cy脚底, x1, y1, x2, y2) 或 None"""
        results = self.model.track(frame, persist=True, classes=[0],
                                   conf=self.conf, verbose=False)
        if not results or results[0].boxes.id is None:
            self.lost_count += 1
            if self.lost_count > self.lost_confirm:
                self.locked_id = None
            return None

        boxes = results[0].boxes
        if self.locked_id is None:
            # 首次锁定：选画面中最靠近中心的 person，避免锁到旁边路人
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
                cy = y2                                  # 底边=脚底点
                return (cx, cy, float(x1), float(y1),
                        float(x2), float(y2))

        self.lost_count += 1
        if self.lost_count > self.lost_confirm:
            self.locked_id = None                        # 确认丢失后重锁
        return None


# ======================================================================
# 模块 3：Homography 地平面标定与反投影
# ======================================================================
def _quad_cross_values(pts):
    """相邻三点的转向叉积：同号且非零 = 顺序一致的凸四边形。"""
    if np.asarray(pts).shape != (4, 2):
        return []
    crosses = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1])
                       - (b[1] - a[1]) * (c[0] - b[0]))
    return crosses


def _is_ordered_quad(pts):
    """四点必须构成凸四边形且顺序一致，否则点序错误无法被残差发现。"""
    crosses = _quad_cross_values(pts)
    if len(crosses) != 4:
        return False
    return (all(v > 1e-6 for v in crosses)
            or all(v < -1e-6 for v in crosses))


def calibrate_interactive(frame, dst_pts=None, save_path="H.npy"):
    """鼠标点选 4 个地面标记点。

    dst_pts 是这 4 个标记在“机器人坐标系”中的真实位置，单位米：
       原点 = 机器人机身在地面上的投影，X 右正，Y 前正。
    默认 (-1,1) (1,1) (1,3) (-1,3)，即前方 1~3 m、左右各 1 m 的矩形。
    这些数值必须由你用卷尺实际测量/摆放，程序无法自动知道。
    标定姿态需与跟随运行姿态一致：相机装在机器人上时，必须站立静止标定；
    趴着标定得到的 H 不能用于站立跟随。
    """
    if dst_pts is None:
        dst_pts = np.array([[-1, 1], [1, 1], [1, 3], [-1, 3]],
                           dtype=np.float32)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)
    if dst_pts.shape != (4, 2) or not _is_ordered_quad(dst_pts):
        print("❌ 给定的 4 个地面坐标不是顺序一致的凸四边形，请检查 --dst")
        return None

    points = []
    base = frame.copy()
    labels = ["近左", "近右", "远右", "远左"]

    print("坐标定义：原点 = 机器人地面投影，X 右正，Y 前正（米）")
    print("请先用卷尺在地面放置 4 个标记（胶带画十字），真实坐标为：")
    for i, (x, y) in enumerate(dst_pts):
        text = f"{i + 1}: {labels[i]} ({x:+.2f}, {y:+.2f})"
        print("   " + text)
        cv2.putText(base, text, (10, 24 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    print("然后按同样顺序左键点击 4 个标记；点错可点右键或按 R 清除重来；"
          "Esc 中止。")

    def redraw():
        disp = base.copy()
        for n, (px, py) in enumerate(points, 1):
            cv2.circle(disp, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(disp, str(n), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration", disp)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:          # 右键：全部清除
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
        if key in (ord('r'), ord('R')):             # R：全部清除
            points.clear()
            redraw()
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    src = np.array(points, dtype=np.float32)
    crosses = _quad_cross_values(src)
    if not (all(v > 1e-6 for v in crosses)
            or all(v < -1e-6 for v in crosses)):
        print("❌ 四点未构成顺序一致的凸四边形，已拒绝标定")
        print(f"   转向叉积 = {np.round(crosses, 1)} "
              "（应全部同号且非零）")
        print("   正确顺序：近左 → 近右 → 远右 → 远左（或整体反向也可）。")
        print("   常见原因：点序交叉、点错标记、点到人脚/画面边缘、两点重合。")
        print("   请在窗口中右键或按 R 清除重来，或重新运行 --calibrate。")
        return None
    H, _ = cv2.findHomography(src, dst_pts)
    if H is None or not np.all(np.isfinite(H)):
        print("❌ 单应矩阵求解失败，请重新标定")
        return None

    # 4 点精确拟合时残差必然很小，故额外做点序/凸性校验
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    resid = np.linalg.norm(proj - dst_pts, axis=1)
    print("重投影残差(m):", np.round(resid, 4))
    if resid.max() > 5e-3:
        print("❌ 残差异常，点选可能有误，请重新标定")
        return None

    np.save(save_path, H)
    print(f"✅ 标定通过，H 已保存至 {save_path}\n{H}")
    return H


def pixel_to_ground(H, u, v):
    """像素 → 地面坐标。返回 (lateral_x 右正, forward_y 前正) 米"""
    pt = np.array([u, v, 1.0])
    world = H @ pt
    if abs(world[2]) < 1e-9:                 # 地平线附近，反投影无意义
        return float("nan"), float("nan")
    world /= world[2]
    return float(world[0]), float(world[1])


def make_dummy_h(width=640, height=480, near=1.0, far=3.0):
    """生成仅供脱机冒烟测试的占位仿射矩阵（无物理标定意义）。

    假设 640x480：图像下半部分由远(y=far)线性映射到近(y=near)，
    横向 [-width/2, width/2] 映射到 [-1, +1] 米。
    """
    sx = 2.0 / float(width)
    sy = (far - near) / (height / 2.0)
    return np.array([
        [sx, 0.0, -sx * width / 2.0],
        [0.0, -sy, far + sy * height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# ======================================================================
# 模块 4：SafetySupervisor —— 安全校验（含超声波，全带滞回）
# ======================================================================
class SafetySupervisor:
    """任何一环失效都返回 (False, 原因, 'STOP')。

    ok=True 且 decision='ONLY_TURN' 表示目标过近：禁走、允转。
    脱机模式可通过 require_ultrasonic=False 关闭超声波硬约束。
    """

    def __init__(self, require_ultrasonic=True):
        # 1 帧龄
        self.frame_age_max = 0.2
        # 2 目标丢失计数（仅用于日志/重锁；任何丢帧都立即停车）
        self.lost_confirm = 10
        self.lost_frames = 0
        # 3 视觉距离（双阈值滞回）
        self.dist_stop, self.dist_resume = 1.0, 1.3
        self.dist_giveup, self.dist_back = 3.5, 3.0
        self.too_close = False
        self.too_far = False
        # 4 超声波（硬刹停层，独立于目标）
        #    原 0.4 m 阈值在实测轴值下刹车距离不足，回归 task10 的 1.0 m
        self.require_ultrasonic = require_ultrasonic
        self.us_min, self.us_resume = 1.0, 1.3
        self.us_timeout = 0.3
        self.us_value = None
        self.us_stamp = 0.0
        self.us_blocked = False
        # 5 感知链路心跳
        self.link_timeout = 0.5
        self.last_perception_ok = time.monotonic()

    # ---- 超声波数据源（由感知线程每个有效状态帧调用）----
    def update_ultrasonic(self, distance_m):
        self.us_value = float(distance_m)
        self.us_stamp = time.monotonic()

    def _check_ultrasonic(self):
        if not self.require_ultrasonic:
            return True, "OK(脱机)"
        if time.monotonic() - self.us_stamp > self.us_timeout:
            return False, "超声波数据超时"
        us = self.us_value
        if us is None or not np.isfinite(us) or us <= 0:
            return False, "超声波数据无效"
        if us < self.us_min:
            self.us_blocked = True
        elif us > self.us_resume:
            self.us_blocked = False
        if self.us_blocked:
            return False, f"前方障碍 {us:.2f}m"
        return True, "OK"

    def check(self, frame_age, target, distance):
        # 1 帧龄
        if frame_age is None or frame_age > self.frame_age_max:
            return False, "帧过期", "STOP"

        # 2 感知心跳（感知线程在每次拿到有效帧时刷新）
        if time.monotonic() - self.last_perception_ok > self.link_timeout:
            return False, "感知链路超时", "STOP"

        # 3 目标丢失：立即停车，不让旧速度残留
        if target is None:
            self.lost_frames += 1
            return False, f"目标丢失({self.lost_frames})", "STOP"
        self.lost_frames = 0

        # 4 视觉距离有效性
        if distance is None or not np.isfinite(distance):
            return False, "距离数据无效", "STOP"

        # 5 超声波硬刹停（先于距离决策）
        ok, reason = self._check_ultrasonic()
        if not ok:
            return False, reason, "STOP"

        # 6 距离滞回
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


# ======================================================================
# 模块 5：SafetyGate —— 按决策屏蔽运动轴
# ======================================================================
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


# ======================================================================
# 模块 6：DualLoopPID —— 航向 PD + 距离 PI
# ======================================================================
class DualLoopPID:
    """
    符号约定（自检确认）:
        error_x : 方位角误差(弧度) = atan2(lateral, distance)，目标在右为正
        vx      : 前进为正 m/s；error_dist = distance - target，远为正 → 前进
        wz      : 右转为正 rad/s；真机方向不符时翻转 WZ_SIGN
    """

    WZ_SIGN = +1     # 目标在右(error_x>0) → 右转(wz>0)。真机不符则改 -1

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
        """停车/重新锁定目标时调用，清除状态残留。"""
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

        # 航向环：低通 → 死区 → PD
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

        # 距离环：PI（死区不积累 + 限幅）
        error_dist = distance - self.target_distance
        vx = 0.0
        if abs(error_dist) >= self.dist_deadzone:
            self.integral_dist = max(
                min(self.integral_dist + error_dist * dt, self.max_integral),
                -self.max_integral)
            # 远 → 正 → 前进；近 → 负 → 后退
            vx = self.kp_vx * error_dist + self.ki_vx * self.integral_dist
            vx = max(min(vx, self.vx_max), self.vx_min)

        # 斜率限制：防突启突停
        dvx = self.dvx_max * dt / 0.05
        vx = max(min(vx, self.prev_vx + dvx), self.prev_vx - dvx)
        self.prev_vx = vx
        return vx, wz


# ======================================================================
# 模块 7：三轴保活轴流 + RobotSender
# ======================================================================
class AxisStream:
    """≥20Hz 持续发送 forward/lateral/yaw；看门狗超时自动归零。"""

    def __init__(self, ctrl, hz=30.0, watchdog=0.35):
        if hz < 20.0:
            raise ValueError("轴流频率必须 ≥20Hz")
        self.ctrl = ctrl
        self.hz = hz
        self.watchdog = watchdog
        self._forward = 0
        self._yaw = 0
        self._last_set = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.frames = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_axes(self, forward, yaw):
        with self._lock:
            self._forward = int(forward)
            self._yaw = int(yaw)
            self._last_set = time.monotonic()

    def _loop(self):
        period = 1.0 / self.hz
        while not self._stop.is_set():
            with self._lock:
                if time.monotonic() - self._last_set > self.watchdog:
                    forward, yaw = 0, 0
                else:
                    forward, yaw = self._forward, self._yaw
            self.ctrl.send_simple(l3.CMD_AXIS_FORWARD, forward)
            self.ctrl.send_simple(l3.CMD_AXIS_LATERAL, 0)
            self.ctrl.send_simple(l3.CMD_AXIS_YAW, yaw)
            self.frames += 1
            self._stop.wait(period)

    def stop(self):
        self.set_axes(0, 0)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None


def _wait_stable(ctrl, predicate, stable_sec, timeout, tag):
    """等待 predicate(st, knee_mean) 连续 stable_sec 秒成立。"""
    t0 = time.monotonic()
    stable_t0 = None
    last_print = 0.0
    last_mean = None
    while time.monotonic() - t0 < timeout:
        st = ctrl.get_state()
        mean = ctrl.knee_mean_deg()
        if mean is not None:
            last_mean = mean
        elapsed = time.monotonic() - t0
        if elapsed - last_print >= 1.0:
            t, desc = l3.state_tuple(st)
            m_str = f"{mean:+.1f}°" if mean is not None else "?"
            print(f"[{tag}] {elapsed:4.1f}/{timeout:.0f}s "
                  f"状态={t} {desc} 四膝={m_str}")
            last_print = elapsed
        if predicate(st, mean):
            if stable_t0 is None:
                stable_t0 = time.monotonic()
            elif time.monotonic() - stable_t0 >= stable_sec:
                return True, last_mean
        else:
            stable_t0 = None
        time.sleep(0.1)
    return False, last_mean


def _is_stand_ready(st, mean):
    return (l3.is_stand_still(st) and mean is not None
            and l3.STAND_MIN_DEG <= mean <= l3.STAND_MAX_DEG)


def _is_gait_ready(st, mean):
    return (st is not None and st.robot_basic_state == 6
            and st.robot_gait_state == l3.GAIT_LOW_SPEED
            and mean is not None
            and l3.STAND_MIN_DEG <= mean <= l3.SQUAT_MAX_DEG)


def _axis_value(value, scale, deadzone):
    axis = int(round(max(min(value * scale, 32767), -32767)))
    # 非零指令若落在固件死区内，机器人完全不动；抬到死区外最小可动值
    if 0 < abs(axis) < deadzone:
        axis = deadzone + 1 if axis > 0 else -(deadzone + 1)
    return axis


def twist_to_axis(vx, wz, fwd_scale=40000, yaw_scale=65534):
    """速度 → 轴值，并补偿 lite3_common 的轴死区。"""
    axis_fwd = _axis_value(vx, fwd_scale, l3.AXIS_DEADZONE_FWD)
    axis_yaw = _axis_value(wz, yaw_scale, l3.AXIS_DEADZONE_YAW)
    return axis_fwd, axis_yaw


class RobotSender:
    AXIS_MAX = 32767

    def __init__(self, ip="192.168.1.120", state_port=43897, hz=30.0,
                 fwd_scale=40000, yaw_scale=65534, do_home=True):
        state_port = state_port if state_port and state_port > 0 else None
        self.ctrl = l3.Lite3Controller(ip, state_port=state_port)
        self.ctrl.start_heartbeat(hz=5)
        if state_port:
            self.ctrl.start_state_receiver()
        self.stream = AxisStream(self.ctrl, hz=hz)
        # 默认 40000：vx=0.4→16000，vx=-0.2→-8000，均超过轴死区 6553
        self.fwd_scale = fwd_scale
        self.yaw_scale = yaw_scale          # wz=0.5 → 32767，恰不溢出
        self.do_home = do_home
        self._prepared = False
        self._closed = False

    def prepare(self):
        """起立 → 移动模式 → 轴流保活 → 低速步态，全部确认后才允许跟随。"""
        st = self.ctrl.get_state()
        mean = self.ctrl.knee_mean_deg()
        if st is None or mean is None:
            raise RuntimeError("未收到 0x0901/0x0902，先运行 task01/task02")

        if _is_stand_ready(st, mean):
            print("✅ 当前已稳定站立，跳过起立")
        else:
            if l3.is_initial_standby(st) and st.battery_level <= 20.0:
                raise RuntimeError("初始待机(98)下电量不高于20%，禁止首次起立")
            print("发送 0x21010202 起立 ...")
            self.ctrl.send_simple(l3.CMD_STAND_SIT)
            ok, last = _wait_stable(self.ctrl, _is_stand_ready,
                                    3.0, 30.0, "起立")
            if not ok:
                raise RuntimeError(f"30s 内未稳定站立，最后四膝={last}")
            print("✅ 起立完成，四膝 60~90° 稳定 3s")

        print("发送 0x21010D06 移动模式 ...")
        self.ctrl.send_simple(l3.CMD_MOVE_MODE)
        time.sleep(0.5)

        self.stream.set_axes(0, 0)
        self.stream.start()                 # 先起轴流，再切步态（task09 流程）
        time.sleep(0.3)
        print("发送 0x21010300 平地低速步态 ...")
        self.ctrl.send_simple(l3.CMD_GAIT_LOW)
        ok, last = _wait_stable(self.ctrl, _is_gait_ready,
                                0.5, 8.0, "切步态")
        if not ok:
            self.stream.set_axes(0, 0)
            self.stream.stop()
            raise RuntimeError(f"8s 内未确认低速步态就绪，最后四膝={last}")
        print("✅ 移动模式 + 低速步态就绪，跟随使能")
        self._prepared = True

    def send_twist(self, vx, wz):
        """把 m/s、rad/s 映射成轴值；零值也主动刷新看门狗。"""
        if not self._prepared or self._closed:
            return
        axis_fwd, axis_yaw = twist_to_axis(
            vx, wz, self.fwd_scale, self.yaw_scale)
        self.stream.set_axes(axis_fwd, axis_yaw)

    def emergency_stop(self):
        """持续发 (0,0,0)，不停止轴流，避免 250ms 超时失控。"""
        self.stream.set_axes(0, 0)

    def stop(self):
        """安全退出：归零 → 等减速 → 停轴流 → 趴下/回零 → 关闭 UDP。"""
        if self._closed:
            return
        self._closed = True
        self._prepared = False
        self.stream.set_axes(0, 0)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            st = self.ctrl.get_state()
            if st is not None and abs(st.vel_body[0]) < 0.02:
                break
            time.sleep(0.1)
        self.stream.stop()
        print(f"轴流已停止，共发 {self.stream.frames} 帧")
        if self.do_home:
            ok = l3.ensure_home(self.ctrl)
        else:
            ok = l3.ensure_sit(self.ctrl)
        print("✅ 回零完成" if ok else "⚠ 趴下/回零未完全确认，请人工检查")
        self.ctrl.close()
        print("✅ 已安全退出")


# ======================================================================
# 主应用：双线程集成（感知 10Hz / 控制 20Hz 解耦）
# ======================================================================
class VisualTrackingApp:
    STATE_IDLE = "IDLE"          # 尚未准备，不发轴指令
    STATE_TRACK = "TRACK"        # 正常跟随
    STATE_STOP = "STOP"          # 安全拦截，持续发 (0,0,0)

    def __init__(self, rtsp_url, h_path="H.npy", ip="192.168.1.120",
                 state_port=43897, model_path="yolov8n.pt", conf=0.5,
                 offline=False, do_home=True, fwd_scale=40000,
                 target_dist=2.0):
        if not os.path.exists(h_path):
            raise ConfigurationError(self._h_missing_help(h_path))
        self.H = np.load(h_path)
        if self.H.shape != (3, 3) or not np.all(np.isfinite(self.H)):
            raise ConfigurationError(
                f"{h_path} 不是有效的 3x3 单应矩阵，请重新标定：\n"
                f"  python3 task11_follow.py --calibrate --h-file {h_path}\n"
                "仅做数值自检（不需要 H）：\n"
                "  python3 task11_follow.py --selfcheck")
        if not offline and "dummy" in os.path.basename(h_path).lower():
            raise ConfigurationError(
                f"{h_path} 是 --dummy-h 生成的占位矩阵，换算结果无物理意义，\n"
                "已硬性禁止用于真机跟随。请改用真实标定的 H.npy，或加 --offline。")

        self.gate = SafetyGate()
        self.tracker = TargetTracker(model_path, conf=conf)  # 先加载模型，避免线程泄漏
        self.grabber = RTSPGrabber(rtsp_url)
        self.supervisor = SafetySupervisor(
            require_ultrasonic=not offline)
        self.pid = DualLoopPID(target_distance=target_dist)
        self.sender = None
        self.offline = offline
        self._ip = ip
        self._state_port = state_port
        self._do_home = do_home
        self._fwd_scale = fwd_scale
        self._yaw_scale = 65534
        self._target_dist = target_dist
        self.state = self.STATE_IDLE
        self._vx, self._wz = 0.0, 0.0
        self._lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False
        self._running = True

    @staticmethod
    def _h_missing_help(h_path):
        return (
            f"未找到标定文件 {h_path}，无法把像素坐标换算成米，已拒绝启动。\n"
            "请先完成标定（机器人与相机保持静止）：\n"
            "  1) python3 task11_follow.py --calibrate\n"
            "     画面出现 Calibration 窗口后，依次点击四个地面点：\n"
            "       近左(-1,1) → 近右(1,1) → 远右(1,3) → 远左(-1,3)\n"
            f"  2) 标定通过后会在当前目录生成 {h_path}\n"
            "     标记不在默认位置时，用 --dst 指定真实坐标：\n"
            "       python3 task11_follow.py --calibrate --dst x1 y1 x2 y2 x3 y3 x4 y4\n"
            "  3) 再运行 python3 task11_follow.py\n"
            "如果你已有标定文件：\n"
            f"  python3 task11_follow.py --h-file /路径/到/{h_path}\n"
            "仅做数值自检（不需要 H）：\n"
            "  python3 task11_follow.py --selfcheck\n"
            "仅做脱机冒烟测试（无需真实标定）：\n"
            "  python3 task11_follow.py --offline --dummy-h")

    def _set_state(self, state):
        with self._lock:
            self.state = state

    def _safe_stop(self, reason):
        """所有异常路径的统一出口：清零指令 + PID 复位 + 停轴。"""
        self.pid.reset()
        with self._lock:
            self._vx, self._wz = 0.0, 0.0
        self._set_state(self.STATE_STOP)
        if self.sender is not None:
            self.sender.emergency_stop()
        return reason

    # ---- 控制线程：绝对节拍 20Hz ----
    def _control_loop(self):
        next_t = time.monotonic()
        while self._running:
            next_t += 0.05
            now = time.monotonic()
            # 控制线程独立看门狗：感知线程卡死时强制停轴
            if now - self.supervisor.last_perception_ok \
                    > self.supervisor.link_timeout:
                if self.sender is not None:
                    self.sender.emergency_stop()
                time.sleep(max(0, next_t - time.monotonic()))
                continue
            with self._lock:
                vx, wz, state = self._vx, self._wz, self.state
            # 零值也必须持续下发，AxisStream 负责保活与看门狗
            if self.sender is not None and state != self.STATE_IDLE:
                self.sender.send_twist(vx, wz)
            time.sleep(max(0, next_t - time.monotonic()))

    # ---- 感知线程：10Hz ----
    def run_vision(self):
        try:
            if not self.offline:
                self.sender = RobotSender(
                    ip=self._ip, state_port=self._state_port,
                    fwd_scale=self._fwd_scale, do_home=self._do_home)
                self.sender.prepare()
                self._set_state(self.STATE_TRACK)

            threading.Thread(target=self._control_loop, daemon=True).start()
            while self._running:
                try:
                    frame, age = self.grabber.get_latest()
                    if frame is None:
                        reason = self._safe_stop("帧过期/断流")
                        if self.offline:
                            print(f"[拦截] {reason}")
                        time.sleep(0.1)
                        continue

                    target = self.tracker.update(frame)
                    # 每个有效帧刷新感知心跳（不限于是否检测到目标）
                    self.supervisor.last_perception_ok = time.monotonic()

                    # 超声波每个感知周期都刷新，不依赖目标是否存在
                    if self.sender is not None:
                        st = self.sender.ctrl.get_state(max_age=0.2)
                        if st is not None and len(st.ultrasound) >= 1:
                            self.supervisor.update_ultrasonic(
                                float(st.ultrasound[0]))

                    if target is None:
                        ok, reason, decision = self.supervisor.check(
                            age, None, None)
                        reason = self._safe_stop(reason)
                        if self.offline:
                            print(f"[拦截] {reason}")
                        time.sleep(0.1)
                        continue

                    cx, cy = target[0], target[1]
                    X, Y = pixel_to_ground(self.H, cx, cy)
                    if not (np.isfinite(X) and np.isfinite(Y)) or Y <= 0:
                        reason = self._safe_stop("地面反投影无效")
                        if self.offline:
                            print(f"[拦截] {reason}")
                        time.sleep(0.1)
                        continue

                    error_x = math.atan2(X, Y)
                    distance = Y
                    ok, reason, decision = self.supervisor.check(
                        age, target, distance)
                    if not ok:
                        reason = self._safe_stop(reason)
                        if self.offline:
                            print(f"[拦截] {reason}")
                    else:
                        vx, wz = self.pid.update(
                            error_x, distance, active=True)
                        vx, wz = self.gate.apply(vx, wz, decision)
                        with self._lock:
                            self._vx, self._wz = vx, wz
                        self._set_state(self.STATE_TRACK)
                        if self.offline:
                            tag = "受限" if decision != "NORMAL" else "脱机"
                            af, ay = twist_to_axis(
                                vx, wz, self._fwd_scale, self._yaw_scale)
                            note = ""
                            if vx != 0 and abs(vx * self._fwd_scale) \
                                    < l3.AXIS_DEADZONE_FWD:
                                note = " ⚠前进轴值已抬出死区(最低速)"
                            elif wz != 0 and abs(wz * self._yaw_scale) \
                                    < l3.AXIS_DEADZONE_YAW:
                                note = " ⚠转向轴值已抬出死区(最低速)"
                            print(f"[{tag}] vx={vx:+.3f} wz={wz:+.3f} "
                                  f"dist={distance:.2f} 目标={self._target_dist:.2f} "
                                  f"轴=({af:+d},{ay:+d}){note} {reason}")
                except Exception as e:
                    reason = self._safe_stop(f"感知异常: {e}")
                    print(reason)
                time.sleep(0.1)
        finally:
            self.shutdown()

    def shutdown(self):
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        self._running = False
        self._safe_stop("shutdown")
        time.sleep(0.3)                      # 让轴流发完最后的 (0,0,0)
        if self.sender is not None:
            self.sender.stop()
        self.grabber.close()


# ======================================================================
# PID 数值自检（不连真机）
# ======================================================================
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
    print("\n全部自检通过 ✅（WZ_SIGN 方向需悬挂测试真机确认）")


# ======================================================================
# 入口
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Lite3 视觉跟随系统")
    l3.add_common_args(ap)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="脱机：只跑视觉与PID，不发真机指令")
    ap.add_argument("--no-home", action="store_true",
                    help="收尾只趴下，不归零")
    ap.add_argument("--url", default="rtsp://192.168.1.120:8554/test?tcp")
    ap.add_argument("--h-file", default="H.npy", help="单应矩阵文件")
    ap.add_argument("--dst", type=float, nargs=8,
                    metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
                    help="4 个地面标记在机器人坐标系中的真实坐标(米)，"
                         "默认 -1 1 1 1 1 3 -1 3")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO 权重")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="YOLO 检测置信度阈值（默认 0.5，人小/远可调低到 0.25）")
    ap.add_argument("--fwd-scale", type=int, default=40000,
                    help="vx→轴值系数（默认 40000：vx=0.4→16000，"
                         "vx=-0.2→-8000，均超过前进死区 6553）")
    ap.add_argument("--target-dist", type=float, default=2.0,
                    help="期望跟随距离，单位米（默认 2.0）")
    ap.add_argument("--dummy-h", action="store_true",
                    help="生成仅供脱机冒烟测试的占位矩阵；真机模式禁用")
    args = ap.parse_args()

    if args.selfcheck:
        if args.dummy_h:
            print("[提示] --dummy-h 与 --selfcheck 无关，已忽略")
        pid_selfcheck()
        return

    if args.dummy_h and args.calibrate:
        print("--dummy-h 不能与 --calibrate 同时使用：请先生成真实标定，"
              "或只用 --offline --dummy-h 做冒烟测试。")
        return
    if args.dummy_h and not args.offline:
        print("❌ 已拒绝启动：--dummy-h 生成的是占位矩阵，无物理意义，"
              "严禁用于真机跟随。\n"
              "   请改成：python3 task11_follow.py --offline --dummy-h\n"
              "   真机跟随请先做标定：python3 task11_follow.py --calibrate")
        return

    if args.calibrate:
        try:
            g = RTSPGrabber(args.url)
        except ConfigurationError as e:
            print(str(e))
            return
        print("标定姿态要求：必须与跟随运行时一致，且机器人保持静止。")
        print("  - 相机装在机器人上：请站立静止（建议先切好移动模式+低速步态，"
              "但不要发轴指令）")
        print("  - 外部固定相机：以机器人地面投影为原点，姿态影响较小，"
              "但仍建议站立静止")
        print("  - 趴着标定的 H 只能用于趴着；跟随是站立运行，趴着标定会失准")
        print("机器人与相机静止，3 秒后取标定帧...")
        time.sleep(3)
        frame, _ = g.get_latest()
        g.close()
        if frame is None:
            print("❌ 未取到有效帧，请检查 RTSP 流后重试："
                  f"python3 task10_gui.py --video-url {args.url}")
            return
        dst_pts = None
        if args.dst is not None:
            dst_pts = np.array(args.dst, dtype=np.float32).reshape(4, 2)
        calibrate_interactive(frame, dst_pts=dst_pts, save_path=args.h_file)
        return

    h_path = args.h_file
    if args.dummy_h:
        h_path = "H_dummy.npy"
        np.save(h_path, make_dummy_h())
        print("⚠ 使用占位矩阵 H_dummy.npy：仅用于脱机冒烟测试，"
              "距离/角度换算无物理意义，严禁真机使用。")

    try:
        app = VisualTrackingApp(
            args.url, h_path=h_path, ip=args.ip,
            state_port=args.state_port, model_path=args.model, conf=args.conf,
            offline=args.offline, do_home=not args.no_home,
            fwd_scale=args.fwd_scale, target_dist=args.target_dist)
    except ConfigurationError as e:
        print(str(e))
        return

    try:
        app.run_vision()
    except KeyboardInterrupt:
        print("\n用户中断，正在安全退出 ...")
        app.shutdown()
    except Exception as e:
        print(f"\n跟随启动/运行失败：{e}")
        print("排查：1) 状态帧 0x0901/0x0902  2) 起立与移动模式  "
              "3) RTSP 流  4) 先跑 --offline 验证视觉链路")
        app.shutdown()


if __name__ == "__main__":
    main()
