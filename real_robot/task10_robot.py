#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot state, posture transitions, keyboard motion, and front/rear guards."""

import math
import threading
import time

import lite3_common as l3

FORWARD_SPEED = 9000
LATERAL_SPEED = 16000
OBSTACLE_DISTANCE = 1.0
AXIS_HZ = 30.0
KEY_DIRECTIONS = frozenset({"up", "down", "left", "right"})
GAIT_COMMANDS = {
    "平地低速步态": (l3.CMD_GAIT_LOW, 0),
    "平地中速步态": (l3.CMD_GAIT_MID, 4),
    "平地高速步态": (l3.CMD_GAIT_HIGH, 5),
    "正常/匍匐切换": (l3.CMD_GAIT_NORMAL_CRAWL, None),
    "抓地越障步态": (l3.CMD_GAIT_GRIP, 6),
    "通用越障步态": (l3.CMD_GAIT_GENERAL, 2),
    "高踏步越障步态": (l3.CMD_GAIT_HIGH_STEP, 13),
}


def safe_axis_values(keys, forward_distance, backward_distance,
                     motion_allowed):
    """Return commanded forward/lateral axes and any ultrasonic warning."""
    if not motion_allowed:
        return 0, 0, ""

    forward = (("up" in keys) - ("down" in keys)) * FORWARD_SPEED
    lateral = (("right" in keys) - ("left" in keys)) * LATERAL_SPEED
    warning = ""

    if forward > 0 and (
            not math.isfinite(forward_distance)
            or forward_distance < OBSTACLE_DISTANCE):
        forward = 0
        warning = "前方距离小于 1 m 或数据无效，前进已拦截"
    elif forward < 0 and (
            not math.isfinite(backward_distance)
            or backward_distance < OBSTACLE_DISTANCE):
        forward = 0
        warning = "后方距离小于 1 m 或数据无效，后退已拦截"

    return forward, lateral, warning


def safe_follow_axes(forward, forward_distance, backward_distance):
    if forward > 0 and (
            not math.isfinite(forward_distance)
            or forward_distance < OBSTACLE_DISTANCE):
        return 0, "前方距离小于 1 m 或数据无效，跟踪前进已拦截"
    if forward < 0 and (
            not math.isfinite(backward_distance)
            or backward_distance < OBSTACLE_DISTANCE):
        return 0, "后方距离小于 1 m 或数据无效，跟踪后退已拦截"
    return forward, ""


class AxisStream:
    """Continuously send forward/lateral/yaw axes to keep the robot controlled."""

    def __init__(self, ctrl, hz=AXIS_HZ):
        self.ctrl = ctrl
        self.hz = hz
        self._axes = (0, 0, 0)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_axes(self, forward, lateral, yaw=0):
        with self._lock:
            self._axes = (forward, lateral, yaw)

    def _loop(self):
        period = 1.0 / self.hz
        while not self._stop.is_set():
            with self._lock:
                forward, lateral, yaw = self._axes
            self.ctrl.send_simple(l3.CMD_AXIS_FORWARD, forward)
            self.ctrl.send_simple(l3.CMD_AXIS_LATERAL, lateral)
            self.ctrl.send_simple(l3.CMD_AXIS_YAW, yaw)
            self._stop.wait(period)

    def stop(self):
        self.set_axes(0, 0)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None


class RobotController:
    """Owns the UDP controller and enforces posture/state/sensor gates."""

    def __init__(self, ip, state_port=43897, control_enabled=True):
        self.ctrl = l3.Lite3Controller(ip, state_port=state_port)
        self.control_enabled = control_enabled
        if control_enabled:
            self.ctrl.start_heartbeat(hz=5)
            self.ctrl.start_state_receiver()

        self._lock = threading.RLock()
        self._keys = set()
        self._follow_active = False
        self._follow_axes = (0, 0)
        self._posture = "unknown"
        self._motion_ready = False
        self._transition = ""
        self._status = "正在等待机器人状态帧"
        self._safety_status = "前后超声波防撞；左右侧向没有测距保护"
        self._speaker_command_status = "尚未查询扬声器状态"
        self._axis_stream = None
        self._closing = False
        self._cancel_transition = threading.Event()
        self._control_thread = threading.Thread(
            target=self._control_loop, daemon=True)
        self._control_thread.start()

    @staticmethod
    def _is_sit(st, knee_mean):
        return (st is not None
                and l3.state_tuple(st)[0] == (1, 0, 0)
                and knee_mean is not None
                and knee_mean > l3.SIT_DONE_DEG)

    @staticmethod
    def _is_stand(st, knee_mean):
        return (l3.is_stand_still(st)
                and knee_mean is not None
                and l3.STAND_MIN_DEG <= knee_mean <= l3.STAND_MAX_DEG)

    @staticmethod
    def _is_motion_posture(st, knee_mean):
        return (st is not None
                and st.robot_basic_state == 6
                and knee_mean is not None
                and l3.STAND_MIN_DEG <= knee_mean <= l3.SQUAT_MAX_DEG)

    def _control_loop(self):
        while True:
            with self._lock:
                if self._closing:
                    return

            raw_st = self.ctrl.get_state()
            st = raw_st
            state_available = raw_st is not None
            state_98 = l3.is_initial_standby(raw_st)
            knee_mean = self.ctrl.knee_mean_deg()
            front = (float(raw_st.ultrasound[0]) if raw_st is not None
                     else float("nan"))
            back = (float(raw_st.ultrasound[1]) if raw_st is not None
                    else float("nan"))

            with self._lock:
                if not self._transition:
                    if self._is_sit(st, knee_mean):
                        self._posture = "sit"
                        self._motion_ready = False
                    elif (self._posture in ("unknown", "sit")
                          and self._is_motion_posture(st, knee_mean)):
                        self._posture = "stand"
                        self._motion_ready = False
                    elif state_98:
                        self._posture = "unknown"
                        self._motion_ready = False

                busy = bool(self._transition)
                posture_valid = self._is_motion_posture(raw_st, knee_mean)
                motion_allowed = (state_available and posture_valid
                                  and self._posture == "stand"
                                  and self._motion_ready and not busy)

                if not state_available:
                    self._keys.clear()
                    self._status = "状态帧超时/未收到，运动已停止"
                elif busy:
                    self._status = self._transition
                elif state_98:
                    self._keys.clear()
                    self._status = (
                        "初始待机（98）：尚未首次起立；电量高于20%时"
                        "按空格首次起立，或使用回零按钮"
                    )
                elif self._posture == "sit":
                    self._keys.clear()
                    self._status = "趴下状态：方向键禁用，按空格起立"
                elif self._posture != "stand":
                    self._keys.clear()
                    self._status = "姿态未确认，运动已禁用"
                elif not self._motion_ready:
                    self._keys.clear()
                    self._status = "已站立但运动模式未就绪，按空格准备"
                elif not posture_valid:
                    self._keys.clear()
                    self._status = "站立状态异常，运动已停止"
                else:
                    self._status = "站立就绪：按住方向键移动，松开即停止"

                if self._follow_active and not motion_allowed:
                    self._follow_active = False
                    self._follow_axes = (0, 0)
                    self._status = "人体跟踪已停止：机器人状态不再满足运动条件"

                if self._follow_active:
                    forward, yaw = self._follow_axes
                    command_forward, warning = safe_follow_axes(
                        forward, front, back)
                    command_lateral = 0
                    command_yaw = 0 if warning else yaw
                    self._status = "人体跟踪中：按 T 或停止按钮退出"
                    self._safety_status = (
                        warning or "人体跟踪中：前后超声波防撞保护开启")
                else:
                    command_forward, command_lateral, warning = safe_axis_values(
                        self._keys if motion_allowed else set(),
                        front, back, motion_allowed)
                    command_yaw = 0
                    self._safety_status = (
                        warning or
                        "前/后测距未报警；左右侧向没有测距保护"
                    )
                    if not state_available:
                        self._safety_status = (
                            "状态数据超时/未收到；所有移动均已拦截"
                        )
                    elif state_98:
                        self._safety_status = (
                            "初始待机（98）：未首次起立，方向键禁用"
                        )
                    elif not warning:
                        nearby = []
                        if not math.isfinite(front) or front < OBSTACLE_DISTANCE:
                            nearby.append("前方近障/距离无效")
                        if not math.isfinite(back) or back < OBSTACLE_DISTANCE:
                            nearby.append("后方近障/距离无效")
                        if nearby:
                            self._safety_status = (
                                "；".join(nearby)
                                + "；左右侧向没有测距保护"
                            )
                stream = self._axis_stream

            if stream is not None:
                stream.set_axes(
                    command_forward, command_lateral, command_yaw)
            time.sleep(0.05)

    def start_tracking(self):
        with self._lock:
            if not getattr(self, "control_enabled", True):
                raise RuntimeError("脱机模式不会发送真机运动指令")
            if self._closing:
                raise RuntimeError("机器人正在关闭")
            if self._transition:
                raise RuntimeError("机器人正在切换状态，请稍后再启动跟踪")
            if (self._posture != "stand" or not self._motion_ready
                    or getattr(self, "_axis_stream", None) is None):
                raise RuntimeError("请先按空格起立并完成移动模式准备，再启动跟踪")
            self._keys.clear()
            self._follow_axes = (0, 0)
            self._follow_active = True
            self._status = "人体跟踪已启动，正在搜索目标"

    def set_tracking_axes(self, forward, yaw):
        with self._lock:
            if not getattr(self, "_follow_active", False) or self._closing:
                return
            self._follow_axes = (
                max(min(int(forward), 32767), -32767),
                max(min(int(yaw), 32767), -32767),
            )

    def stop_tracking(self):
        with self._lock:
            was_active = getattr(self, "_follow_active", False)
            self._follow_active = False
            self._follow_axes = (0, 0)
            stream = getattr(self, "_axis_stream", None) if was_active else None
            if was_active:
                self._status = "人体跟踪已停止，运动轴归零"
        if stream is not None:
            stream.set_axes(0, 0)

    def set_direction(self, direction, pressed):
        if direction not in KEY_DIRECTIONS:
            return
        with self._lock:
            if pressed and not getattr(self, "control_enabled", True):
                self._status = "脱机模式：手动运动已禁用"
                return
            if pressed and getattr(self, "_follow_active", False):
                self._status = "人体跟踪中，方向键已禁用；按 T 停止跟踪"
                return
            if pressed:
                if (self._posture != "stand" or not self._motion_ready
                        or self._transition):
                    if self._posture == "sit":
                        self._status = "趴下状态禁止移动，请先按空格起立"
                    else:
                        self._status = "机器人未处于可移动的稳定站立状态"
                    return
                self._keys.add(direction)
            else:
                self._keys.discard(direction)

    def stop_motion(self):
        self.stop_tracking()
        with self._lock:
            self._keys.clear()
            stream = self._axis_stream
        if stream is not None:
            stream.set_axes(0, 0)
        with self._lock:
            self._status = "已停止：所有运动轴归零"

    def switch_gait(self, gait_name):
        if gait_name not in GAIT_COMMANDS:
            raise ValueError(f"不支持的步态：{gait_name}")
        self.stop_tracking()
        if not getattr(self, "control_enabled", True):
            with self._lock:
                self._status = "脱机模式：步态切换已禁用"
            return

        st = self.ctrl.get_state()
        knee_mean = self.ctrl.knee_mean_deg()
        with self._lock:
            if self._closing:
                return
            if self._transition:
                self._status = "当前操作进行中，请等待完成后切换步态"
                return
            if (self._posture != "stand" or not self._motion_ready
                    or st is None or st.robot_basic_state != 6
                    or knee_mean is None
                    or not l3.STAND_MIN_DEG <= knee_mean <= l3.SQUAT_MAX_DEG):
                self._status = (
                    "步态切换要求非 AI 状态且已进入移动模式；"
                    "请先站立并准备移动模式"
                )
                return

        command, expected_gait = GAIT_COMMANDS[gait_name]
        self._start_transition(
            f"切换步态：{gait_name}",
            lambda: self._switch_gait(
                gait_name, command, expected_gait, st.robot_gait_state))

    def _switch_gait(self, gait_name, command, expected_gait, previous_gait):
        with self._lock:
            stream = self._axis_stream
        if stream is not None:
            stream.set_axes(0, 0)

        stopped = lambda st, _knee: (
            st is not None
            and st.robot_basic_state == 6
            and abs(st.vel_body[0]) < 0.05
            and abs(st.vel_body[1]) < 0.05
        )
        if not self._wait_until(stopped, 0.3, 5.0):
            raise RuntimeError("未能确认机器人停稳，未切换步态")

        self.ctrl.send_simple(command)
        if expected_gait is None:
            gait_changed = lambda st, _knee: (
                st is not None
                and st.robot_basic_state == 6
                and st.robot_gait_state != previous_gait
            )
            description = "步态已切换"
            predicate = gait_changed
        else:
            predicate = lambda st, _knee: (
                st is not None
                and st.robot_basic_state == 6
                and st.robot_gait_state == expected_gait
            )
            description = f"步态已切换：{gait_name}"

        if not self._wait_until(predicate, 0.3, 8.0):
            raise RuntimeError(f"未确认目标步态：{gait_name}")
        return "stand", True, description

    def speaker_command(self, value):
        if value not in (0, 1, 2):
            raise ValueError("扬声器指令值必须是0（关闭）、1（打开）或2（查询）")
        if not getattr(self, "control_enabled", True):
            with self._lock:
                self._speaker_command_status = "脱机模式：扬声器控制已禁用"
            return
        with self._lock:
            if self._closing:
                return
            self._speaker_command_status = {
                0: "已发送关闭指令，等待状态上报",
                1: "已发送打开指令，等待状态上报",
                2: "正在查询扬声器状态",
            }[value]
        self.ctrl.clear_speaker_state()
        self.ctrl.send_simple(l3.CMD_SPEAKER, value)

    def toggle_posture(self):
        self.stop_tracking()
        if not getattr(self, "control_enabled", True):
            with self._lock:
                self._status = "脱机模式：姿态控制已禁用"
            return
        with self._lock:
            if self._closing:
                return
            if self._transition:
                self._status = "姿态切换进行中，请等待完成"
                return

        st = self.ctrl.get_state()
        knee_mean = self.ctrl.knee_mean_deg()
        if st is None or knee_mean is None:
            with self._lock:
                self._status = "未收到有效状态/关节帧，拒绝姿态切换"
            return

        if self._is_sit(st, knee_mean):
            self._start_transition("起立中，请保持周围空间安全", self._stand_and_prepare)
            return

        if l3.is_initial_standby(st):
            if st.battery_level <= 20.0:
                with self._lock:
                    self._status = (
                        "初始待机（98）：电量不高于20%，禁止首次起立"
                    )
                return
            self._start_transition(
                "初始待机（98）：首次起立中，请保持周围空间安全",
                self._stand_and_prepare)
            return

        with self._lock:
            posture = self._posture
            motion_ready = self._motion_ready

        if posture == "stand" and motion_ready:
            self._start_transition("正在趴下，请勿操作", self._sit_down)
        elif self._is_motion_posture(st, knee_mean):
            self._start_transition("正在准备移动模式", self._prepare_motion_mode)
        else:
            with self._lock:
                self._status = "当前姿态不稳定或未识别，拒绝切换；请先检查机器人状态"

    def home(self):
        self.stop_tracking()
        if not getattr(self, "control_enabled", True):
            with self._lock:
                self._status = "脱机模式：回零已禁用"
            return
        with self._lock:
            if self._closing:
                return
            if self._transition:
                self._status = "当前操作进行中，请等待完成后再回零"
                return

        st = self.ctrl.get_state()
        state = l3.state_tuple(st)[0] if st is not None else None
        if state != (1, 0, 0) and not l3.is_initial_standby(st):
            with self._lock:
                self._status = (
                    "回零需要趴下状态 (1,0,0) 或状态码98；"
                    "其它状态请先按空格完成姿态切换"
                )
            return
        self._start_transition("正在回零", self._home)

    def _home(self):
        self._stop_axis_stream()
        print("[task10] 发送回零指令 0x21010C05")
        self.ctrl.send_simple(l3.CMD_HOME)

        seen_homing = False
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            if self._cancel_transition.is_set():
                raise RuntimeError("回零操作已取消")
            st = self.ctrl.get_state()
            state = l3.state_tuple(st)[0] if st is not None else None
            if state == (17, 0, 0):
                seen_homing = True
            elif seen_homing and state == (1, 0, 0):
                return "sit", False, "回零完成：已返回趴下状态 (1,0,0)"
            time.sleep(0.1)

        if seen_homing:
            raise RuntimeError("已观察到 (17,0,0)，但未确认返回 (1,0,0)")
        raise RuntimeError("12 秒内未观察到回零状态 (17,0,0)")

    def _start_transition(self, label, action):
        with self._lock:
            if self._closing or self._transition:
                self._status = "机器人正在切换状态，请等待"
                return
            self._keys.clear()
            self._posture = "transition"
            self._motion_ready = False
            self._transition = label
            self._status = label
            self._cancel_transition.clear()
            worker = threading.Thread(
                target=self._run_transition, args=(label, action), daemon=True)
            self._worker = worker
            worker.start()

    def _run_transition(self, label, action):
        try:
            posture, ready, result = action()
            with self._lock:
                if not self._closing:
                    self._posture = posture
                    self._motion_ready = ready
                    self._status = result
        except Exception as exc:
            self._stop_axis_stream()
            print(f"[task10] {label}失败：{exc}")
            with self._lock:
                if not self._closing:
                    self._posture = "unknown"
                    self._motion_ready = False
                    self._status = f"{label}失败：{exc}"
        finally:
            with self._lock:
                self._transition = ""

    def _wait_until(self, predicate, stable_seconds, timeout):
        started = time.monotonic()
        stable_started = None
        while time.monotonic() - started < timeout:
            if self._cancel_transition.is_set():
                raise RuntimeError("操作已取消")
            st = self.ctrl.get_state()
            knee_mean = self.ctrl.knee_mean_deg()
            if predicate(st, knee_mean):
                if stable_started is None:
                    stable_started = time.monotonic()
                elif time.monotonic() - stable_started >= stable_seconds:
                    return True
            else:
                stable_started = None
            time.sleep(0.1)
        return False

    def _stand_and_prepare(self):
        st = self.ctrl.get_state()
        if l3.is_initial_standby(st) and st.battery_level <= 20.0:
            raise RuntimeError("初始待机（98）下电量不高于20%，禁止首次起立")
        self.ctrl.send_simple(l3.CMD_STAND_SIT)
        if not self._wait_until(self._is_stand, 3.0, 30.0):
            raise RuntimeError("30 秒内未确认稳定站立")
        return self._prepare_motion_mode()

    def _prepare_motion_mode(self):
        if self._cancel_transition.is_set():
            raise RuntimeError("操作已取消")
        self.ctrl.send_simple(l3.CMD_MOVE_MODE)
        time.sleep(0.5)
        if self._cancel_transition.is_set():
            raise RuntimeError("操作已取消")

        stream = AxisStream(self.ctrl)
        with self._lock:
            if self._closing or self._cancel_transition.is_set():
                raise RuntimeError("操作已取消")
            self._axis_stream = stream
            stream.start()
        if self._cancel_transition.is_set():
            raise RuntimeError("操作已取消")
        self.ctrl.send_simple(l3.CMD_GAIT_LOW)

        ready = lambda st, knee: (
            st is not None
            and st.robot_basic_state == 6
            and st.robot_gait_state == l3.GAIT_LOW_SPEED
            and knee is not None
            and l3.STAND_MIN_DEG <= knee <= l3.SQUAT_MAX_DEG
        )
        if not self._wait_until(ready, 0.5, 8.0):
            raise RuntimeError("8 秒内未确认低速步态就绪")
        return "stand", True, "站立就绪：方向键可移动，空格趴下"

    def _sit_down(self):
        with self._lock:
            stream = self._axis_stream
            self._axis_stream = None
        if stream is not None:
            stream.set_axes(0, 0)
            time.sleep(0.5)
            stream.stop()

        self.ctrl.send_simple(l3.CMD_STAND_SIT)
        if not self._wait_until(self._is_sit, 0.5, 20.0):
            raise RuntimeError("20 秒内未确认稳定趴下")
        return "sit", False, "已稳定趴下：方向键禁用，按空格起立"

    def _stop_axis_stream(self):
        with self._lock:
            stream = self._axis_stream
            self._axis_stream = None
        if stream is not None:
            stream.set_axes(0, 0)
            time.sleep(0.25)
            stream.stop()

    def snapshot(self):
        st = self.ctrl.get_state()
        knee_mean = self.ctrl.knee_mean_deg()
        state_98 = l3.is_initial_standby(st)
        speaker_enabled = self.ctrl.get_speaker_state()
        with self._lock:
            posture = self._posture
            state_text = (
                "初始待机（98）：尚未触发首次起立；不能据此判断回零完成"
                if state_98 else
                "无状态帧" if st is None else
                l3.state_tuple(st)[1]
            )
            return {
                "status": self._status,
                "safety": self._safety_status,
                "tracking": self._follow_active,
                "posture": posture,
                "motion_ready": self._motion_ready,
                "transition": self._transition,
                "initial_standby": state_98,
                "state": state_text,
                "state_tuple": l3.state_tuple(st)[0] if st is not None else None,
                "knee_mean": knee_mean,
                "front": float(st.ultrasound[0]) if st is not None else None,
                "back": float(st.ultrasound[1]) if st is not None else None,
                "state_age": self.ctrl.state_age(),
                "speaker_enabled": speaker_enabled,
                "speaker_command_status": self._speaker_command_status,
                "forward_speed": float(st.vel_body[0]) if st is not None else None,
                "lateral_speed": float(st.vel_body[1]) if st is not None else None,
            }

    def close(self):
        with self._lock:
            if self._closing:
                return
            self._closing = True
            self._keys.clear()
            self._follow_active = False
            self._follow_axes = (0, 0)
            self._cancel_transition.set()
            worker = getattr(self, "_worker", None)
            stream = self._axis_stream
            self._axis_stream = None

        self._control_thread.join(timeout=1.0)
        if stream is not None:
            stream.set_axes(0, 0)
            time.sleep(0.25)
            stream.stop()
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.0)
        self.ctrl.close()
