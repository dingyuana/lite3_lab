#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lite3_common.py —— 绝影 Lite3 UDP 控制公共模块
协议依据：《运动主机通讯接口 V1.1.0》
兼容 Python 3.8+

【协议读取与 98 状态码处理】
  UDP 前 12 字节是 CommandHead（code、payload 长度、type）；只有 code=0x0901
  才按 RobotStateUpload 读取。日志中的 98 位于 payload 第一个字段
  robot_basic_state，不是 CommandHead.code。98 表示运控程序启动后的初始待机，
  尚未触发首次起立；它不是故障，也不表示起立或回零已完成。收到起立指令且
  电量高于 20% 后，系统解除该状态并上报实际运行状态。98 帧中的遥测仍保留，
  但 98 本身不能用于确认站立或移动就绪。
  兼容两种 0x0901 格式：
  - 212 字节 payload：[4 字节 98 前缀][208 字节 RobotStateUpload]，解析时跳过前缀。
  - 208 字节 payload：直接按 RobotStateUpload 解析。

【v2 变更（2026-11）】
  ensure_sit：返回条件由“四膝 > 130° 或 is_sit”改为
              “状态稳定在 (1,0,0) 且四膝 > 130°，连续 stable_sec 秒”。
              若机器人正处于 (7,0,0) 趴下过渡态，先等自然稳定，避免重复发命令。
  ensure_home：发送 0x21010C05 前二次确认状态 == (1,0,0)；
              成功判据要求“先观察到 (17,0,0) 再回到 (1,0,0)”；
              超时未见到 17 时明确返回 False，不再用“四膝 > 130°”兜底宽恕。
"""

import ctypes
import math
import socket
import struct
import threading
import time
from typing import Optional

# ================= 指令码 =================
CMD_HEARTBEAT       = 0x21040001
CMD_HOME            = 0x21010C05
CMD_STAND_SIT       = 0x21010202
CMD_SOFT_ESTOP      = 0x21020C0E
CMD_MOVE_MODE       = 0x21010D06
CMD_STAND_MODE      = 0x21010D05
CMD_GAIT_LOW        = 0x21010300
CMD_GAIT_MID        = 0x21010307
CMD_GAIT_HIGH       = 0x21010303
CMD_GAIT_NORMAL_CRAWL = 0x21010406
CMD_GAIT_GRIP        = 0x21010402
CMD_GAIT_GENERAL     = 0x21010401
CMD_GAIT_HIGH_STEP   = 0x21010407
CMD_SPEAKER          = 0x2101030D
CMD_AXIS_FORWARD    = 0x21010130
CMD_AXIS_LATERAL    = 0x21010131
CMD_AXIS_YAW        = 0x21010135
CMD_AUTONOMOUS      = 0x21010C03
CMD_MANUAL          = 0x21010C02
CMD_VEL_X           = 0x0140
CMD_VEL_YAW         = 0x0141
CMD_VEL_Y           = 0x0145

CODE_ROBOT_STATE    = 0x0901
CODE_JOINT_ANGLE    = 0x0902
CODE_SPEAKER_STATUS = 0x11050F08
ROBOT_STATE_INITIAL_STANDBY = 98
# Backward-compatible name; 98 is an initial-standby state, not a connection error.
ROBOT_STATE_CONNECT_ERROR = ROBOT_STATE_INITIAL_STANDBY

# ================= 官方状态表 =================
STATE_TABLE = {
    (98, 0, 0):  "初始待机（运控程序启动后尚未触发首次起立）",
    (1, 0, 0):   "趴下",
    (1, 0, 11):  "正在执行向前跳",
    (4, 0, 0):   "准备起立",
    (5, 0, 0):   "正在起立",
    (6, 0, 0):   "力控站立+平地低速",
    (6, 0, 1):   "踏步/扭动(移动就绪)",
    (6, 0, 2):   "扭身体",
    (6, 0, 4):   "扭身跳",
    (6, 2, 0):   "力控站立+通用越障",
    (6, 2, 1):   "通用越障踏步",
    (6, 4, 0):   "力控站立+平地中速",
    (6, 4, 1):   "平地中速踏步",
    (6, 5, 0):   "力控站立+平地高速",
    (6, 5, 1):   "平地高速踏步",
    (6, 6, 0):   "力控站立+抓地越障",
    (6, 6, 1):   "抓地越障踏步",
    (6, 12, 1):  "太空步",
    (6, 13, 0):  "力控站立+高踏步",
    (6, 13, 1):  "高踏步踏步",
    (7, 0, 0):   "正在趴下",
    (8, 0, 0):   "失控保护",
    (9, 0, 0):   "姿态调整",
    (11, 0, 0):  "正在翻身",
    (16, 0, 1):  "AI 基础步态",
    (16, 16, 1): "AI 跳跃步态",
    (16, 18, 1): "AI 站立步态",
    (16, 20, 1): "AI 极速步态",
    (17, 0, 0):  "回零",
    (18, 0, 0):  "后空翻",
    (20, 0, 0):  "打招呼",
}

# 姿态判据（度）
SIT_DONE_DEG    = 130.0
STAND_MIN_DEG   = 60.0
STAND_MAX_DEG   = 90.0
SQUAT_MAX_DEG   = 130.0
STABLE_SECONDS  = 3.0

# 步态值（文档 1.3.1 robot_gait_state）
GAIT_VALID     = frozenset({0, 2, 4, 5, 6, 12, 13})
GAIT_LOW_SPEED = 0

# 轴指令死区
AXIS_DEADZONE_FWD = 6553
AXIS_DEADZONE_LAT = 12553
AXIS_DEADZONE_YAW = 9553

# 12 关节
JOINT_NAMES = [
    "左前侧摆", "左前髋", "左前膝",
    "右前侧摆", "右前髋", "右前膝",
    "左后侧摆", "左后髋", "左后膝",
    "右后侧摆", "右后髋", "右后膝",
]
KNEE_INDICES = [2, 5, 8, 11]

# 98 前缀
PREFIX98_SIZE = 4


# ================= 打包 =================
def pack_simple(code: int, value: int = 0, type_: int = 0) -> bytes:
    return struct.pack("<III", code & 0xFFFFFFFF,
                       value & 0xFFFFFFFF,
                       type_ & 0xFFFFFFFF)


def pack_complex(code: int, payload: bytes, type_: int = 1) -> bytes:
    return struct.pack("<III", code & 0xFFFFFFFF, len(payload),
                       type_ & 0xFFFFFFFF) + payload


# ================= 结构体 =================
class RobotStateUpload(ctypes.LittleEndianStructure):
    _fields_ = [
        ("robot_basic_state",         ctypes.c_int),
        ("robot_gait_state",          ctypes.c_int),
        ("robot_policy_state",        ctypes.c_int),
        ("rpy",                       ctypes.c_double * 3),
        ("rpy_vel",                   ctypes.c_double * 3),
        ("xyz_acc",                   ctypes.c_double * 3),
        ("pos_world",                 ctypes.c_double * 3),
        ("vel_world",                 ctypes.c_double * 3),
        ("vel_body",                  ctypes.c_double * 3),
        ("touch_down_and_stair_trot", ctypes.c_uint),
        ("is_charging",               ctypes.c_bool),
        ("error_state",               ctypes.c_uint),
        ("robot_motion_state",        ctypes.c_int),
        ("battery_level",             ctypes.c_double),
        ("task_state",                ctypes.c_int),
        ("is_robot_need_move",        ctypes.c_bool),
        ("zero_position_flag",        ctypes.c_bool),
        ("is_after_first_start",      ctypes.c_bool),
        ("is_voice_ctrl_enable",      ctypes.c_bool),
        ("ultrasound",                ctypes.c_double * 2),
    ]


class RobotJointAngle(ctypes.LittleEndianStructure):
    _fields_ = [("joint_angle", ctypes.c_double * 12)]


ROBOT_STATE_SIZE = ctypes.sizeof(RobotStateUpload)
JOINT_ANGLE_SIZE = ctypes.sizeof(RobotJointAngle)
assert ROBOT_STATE_SIZE == 208, ROBOT_STATE_SIZE
assert JOINT_ANGLE_SIZE == 96, JOINT_ANGLE_SIZE


def is_valid_robot_state(st) -> bool:
    return st is not None


def is_initial_standby(st) -> bool:
    return (st is not None
            and st.robot_basic_state == ROBOT_STATE_INITIAL_STANDBY)


def needs_home(st) -> bool:
    """Compatibility helper; initial standby alone never proves home is required."""
    return False


# ================= 状态表辅助 =================
def state_tuple(st):
    if st is None:
        return None, "无状态"
    t = (st.robot_basic_state, st.robot_gait_state, st.robot_motion_state)
    return t, STATE_TABLE.get(t, f"未知{t}")


def format_state_code(st) -> str:
    """Format the state tuple for CLI output, highlighting basic_state=98."""
    if st is None:
        return "?"
    code = str(state_tuple(st)[0])
    if is_initial_standby(st):
        return f"\033[31m{code}\033[0m"
    return code


def is_stand_still(st) -> bool:
    if st is None:
        return False
    return (st.robot_basic_state == 6
            and st.robot_gait_state == 0
            and st.robot_motion_state == 0)


def is_sit(st) -> bool:
    if st is None:
        return False
    return st.robot_basic_state == 1


def is_gait_low(st) -> bool:
    return st is not None and st.robot_gait_state == GAIT_LOW_SPEED


# ================= 控制器 =================
class Lite3Controller:
    def __init__(self, ip: str, port: int = 43893,
                 state_port: Optional[int] = None):
        self.addr = (ip, port)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = None
        if state_port:
            self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rx.bind(("0.0.0.0", state_port))
            self.rx.settimeout(0.1)

        self._stop = threading.Event()
        self._hb_thread = None
        self._axis_thread = None
        self._rx_thread = None

        self._axis_active = threading.Event()
        self._axis_value = 0
        self.axis_hz = 30

        self._state_lock = threading.Lock()
        self._last_state = None
        self._last_state_t = 0.0

        self._joint_lock = threading.Lock()
        self._last_joint = None
        self._last_joint_t = 0.0

        self._speaker_lock = threading.Lock()
        self._speaker_enabled = None

        self._stats_lock = threading.Lock()
        self.tx_count = 0
        self.rx_count = 0
        self.joint_rx_count = 0
        self.invalid_state_rx_count = 0
        self.icmp_unreachable = False
        self._send_error_logged = False

    def send(self, pkt: bytes):
        try:
            self.tx.sendto(pkt, self.addr)
            with self._stats_lock:
                self.tx_count += 1
        except ConnectionRefusedError:
            with self._stats_lock:
                self.icmp_unreachable = True
        except OSError as e:
            if not self._send_error_logged:
                print(f"[发送异常] {e}")
                self._send_error_logged = True

    def send_simple(self, code, value=0):
        self.send(pack_simple(code, value, 0))

    def send_complex(self, code, payload: bytes, type_=1):
        self.send(pack_complex(code, payload, type_))

    def _heartbeat_loop(self, hz):
        period = 1.0 / hz
        while not self._stop.is_set():
            self.send_simple(CMD_HEARTBEAT)
            time.sleep(period)

    def start_heartbeat(self, hz=5):
        if hz < 2:
            print(f"[警告] 心跳频率 {hz}Hz < 2Hz")
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, args=(hz,), daemon=True)
        self._hb_thread.start()

    def _axis_loop(self):
        period = 1.0 / self.axis_hz
        while not self._stop.is_set():
            if self._axis_active.is_set():
                self.send_simple(CMD_AXIS_FORWARD, self._axis_value)
                self.send_simple(CMD_AXIS_LATERAL, 0)
                self.send_simple(CMD_AXIS_YAW, 0)
            time.sleep(period)

    def start_axis_stream(self):
        self._axis_thread = threading.Thread(target=self._axis_loop, daemon=True)
        self._axis_thread.start()

    def set_forward(self, value: int):
        self._axis_value = value
        if value != 0:
            self._axis_active.set()
        else:
            self._axis_active.clear()

    def _rx_loop(self):
        while not self._stop.is_set():
            try:
                data, _ = self.rx.recvfrom(4096)
            except socket.timeout:
                continue
            except ConnectionRefusedError:
                with self._stats_lock:
                    self.icmp_unreachable = True
                continue
            except OSError:
                continue
            if len(data) < 12:
                continue
            code, size, type_ = struct.unpack_from("<III", data, 0)
            if code == CODE_SPEAKER_STATUS and type_ == 0:
                if len(data) == 12 and size in (0, 1):
                    with self._speaker_lock:
                        self._speaker_enabled = bool(size)
                continue
            if type_ != 1:
                continue
            payload = data[12:12 + size]

            if code == CODE_ROBOT_STATE:
                # 兼容 4 字节 98 前缀 + 208 字节 RobotStateUpload
                if len(payload) == PREFIX98_SIZE + ROBOT_STATE_SIZE:
                    st = RobotStateUpload.from_buffer_copy(
                        payload[PREFIX98_SIZE:PREFIX98_SIZE + ROBOT_STATE_SIZE])
                # 兼容旧固件：208 字节
                elif len(payload) == ROBOT_STATE_SIZE:
                    st = RobotStateUpload.from_buffer_copy(payload)
                else:
                    st = None

                if st is not None:
                    with self._state_lock:
                        self._last_state = st
                        self._last_state_t = time.time()
                    with self._stats_lock:
                        self.rx_count += 1
                        if not is_valid_robot_state(st):
                            self.invalid_state_rx_count += 1

            elif code == CODE_JOINT_ANGLE and len(payload) == JOINT_ANGLE_SIZE:
                ja = RobotJointAngle.from_buffer_copy(payload)
                with self._joint_lock:
                    self._last_joint = ja
                    self._last_joint_t = time.time()
                with self._stats_lock:
                    self.joint_rx_count += 1

    def start_state_receiver(self):
        if self.rx is None:
            print("[警告] 未配置 state_port，无法接收 0x0901/0x0902")
            return
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

    def get_state(self, max_age=0.5):
        with self._state_lock:
            if self._last_state and (time.time() - self._last_state_t) < max_age:
                return self._last_state
        return None

    def state_age(self):
        with self._state_lock:
            if self._last_state is None:
                return None
            return max(0.0, time.time() - self._last_state_t)

    def get_joint(self, max_age=0.5):
        with self._joint_lock:
            if self._last_joint and (time.time() - self._last_joint_t) < max_age:
                return self._last_joint
        return None

    def get_speaker_state(self):
        with self._speaker_lock:
            return self._speaker_enabled

    def clear_speaker_state(self):
        with self._speaker_lock:
            self._speaker_enabled = None

    def knee_angles(self, max_age=0.5):
        ja = self.get_joint(max_age)
        if ja is None:
            return None
        return [ja.joint_angle[i] for i in KNEE_INDICES]

    def knee_mean_deg(self, max_age=0.5) -> Optional[float]:
        k = self.knee_angles(max_age)
        if k is None:
            return None
        return math.degrees(sum(k) / 4.0)

    def stats(self) -> dict:
        with self._stats_lock:
            return {
                "tx_count": self.tx_count,
                "rx_count": self.rx_count,
                "joint_rx_count": self.joint_rx_count,
                "invalid_state_rx_count": self.invalid_state_rx_count,
                "icmp_unreachable": self.icmp_unreachable,
            }

    def close(self):
        self._stop.set()
        self._axis_active.clear()
        for t in (self._hb_thread, self._axis_thread, self._rx_thread):
            if t:
                t.join(timeout=1.0)
        self.tx.close()
        if self.rx:
            self.rx.close()


# ================= 日志辅助 =================
def _fmt_state(st) -> str:
    if st is None:
        return "状态=无"
    t, desc = state_tuple(st)
    return (f"状态={t} {desc}  电量={st.battery_level:6.2f}%  "
            f"前向速度={st.vel_body[0]:+.3f}  "
            f"姿态角=({st.rpy[0]:+.1f},{st.rpy[1]:+.1f},{st.rpy[2]:+.1f})°")


def wait_for_state(ctrl, predicate, timeout, poll=0.05) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = ctrl.get_state()
        if st is not None and predicate(st):
            return True
        time.sleep(poll)
    return False


# ================= 状态转换辅助 =================
def ensure_sit(ctrl, timeout=15.0, stable_sec=0.5, settle_wait=5.0) -> bool:
    """
    确保机器人稳定停在 (1,0,0) 趴下。

    判据：(robot_basic_state, robot_gait_state, robot_motion_state) == (1,0,0)
          且 四膝均值 > SIT_DONE_DEG，连续 stable_sec 秒。

    若机器人正处于 (7,0,0) 正在趴下 的过渡态，先给 settle_wait 秒让它
    自然稳定，避免在过渡态重复发 0x21010202。
    """
    def _is_stable_sit(st, m) -> bool:
        if st is None or m is None:
            return False
        return (st.robot_basic_state == 1
                and st.robot_gait_state == 0
                and st.robot_motion_state == 0
                and m > SIT_DONE_DEG)

    def _wait_until_stable(deadline):
        """等待 _is_stable_sit 连续 stable_sec 秒。返回 (ok, last_mean)。"""
        stable_t0 = None
        last_print = 0.0
        last_mean = None
        t_start = time.time()
        while time.time() < deadline:
            time.sleep(0.2)
            st = ctrl.get_state()
            m = ctrl.knee_mean_deg()
            if m is not None:
                last_mean = m
            elapsed = time.time() - t_start
            if elapsed - last_print >= 1.0:
                _, desc = state_tuple(st) if st else ("?", "?")
                m_str = f"{m:+.1f}°" if m is not None else "?"
                print(f"[趴下] {elapsed:4.1f}s  "
                      f"状态={format_state_code(st)} {desc}  四膝均值={m_str}")
                last_print = elapsed
            if _is_stable_sit(st, m):
                if stable_t0 is None:
                    stable_t0 = time.time()
                elif time.time() - stable_t0 >= stable_sec:
                    return True, m
            else:
                stable_t0 = None
        return False, last_mean

    st = ctrl.get_state()
    m = ctrl.knee_mean_deg()
    if _is_stable_sit(st, m):
        print(f"[趴下] 当前已稳定在 (1,0,0) 趴下（四膝={m:+.1f}°），跳过")
        return True

    # 已在趴下过渡态：先让机器人自然稳定，避免重复发命令
    in_motion = (st is not None and st.robot_basic_state == 7) or \
                (m is not None and m > SIT_DONE_DEG)
    if in_motion:
        print("[趴下] 机器人正在趴下过程中，等待自然稳定到 (1,0,0) ...")
        deadline = time.time() + min(settle_wait, timeout)
        ok, mm = _wait_until_stable(deadline)
        if ok:
            print(f"[趴下] ✓ 已稳定在 (1,0,0)，四膝均值={mm:+.1f}°")
            return True
        print("[趴下] 未自然稳定，补发 0x21010202 ...")

    print("[趴下] 发送 0x21010202 ...")
    ctrl.send_simple(CMD_STAND_SIT)

    deadline = time.time() + timeout
    ok, mm = _wait_until_stable(deadline)
    if ok:
        print(f"[趴下] ✓ 已稳定在 (1,0,0)，四膝均值={mm:+.1f}°")
        return True
    print(f"[趴下] ✗ {timeout:.0f}s 内未稳定在 (1,0,0)，最后一次四膝={mm}")
    return False


def ensure_home(ctrl, timeout_sit=15.0, timeout_home=12.0) -> bool:
    """
    归零流程：稳定趴下 (1,0,0) → 发 0x21010C05 → 观察 (17,0,0) → 回到 (1,0,0)。

    与 v1 的区别：
      * 发送 0x21010C05 前二次确认状态 == (1,0,0)，避免在 (7,0,0) 过渡态发命令被丢弃。
      * 成功判据要求“先见 17，再回到 (1,0,0)”；仅凭“四膝 > 130°”不再算成功。
      * 超时未见 17 时明确返回 False，让调用方打印失败。
    """
    print("[归零] 步骤 1/2：确保稳定趴下（归零必须在 (1,0,0) 状态执行）")
    if not ensure_sit(ctrl, timeout=timeout_sit):
        print("[归零] ✗ 未能稳定趴下，放弃归零")
        return False

    # 发送前二次确认，防止 ensure_sit 返回后状态机被其他客户端打断
    st = ctrl.get_state()
    cur = state_tuple(st)[0] if st is not None else None
    if cur != (1, 0, 0):
        print(f"[归零] ⚠ 发送前状态={format_state_code(st)}，"
              "等待回到 (1,0,0) ...")
        if not wait_for_state(
                ctrl, lambda s: state_tuple(s)[0] == (1, 0, 0), timeout=2.0):
            print("[归零] ✗ 无法稳定到 (1,0,0)，放弃归零")
            return False

    pre_home_mean = ctrl.knee_mean_deg()
    print(f"[归零] 步骤 2/2：归零前四膝均值={pre_home_mean:+.1f}°")
    print("[归零] 发送 0x21010C05 ...")
    ctrl.send_simple(CMD_HOME)

    seen_homing = False
    t0 = time.time()
    last_print = 0.0
    while time.time() - t0 < timeout_home:
        time.sleep(0.5)
        st = ctrl.get_state()
        m = ctrl.knee_mean_deg()
        elapsed = time.time() - t0
        if elapsed - last_print >= 1.0:
            _, desc = state_tuple(st) if st else ("?", "?")
            m_str = f"{m:+.1f}°" if m is not None else "?"
            print(f"[归零] {elapsed:4.1f}/{timeout_home:.0f}s  "
                  f"状态={format_state_code(st)} {desc}  "
                  f"四膝均值={m_str}")
            last_print = elapsed

        if st is not None:
            t = state_tuple(st)[0]
            if t == (17, 0, 0) and not seen_homing:
                print("[归零] ✓ 已观察到回零状态 (17,0,0)")
                seen_homing = True
            # 成功判据：先见 17，再回到 (1,0,0)
            if seen_homing and t == (1, 0, 0):
                print(f"[归零] ✓ 回零完成，已回到 (1,0,0) 趴下，四膝均值={m:+.1f}°")
                if pre_home_mean is not None and m is not None:
                    delta = m - pre_home_mean
                    print(f"[归零] 归零前后四膝均值：{pre_home_mean:+.1f}° → "
                          f"{m:+.1f}°（Δ {delta:+.1f}°）")
                    print("[归零] 注：归零会将关节零位偏移写入固件，四膝读数为"
                          "编码器置零后的新值，机器人仍停在趴下")
                return True

    if seen_homing:
        print(f"[归零] ~ 观察到 (17,0,0)，但 {timeout_home:.0f}s 内未回到 (1,0,0)，"
              f"请人工确认")
        return False

    st = ctrl.get_state()
    m = ctrl.knee_mean_deg()
    m_str = f"{m:+.1f}°" if m is not None else "?"
    print(f"[归零] ✗ {timeout_home:.0f}s 内未观察到 (17,0,0)，归零未生效")
    print(f"[归零] 发送后状态={format_state_code(st)}  四膝均值={m_str}")
    print("[归零] 排查：")
    print("       1) 发送时是否稳定在 (1,0,0)（见上方 [归零] 发送前状态检查）")
    print("       2) 是否有第二个 UDP 客户端在 43893 端口发指令")
    print("       3) 固件是否要求归零前先发其他命令，或 0x21010C05 payload 需要 value/type 字段")
    return False


# ================= 命令行参数 =================
def add_common_args(parser, state_port_default=43897):
    parser.add_argument("--ip", default="192.168.1.120", help="运动主机 IP")
    parser.add_argument("--state-port", type=int, default=state_port_default,
                        help="本地监听端口（默认 43897；0=不监听）")
    return parser