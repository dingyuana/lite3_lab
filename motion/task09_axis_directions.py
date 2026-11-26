#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task09_axis_directions.py —— 起立 → 切移动模式 + 低速步态 → 前后左右定时运动 → 趴下 → 归零
兼容 Python 3.8+，用 python3 运行。

【完整流程】
  阶段 A 起立：
    趴下（四膝 > 130°）→ 发 0x21010202 → 等 (6,0,0) 站立 + 四膝 [60°,90°]
    连续 3 秒。期间只发心跳，不发轴指令。
  阶段 B 切模式 + 步态：
    0x21010D06 移动模式 → 起 30Hz 轴流 → 0x21010300 低速步态
    验收 gait=0 且四膝 [60°,130°]。
  阶段 C 前后左右运动：
    前进 3 秒 → 停止 2 秒 → 后退 3 秒 → 停止 2 秒
    → 左移 3 秒 → 停止 2 秒 → 右移 3 秒 → 停止 2 秒
    每 0.5 秒打印身体坐标速度和世界坐标位置，动作之间明确归零停顿。
  收尾：
    调用 lite3_common.ensure_home()：
      1) 先 ensure_sit：发 0x21010202 趴下，等四膝 > 130°
      2) 再发 0x21010C05 归零，等 (17,0,0) 回零后回到趴下
      3) 打印归零前后四膝均值变化
  Ctrl-C 中断时同样走“归零轴值 → 趴下 → 归零”流程。

【为什么停车要保活】
  运动时轴流持续发送当前方向轴值，归零后机器人靠惯性减速。
  如果立刻停轴流，250ms 轴指令超时会让底层接管，可能突然失去支撑。
  换向时保持轴流持续发送 0 两秒，让相邻动作之间有明确停顿。

【安全】
  - 每个方向运动 3 秒，前后左右均留出足够空间
  - 每次换向前轴值归零并等待 2 秒
  - Ctrl-C 会安全停车并进入收尾流程
  - 手边准备软急停：send_simple(0x21020C0E)

【用法】
  python3 task09_axis_directions.py
  python3 task09_axis_directions.py --no-home                   # 只趴下不归零
"""

import argparse
import math
import threading
import time
import lite3_common as l3

STABLE_SECONDS = 3.0
STAND_TIMEOUT  = 30.0

MOVE_SECONDS   = 3.0
REST_SECONDS   = 2.0
AXIS_FORWARD   = 12000
AXIS_LATERAL   = 16000


class AxisStream:
    """以 ≥20Hz 持续发送三轴轴指令，运动轴值可动态改。"""
    def __init__(self, ctrl, hz=30.0):
        self.ctrl = ctrl
        self.hz = hz
        self.forward = 0
        self.lateral = 0
        self._axes_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._t = None
        self.frames = 0

    def start(self):
        self._stop_event.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        period = 1.0 / self.hz
        while not self._stop_event.is_set():
            with self._axes_lock:
                forward = self.forward
                lateral = self.lateral
            self.ctrl.send_simple(l3.CMD_AXIS_FORWARD, forward)
            self.ctrl.send_simple(l3.CMD_AXIS_LATERAL, lateral)
            self.ctrl.send_simple(l3.CMD_AXIS_YAW, 0)
            self.frames += 1
            time.sleep(period)

    def set_axes(self, forward: int, lateral: int):
        with self._axes_lock:
            self.forward = forward
            self.lateral = lateral

    def stop(self):
        self._stop_event.set()
        if self._t:
            self._t.join(timeout=1.0)


def _print_pose(ctrl, tag="姿态"):
    st = ctrl.get_state()
    if st is not None:
        t, desc = l3.state_tuple(st)
        print(f"[{tag}] 状态={t} {desc}")
        print(f"[{tag}] 策略={st.robot_policy_state}  "
              f"电量={st.battery_level:.2f}%  "
              f"前向速度={st.vel_body[0]:+.3f} m/s  "
              f"位置=({st.pos_world[0]:+.3f}, {st.pos_world[1]:+.3f}) m")
        print(f"[{tag}] 姿态角=({st.rpy[0]:+.1f},{st.rpy[1]:+.1f},"
              f"{st.rpy[2]:+.1f})°")
    else:
        print(f"[{tag}] 状态=无（0x0901 未到达）")

    m = ctrl.knee_mean_deg()
    k = ctrl.knee_angles()
    if m is not None and k is not None:
        print(f"[{tag}] 四膝均值={m:+.1f}°  "
              f"四膝 deg=[{', '.join(f'{math.degrees(x):+.1f}' for x in k)}]")
    else:
        print(f"[{tag}] 四膝均值=无（0x0902 未到达）")


def _is_stand_success(st, mean_deg) -> bool:
    if st is None or mean_deg is None:
        return False
    if not l3.is_stand_still(st):
        return False
    return l3.STAND_MIN_DEG <= mean_deg <= l3.STAND_MAX_DEG


# ===================== 阶段 A：起立 =====================
def _wait_stand(ctrl, timeout):
    t0 = time.time()
    stable_t0 = None
    last_mean = None
    last_dot = 0.0
    while time.time() - t0 < timeout:
        time.sleep(0.5)
        st = ctrl.get_state()
        m = ctrl.knee_mean_deg()
        if st is None or m is None:
            continue
        last_mean = m
        elapsed = time.time() - t0
        if elapsed - last_dot >= 3.0:
            t, desc = l3.state_tuple(st)
            print(f"[起立] ... 等待中（{elapsed:.0f}s，状态={t} {desc}，"
                  f"四膝均值={m:+.1f}°）")
            last_dot = elapsed
        if _is_stand_success(st, m):
            if stable_t0 is None:
                stable_t0 = time.time()
            elif time.time() - stable_t0 >= STABLE_SECONDS:
                return True, m
        else:
            stable_t0 = None
    return False, last_mean


def _phase_a_stand(ctrl) -> bool:
    print("=" * 108)
    print("[阶段A] 起立到稳定站立")
    print("=" * 108)
    st = ctrl.get_state()
    mean_deg = ctrl.knee_mean_deg()
    if st is None or mean_deg is None:
        print("[阶段A] ✗ 未收到 0x0901/0x0902")
        print("[阶段A] 先跑 task01/task02 确认状态帧和关节帧正常")
        return False
    if _is_stand_success(st, mean_deg):
        print(f"[阶段A] ✓ 当前已是 (6,0,0) 稳定站立（四膝={mean_deg:+.1f}°），"
              f"跳过起立")
        return True
    if l3.is_sit(st) or mean_deg > l3.SIT_DONE_DEG:
        print(f"[阶段A] 当前趴下（四膝={mean_deg:+.1f}°），发 0x21010202 起立")
    else:
        t, desc = l3.state_tuple(st)
        print(f"[阶段A] 当前状态={t} {desc}，四膝={mean_deg:+.1f}°")
        print("[阶段A] 不在趴下也不在稳定站立，仍发 0x21010202 触发状态机")
    print("[阶段A] 起立期间只发心跳，不发轴指令"
          "（否则会切到 121.4° 移动就绪态）")
    print("[阶段A] 发送 0x21010202 ...")
    if l3.is_initial_standby(st) and st.battery_level <= 20.0:
        print("[阶段A] ✗ 初始待机(98)下电量不高于20%，禁止首次起立")
        return False
    ctrl.send_simple(l3.CMD_STAND_SIT)
    ok, m = _wait_stand(ctrl, timeout=STAND_TIMEOUT)
    if not ok:
        print(f"[阶段A] ✗ {STAND_TIMEOUT:.0f}s 内未进入稳定站立，"
              f"最后一次四膝={m}")
        print("[阶段A] 排查：")
        print("       1) 是否有其他客户端在发指令")
        print("       2) 机器人是否被外力阻挡")
        print("       3) 重发 0x21010202，或断电重启")
        return False
    print(f"[阶段A] ✓ 起立成功，四膝均值={m:+.1f}°，状态=(6,0,0) 稳定站立")
    return True


# ===================== 阶段 B：切模式 =====================
def _wait_gait(ctrl, target_gait, timeout, tag):
    t0 = time.time()
    last_print = 0.0
    while time.time() - t0 < timeout:
        time.sleep(0.5)
        st = ctrl.get_state()
        m = ctrl.knee_mean_deg()
        if st is not None:
            elapsed = time.time() - t0
            if elapsed - last_print >= 1.0:
                t, desc = l3.state_tuple(st)
                m_str = f"{m:+.1f}°" if m is not None else "?"
                print(f"[{tag}] {elapsed:4.1f}/{timeout:.0f}s  "
                      f"状态={t} {desc}  四膝均值={m_str}")
                last_print = elapsed
            if st.robot_gait_state == target_gait:
                return True
    return False


def _phase_b_move_gait(ctrl, hold_hz, stream) -> bool:
    print()
    print("=" * 108)
    print("[阶段B] 切移动模式 + 低速步态")
    print("=" * 108)

    print("[阶段B] 切换移动模式 0x21010D06 ...")
    ctrl.send_simple(l3.CMD_MOVE_MODE)
    time.sleep(0.5)
    st = ctrl.get_state()
    if st is not None:
        t, desc = l3.state_tuple(st)
        m = ctrl.knee_mean_deg()
        m_str = f"{m:+.1f}°" if m is not None else "?"
        print(f"[阶段B] 切移动模式后：状态={t} {desc}  四膝均值={m_str}")

    print(f"[阶段B] 启动 {hold_hz:.0f}Hz 轴流（forward=0，不动）")
    stream.set_axes(0, 0)
    stream.start()
    time.sleep(0.3)

    print("[阶段B] 发送 0x21010300 切平地低速步态 ...")
    ctrl.send_simple(l3.CMD_GAIT_LOW)
    ok = _wait_gait(ctrl, l3.GAIT_LOW_SPEED, timeout=5.0, tag="切步态")
    if not ok:
        st = ctrl.get_state()
        cur = st.robot_gait_state if st else "?"
        print(f"[阶段B] ✗ 5s 内未切到 gait_state={l3.GAIT_LOW_SPEED}，当前={cur}")
        print("[阶段B] 排查：")
        print("       1) 是否在 AI 状态（basic=16）？AI 状态下步态切换无效")
        print("       2) 是否在移动模式？刚才已切")
        return False
    print(f"[阶段B] ✓ 步态已切换为平地低速（gait_state={l3.GAIT_LOW_SPEED}）")

    st = ctrl.get_state()
    m = ctrl.knee_mean_deg()
    if st is None or m is None:
        print("[阶段B] ⚠ 切步态后无法读取状态")
        return False
    if st.robot_gait_state != l3.GAIT_LOW_SPEED:
        print(f"[阶段B] ✗ gait_state={st.robot_gait_state}，不是 0")
        return False
    if not (l3.STAND_MIN_DEG <= m <= l3.SQUAT_MAX_DEG):
        print(f"[阶段B] ⚠ 四膝={m:+.1f}° 不在 "
              f"[{l3.STAND_MIN_DEG:.0f}°, {l3.SQUAT_MAX_DEG:.0f}°]")
        return False
    print(f"[阶段B] ✓ 就绪：gait=0，四膝={m:+.1f}°")
    return True


# ===================== 阶段 C：前后左右运动 =====================
def _run_direction(ctrl, stream, label, forward, lateral) -> None:
    print()
    print("=" * 108)
    print(f"[运动] {label} {MOVE_SECONDS:.0f} 秒"
          f"（forward={forward}, lateral={lateral}）")
    print("=" * 108)

    stream.set_axes(forward, lateral)
    t0 = time.monotonic()
    last_print = -0.5
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= MOVE_SECONDS:
            break
        time.sleep(min(0.1, MOVE_SECONDS - elapsed))
        st = ctrl.get_state()
        if st is None:
            continue
        elapsed = time.monotonic() - t0
        if elapsed - last_print >= 0.5:
            t, desc = l3.state_tuple(st)
            print(f"[{label}] {elapsed:4.1f}s  "
                  f"vel_body=({st.vel_body[0]:+.3f},"
                  f"{st.vel_body[1]:+.3f}) m/s  "
                  f"pos_world=({st.pos_world[0]:+.3f},"
                  f"{st.pos_world[1]:+.3f}) m  状态={t} {desc}")
            last_print = elapsed


def _wait_between_directions(stream, reason) -> None:
    stream.set_axes(0, 0)
    print(f"[换向] {reason}：forward/lateral 归 0，原地停止 {REST_SECONDS:.0f} 秒")
    deadline = time.monotonic() + REST_SECONDS
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.1, remaining))
    print("[换向] 停止结束")


def _phase_c_directions(ctrl, stream) -> bool:
    print()
    print("=" * 108)
    print("[阶段C] 前、后、左、右依次运动")
    print("=" * 108)

    directions = [
        ("前进", AXIS_FORWARD, 0),
        ("后退", -AXIS_FORWARD, 0),
        ("左移", 0, -AXIS_LATERAL),
        ("右移", 0, AXIS_LATERAL),
    ]
    for label, forward, lateral in directions:
        _run_direction(ctrl, stream, label, forward, lateral)
        _wait_between_directions(stream, f"{label}完成")
    print("[阶段C] ✓ 四个方向动作完成")
    return True


# ===================== 收尾 =====================
def _shutdown(ctrl, stream, do_home=True) -> bool:
    """收尾：停轴流 → ensure_home（内部先趴下，再归零）"""
    print()
    print("=" * 108)
    print("[收尾] 停轴流 → " + ("趴下 + 归零" if do_home else "仅趴下"))
    print("=" * 108)

    if stream is not None:
        stream.set_axes(0, 0)
        print("[收尾] forward/lateral 归 0，等待 0.5s 让机器人减速")
        time.sleep(0.5)
        stream.stop()
        print(f"[收尾] 轴流已停止，共发 {stream.frames} 帧")

    if do_home:
        return l3.ensure_home(ctrl)
    else:
        return l3.ensure_sit(ctrl)


# ===================== 主流程 =====================
def main():
    ap = argparse.ArgumentParser(
        description="起立 → 切移动模式 + 低速步态 → 前后左右定时运动 → 趴下 → 归零")
    l3.add_common_args(ap)
    ap.add_argument("--hold-hz", type=float, default=30.0,
                    help="轴流频率，≥20Hz，默认 30")
    ap.add_argument("--no-home", action="store_true",
                    help="收尾只趴下，不归零")
    args = ap.parse_args()

    if args.hold_hz < 20.0:
        print(f"[错误] --hold-hz={args.hold_hz} < 20Hz，请设为 ≥20")
        return

    state_port = args.state_port if args.state_port and args.state_port > 0 else None
    ctrl = l3.Lite3Controller(args.ip, state_port=state_port)
    ctrl.start_heartbeat(hz=5)
    if state_port:
        ctrl.start_state_receiver()

    shutdown_str = "趴下 + 归零" if not args.no_home else "仅趴下（--no-home）"

    print(f"[任务9] {args.ip}:43893  心跳 5Hz")
    if state_port:
        print(f"[任务9] 监听 UDP :{state_port} 接收 0x0901 + 0x0902")
    print("[任务9] 流程：前进 3s → 停 2s → 后退 3s → 停 2s → "
          "左移 3s → 停 2s → 右移 3s → 停 2s → " + shutdown_str)
    print(f"[任务9] 轴值：前进=+{AXIS_FORWARD}，后退=-{AXIS_FORWARD}，"
          f"左移=-{AXIS_LATERAL}，右移=+{AXIS_LATERAL}；"
          f"前进死区={l3.AXIS_DEADZONE_FWD}，侧向死区={l3.AXIS_DEADZONE_LAT}")
    print(f"[任务9] 起立判据：(6,0,0) 力控站立+平地低速+静止，"
          f"四膝 {l3.STAND_MIN_DEG:.0f}°~{l3.STAND_MAX_DEG:.0f}°，"
          f"连续 {STABLE_SECONDS:.0f}s")
    print(f"[任务9] 每段动作后 forward/lateral 归 0，并保持轴流 30Hz 停止 2 秒")
    print(f"[任务9] 安全：确保前后左右均有足够空间，Ctrl-C 会安全停车")
    print(f"[任务9] 软急停：python3 -c \"import lite3_common as l3,time; "
          f"c=l3.Lite3Controller('{args.ip}'); c.start_heartbeat(5); "
          f"c.send_simple(l3.CMD_SOFT_ESTOP); time.sleep(1); c.close()\"")
    print("[任务9] 运行命令：python3 task09_axis_directions.py")
    print("=" * 108)

    stream = None
    home_ok = False
    try:
        time.sleep(0.5)
        _print_pose(ctrl, tag="当前")
        print()

        # 阶段 A
        if not _phase_a_stand(ctrl):
            return
        print()
        _print_pose(ctrl, tag="起立后")
        print()

        # 阶段 B
        stream = AxisStream(ctrl, hz=args.hold_hz)
        if not _phase_b_move_gait(ctrl, args.hold_hz, stream):
            print()
            print("[阶段B] 失败，仍进入收尾")
            home_ok = _shutdown(ctrl, stream, do_home=not args.no_home)
            stream = None
            return
        print()
        _print_pose(ctrl, tag="切步态后")
        print()

        # 阶段 C：前后左右运动
        time.sleep(0.5)
        directions_ok = _phase_c_directions(ctrl, stream)

        # 收尾
        home_ok = _shutdown(ctrl, stream, do_home=not args.no_home)
        stream = None

        # 汇总
        print()
        print("=" * 108)
        print("== task09 总结 ==")
        if directions_ok and home_ok:
            print("[结论] ✓ 全流程成功：起立 → 切模式 → 前后左右运动（每段 3s，"
                  "换向停 2s） → "
                  + ("趴下 + 归零" if not args.no_home else "趴下（未归零）"))
        else:
            print("[结论] ~ 四向运动完成，但收尾未完全确认")
            print("       请查看上方日志中的 [归零] 记录")

    except KeyboardInterrupt:
        print("\n[中断] 用户 Ctrl-C，进入收尾")
        try:
            home_ok = _shutdown(ctrl, stream, do_home=not args.no_home)
            stream = None
        except KeyboardInterrupt:
            print("[中断] 收尾再次被中断，机器人可能仍在中间态")
            print("[中断] 可手动发软急停：")
            print(f"       python3 -c \"import lite3_common as l3,time; "
                  f"c=l3.Lite3Controller('{args.ip}'); c.start_heartbeat(5); "
                  "c.send_simple(l3.CMD_SOFT_ESTOP); time.sleep(1); c.close()\"")
    finally:
        if stream is not None:
            stream.stop()
            print(f"[退出] 轴流已停止，共发 {stream.frames} 帧")
        ctrl.close()
        print("[退出] UDP 已关闭")


if __name__ == "__main__":
    main()