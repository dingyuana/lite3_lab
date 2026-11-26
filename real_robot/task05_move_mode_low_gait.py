#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task05_move_mode_low_gait.py —— 起立 → 切移动模式 + 低速步态 → 保持 → 趴下 → 归零
兼容 Python 3.8+，用 python3 运行。

【完整流程】
  阶段 A 起立：
    趴下（四膝 > 130°）→ 发 0x21010202 → 等 (6,0,0) 站立 + 四膝 [60°,90°]
    连续 3 秒。期间只发心跳，不发轴指令。
  阶段 B 切移动模式 + 低速步态：
    0x21010D06 移动模式 → 起 30Hz 三轴 0 值保活 → 0x21010300 低速步态
    验收 gait=0 且四膝 [60°,130°]。
  阶段 C 保持：
    只发轴流 + 心跳，保持 --hold 秒，观察 gait 与四膝是否稳定。
  收尾：
    调用 lite3_common.ensure_home()：
      1) 先 ensure_sit：发 0x21010202 趴下，等四膝 > 130°
      2) 再发 0x21010C05 归零，等 (17,0,0) 回零后回到趴下
      3) 打印归零前后四膝均值变化

【关键：切移动模式后姿态可能保持在 77.5°，也可能变 121.4°】
  两者都是有效的移动就绪态，验收只看 gait=0 且四膝在 [60°,130°]。
  77.5° 是 (6,0,0) 站立，121.4° 是 (6,0,1) 移动就绪半蹲，都接受。

【用法】
  python3 task05_move_mode_low_gait.py
  python3 task05_move_mode_low_gait.py --hold 20
  python3 task05_move_mode_low_gait.py --hold 0        # 一直保持，Ctrl-C 后收尾
  python3 task05_move_mode_low_gait.py --no-home       # 只趴下不归零
"""

import argparse
import math
import threading
import time
import lite3_common as l3

STABLE_SECONDS = 3.0
STAND_TIMEOUT  = 30.0


class AxisStream:
    """以 ≥20Hz 持续发送三轴轴指令，forward 值可动态改。"""
    def __init__(self, ctrl, hz=30.0):
        self.ctrl = ctrl
        self.hz = hz
        self.forward = 0
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
            self.ctrl.send_simple(l3.CMD_AXIS_FORWARD, self.forward)
            self.ctrl.send_simple(l3.CMD_AXIS_LATERAL, 0)
            self.ctrl.send_simple(l3.CMD_AXIS_YAW, 0)
            self.frames += 1
            time.sleep(period)

    def set_forward(self, value: int):
        self.forward = value

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
              f"前向速度={st.vel_body[0]:+.3f} m/s")
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
    """站立成功：(6,0,0) 力控站立+平地低速+静止，且四膝 60~90。"""
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


# ===================== 阶段 B：切移动模式 + 低速步态 =====================
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
    stream.set_forward(0)
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

    # 验收：gait=0 且四膝在 [60°, 130°]
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


# ===================== 阶段 C：保持 =====================
def _phase_c_hold(ctrl, stream, seconds):
    print()
    print("=" * 108)
    print(f"[阶段C] 保持移动就绪态 "
          f"{seconds if seconds > 0 else '无限（Ctrl-C 结束）'}")
    print("=" * 108)
    print(f"[阶段C] 轴流 forward=0 继续发，不移动位置")

    if seconds <= 0:
        while True:
            time.sleep(1.0)
            st = ctrl.get_state()
            m = ctrl.knee_mean_deg()
            if st is not None and m is not None:
                t, desc = l3.state_tuple(st)
                gait_ok = (st.robot_gait_state == l3.GAIT_LOW_SPEED)
                squat_ok = l3.STAND_MIN_DEG <= m <= l3.SQUAT_MAX_DEG
                flag = "✓" if (gait_ok and squat_ok) else "⚠"
                print(f"[保持] {flag} 状态={t} {desc}  "
                      f"四膝均值={m:+.1f}°  已发轴帧={stream.frames}")
    else:
        t0 = time.time()
        while time.time() - t0 < seconds:
            time.sleep(1.0)
            elapsed = time.time() - t0
            st = ctrl.get_state()
            m = ctrl.knee_mean_deg()
            if st is not None and m is not None:
                t, desc = l3.state_tuple(st)
                gait_ok = (st.robot_gait_state == l3.GAIT_LOW_SPEED)
                squat_ok = l3.STAND_MIN_DEG <= m <= l3.SQUAT_MAX_DEG
                flag = "✓" if (gait_ok and squat_ok) else "⚠"
                print(f"[保持] {elapsed:4.1f}/{seconds:.0f}s  {flag} "
                      f"状态={t} {desc}  四膝均值={m:+.1f}°  "
                      f"已发轴帧={stream.frames}")


# ===================== 收尾 =====================
def _shutdown(ctrl, stream, do_home=True) -> bool:
    """收尾：停轴流 → ensure_home（内部先趴下，再归零）"""
    print()
    print("=" * 108)
    print("[收尾] 停轴流 → " + ("趴下 + 归零" if do_home else "仅趴下"))
    print("=" * 108)

    if stream is not None:
        stream.set_forward(0)
        print("[收尾] forward 归 0，等待 0.5s 让机器人减速")
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
        description="起立 → 切移动模式 + 低速步态 → 保持 → 趴下 → 归零")
    l3.add_common_args(ap)
    ap.add_argument("--hold", type=float, default=10.0,
                    help="保持秒数，默认 10；0=一直保持，Ctrl-C 后收尾")
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

    hold_str = f"{args.hold:.0f}s" if args.hold > 0 else "无限（Ctrl-C 结束）"
    shutdown_str = "趴下 + 归零" if not args.no_home else "仅趴下（--no-home）"

    print(f"[任务5] {args.ip}:43893  心跳 5Hz")
    if state_port:
        print(f"[任务5] 监听 UDP :{state_port} 接收 0x0901 + 0x0902")
    print(f"[任务5] 流程：起立 → 切移动模式 + 低速步态 → 保持 {hold_str} → "
          f"{shutdown_str}")
    print(f"[任务5] 起立判据：(6,0,0) 力控站立+平地低速+静止，"
          f"四膝 {l3.STAND_MIN_DEG:.0f}°~{l3.STAND_MAX_DEG:.0f}°，"
          f"连续 {STABLE_SECONDS:.0f}s")
    print(f"[任务5] 切模式判据：gait=0 且 四膝 "
          f"{l3.STAND_MIN_DEG:.0f}°~{l3.SQUAT_MAX_DEG:.0f}°")
    print(f"[任务5] 注：本任务不前进，轴流 forward=0，只把机器人准备好")
    print(f"[任务5] 运行命令：python3 task05_move_mode_low_gait.py")
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
            # 即使 B 失败，也走收尾，避免机器人停在中间态
            print()
            print("[阶段B] 失败，仍进入收尾")
            home_ok = _shutdown(ctrl, stream, do_home=not args.no_home)
            stream = None
            return
        print()
        _print_pose(ctrl, tag="切步态后")

        # 阶段 C
        _phase_c_hold(ctrl, stream, args.hold)

        # 阶段 C 结论
        st_end = ctrl.get_state()
        m_end = ctrl.knee_mean_deg()
        print()
        print("=" * 108)
        print("== 阶段C 结论 ==")
        if st_end is not None and m_end is not None:
            t, desc = l3.state_tuple(st_end)
            print(f"保持结束状态 = {t} {desc}")
            print(f"保持结束四膝 = {m_end:+.1f}°")
            gait_ok = (st_end.robot_gait_state == l3.GAIT_LOW_SPEED)
            squat_ok = l3.STAND_MIN_DEG <= m_end <= l3.SQUAT_MAX_DEG
            if gait_ok and squat_ok:
                print(f"[结论] ✓ 移动就绪态稳定：gait=0，四膝在 "
                      f"[{l3.STAND_MIN_DEG:.0f}°, {l3.SQUAT_MAX_DEG:.0f}°]")
            else:
                print(f"[结论] ⚠ 保持结束时不在就绪判据内")
        else:
            print("[结论] 数据不足")

        # 收尾
        home_ok = _shutdown(ctrl, stream, do_home=not args.no_home)
        stream = None

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
        print()
        print("=" * 108)
        print("== task05 总结 ==")
        if home_ok:
            if args.no_home:
                print("[结论] ✓ 起立 → 切模式 → 步态 → 趴下 完成"
                      "（--no-home 未归零）")
            else:
                print("[结论] ✓ 起立 → 切模式 → 步态 → 趴下 → 归零 完成")
                print("       机器人停在趴下，关节已归零")
        else:
            print("[结论] ~ 流程完成，但收尾未完全确认")
            print("       请查看上方日志中的 [归零] 记录")
        print()
        print("[提示] 如需前进：python3 task06_axis_tap.py 或 "
              "task07_axis_stream_forward.py")
        print("[提示] 如需重新起立：python3 task03_stand_to_force.py")

        if stream is not None:
            stream.stop()
        ctrl.close()
        print("[退出] UDP 已关闭")


if __name__ == "__main__":
    main()