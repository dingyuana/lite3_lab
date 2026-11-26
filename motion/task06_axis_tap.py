#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task06_axis_tap.py —— 起立 → 切移动模式 + 低速步态 → 点动试探 → 趴下 → 归零
兼容 Python 3.8+，用 python3 运行。

【完整流程】
  阶段 A 起立：
    趴下（四膝 > 130°）→ 发 0x21010202 → 等 (6,0,0) 站立 + 四膝 [60°,90°]
    连续 3 秒。期间只发心跳，不发轴指令。
  阶段 B 切模式 + 步态：
    0x21010D06 移动模式 → 起 30Hz 轴流 → 0x21010300 低速步态
    验收 gait=0 且四膝 [60°,130°]。
  阶段 C 点动试探：
    forward 0→7000 持续 --duration（默认 0.8s）→ 归 0 → 观察 --cooldown（默认 2.0s）
    判据分开：
      方向正确 = 点动期 vel_body[0] 峰值 > 0.02 m/s
      停稳     = 停车后 vel_body[0] 峰值 < 0.02 m/s
  收尾：
    调用 lite3_common.ensure_home()：
      1) 先 ensure_sit：发 0x21010202 趴下，等四膝 > 130°
      2) 再发 0x21010C05 归零，等 (17,0,0) 回零后回到趴下
      3) 打印归零前后四膝均值变化

【为什么停车要保活】
  点动时轴流持续发 forward=7000，发 0 后机器人靠惯性减速。
  如果立刻停轴流，250ms 轴指令超时会让底层接管，可能突然失去支撑。
  保持轴流发 0 直到 vel 降到 < 0.02，是最平稳的停车方式。

【安全】
  - 点动时长默认 0.8s，前进约几厘米到十几厘米
  - 前后留 1m 空间
  - Ctrl-C 会安全停车并进入收尾流程

【用法】
  python3 task06_axis_tap.py
  python3 task06_axis_tap.py --value 8000 --duration 1.0
  python3 task06_axis_tap.py --value 7000 --duration 0.6
  python3 task06_axis_tap.py --no-home       # 只趴下不归零
"""

import argparse
import math
import threading
import time
import lite3_common as l3

STABLE_SECONDS = 3.0
STAND_TIMEOUT  = 30.0

VEL_FWD_EPS    = 0.02    # m/s，方向判据
VEL_STOP_EPS   = 0.02    # m/s，停稳判据


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
        print("       1) 是否有其他客户端在发指令（会在 34.6° 时把机器人拉走）")
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


# ===================== 阶段 C：点动试探 =====================
def _sample_motion(ctrl, duration):
    vel_samples = []
    pos_a = None
    pos_b = None
    t0 = time.time()
    while time.time() - t0 < duration:
        st = ctrl.get_state()
        if st is not None:
            vel_samples.append(st.vel_body[0])
            if pos_a is None:
                pos_a = st.pos_world[0]
            pos_b = st.pos_world[0]
        time.sleep(0.05)
    if not vel_samples:
        return None, None, None, None
    return (max(vel_samples), sum(vel_samples) / len(vel_samples), pos_a, pos_b)


def _phase_c_tap(ctrl, stream, value, duration, cooldown) -> bool:
    print()
    print("=" * 108)
    print("[阶段C] 单次轴指令试探前进方向")
    print("=" * 108)

    st0 = ctrl.get_state()
    if st0 is not None:
        print(f"[基线] 点动前：vel_body[0]={st0.vel_body[0]:+.3f} m/s  "
              f"pos_world[0]={st0.pos_world[0]:+.4f} m")

    print(f"[点动] forward 0 → {value}，持续 {duration}s")
    stream.set_forward(value)
    vel_peak, vel_avg, pos_a, pos_b = _sample_motion(ctrl, duration)

    stream.set_forward(0)
    print(f"[点动] forward 归 0，观察 {cooldown}s 是否停稳")
    vel_after_peak, _, _, _ = _sample_motion(ctrl, cooldown)

    print()
    print("=" * 108)
    print("== 点动结果分析 ==")
    print(f"点动时长       : {duration:.2f}s")
    print(f"前进轴值       : {value}")
    if vel_peak is None:
        print("[失败] 点动期间未收到状态帧")
        return False

    print(f"点动期 vel 峰值 : {vel_peak:+.4f} m/s")
    print(f"点动期 vel 均值 : {vel_avg:+.4f} m/s")
    if pos_a is not None and pos_b is not None:
        print(f"pos_world[0]   : {pos_a:+.4f} → {pos_b:+.4f} m "
              f"（变化 {pos_b - pos_a:+.4f} m，IMU 积分，仅参考）")
    if vel_after_peak is not None:
        print(f"停车后 vel 峰值 : {vel_after_peak:+.4f} m/s")

    vel_ok = vel_peak > VEL_FWD_EPS
    vel_negative = vel_peak < -VEL_FWD_EPS
    stopped_ok = (vel_after_peak is None or vel_after_peak < VEL_STOP_EPS)

    print()
    if vel_negative:
        print(f"[方向] ✗ 方向反了：前向速度 {vel_peak:+.4f} m/s 为负")
        print(f"       检查是否有其他客户端在发负值")
        return False

    if vel_ok:
        print(f"[方向] ✓ 正确：点动期前向速度 {vel_peak:+.4f} m/s > "
              f"{VEL_FWD_EPS}，方向为向前")
        if vel_peak < 0.05:
            print(f"       速度偏小（{vel_peak:+.4f} m/s），可增大 --duration "
                  f"到 1.0s 或 --value 到 9000")
    else:
        print(f"[方向] ✗ 未动：点动期前向速度峰值 {vel_peak:+.4f} m/s，"
              f"低于 {VEL_FWD_EPS}")
        print(f"       可能原因：")
        print(f"       1) 轴指令未被采纳（检查是否在移动模式）")
        print(f"       2) 轴值在死区内（{value} 应 > {l3.AXIS_DEADZONE_FWD}）")
        print(f"       3) 机器人被外力阻挡")
        print(f"       4) 机器人实际趴着（阶段 A 起立失败）")

    if stopped_ok:
        print(f"[停稳] ✓ 停车后 vel 峰值 "
              f"{vel_after_peak if vel_after_peak is not None else 0:+.4f} m/s，"
              f"已停稳")
    else:
        print(f"[停稳] ⚠ 停车后仍有速度 {vel_after_peak:+.4f} m/s > "
              f"{VEL_STOP_EPS}，可能有惯性或指令残留")
        print(f"       建议增大 --cooldown 到 3.0s，或检查是否有其他客户端在发指令")

    if vel_ok and stopped_ok:
        print()
        print(f"[结论] ✓ 成功：方向正确 + 已停稳")
        return True
    elif vel_ok and not stopped_ok:
        print()
        print(f"[结论] ~ 方向正确，但停车后未完全停稳")
        return True
    else:
        print()
        print(f"[结论] ✗ 失败：方向未动，请根据上方分析排查")
        return False


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
        description="起立 → 切移动模式 + 低速步态 → 点动试探 → 趴下 → 归零")
    l3.add_common_args(ap)
    ap.add_argument("--value", type=int, default=7000,
                    help="点动前进轴值，需 >6553，默认 7000")
    ap.add_argument("--duration", type=float, default=0.8,
                    help="点动持续秒数，默认 0.8")
    ap.add_argument("--cooldown", type=float, default=2.0,
                    help="停车后观察秒数，默认 2.0")
    ap.add_argument("--hold-hz", type=float, default=30.0,
                    help="轴流频率，≥20Hz，默认 30")
    ap.add_argument("--no-home", action="store_true",
                    help="收尾只趴下，不归零")
    args = ap.parse_args()

    if args.value <= l3.AXIS_DEADZONE_FWD:
        print(f"[错误] --value={args.value} 落在死区内，请 > {l3.AXIS_DEADZONE_FWD}")
        return
    if args.hold_hz < 20.0:
        print(f"[错误] --hold-hz={args.hold_hz} < 20Hz，请设为 ≥20")
        return
    if args.duration > 2.0:
        print(f"[警告] --duration={args.duration}s 偏长，前进距离可能较大")

    state_port = args.state_port if args.state_port and args.state_port > 0 else None
    ctrl = l3.Lite3Controller(args.ip, state_port=state_port)
    ctrl.start_heartbeat(hz=5)
    if state_port:
        ctrl.start_state_receiver()

    shutdown_str = "趴下 + 归零" if not args.no_home else "仅趴下（--no-home）"

    print(f"[任务6] {args.ip}:43893  心跳 5Hz")
    if state_port:
        print(f"[任务6] 监听 UDP :{state_port} 接收 0x0901 + 0x0902")
    print(f"[任务6] 流程：起立 → 切移动模式 + 低速步态 → 点动"
          f"（0x21010130={args.value}，持续 {args.duration}s） → {shutdown_str}")
    print(f"[任务6] 起立判据：(6,0,0) 力控站立+平地低速+静止，"
          f"四膝 {l3.STAND_MIN_DEG:.0f}°~{l3.STAND_MAX_DEG:.0f}°，"
          f"连续 {STABLE_SECONDS:.0f}s")
    print(f"[任务6] 点动判据：方向 = 点动期 vel 峰值 > {VEL_FWD_EPS}；"
          f"停稳 = 停车后 vel 峰值 < {VEL_STOP_EPS}")
    print(f"[任务6] 安全：前后留 1m 空间，点动后立即发 0 停车")
    print(f"[任务6] 运行命令：python3 task06_axis_tap.py")
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

        # 阶段 C
        time.sleep(0.5)
        tap_ok = _phase_c_tap(ctrl, stream, args.value,
                              args.duration, args.cooldown)

        # 收尾
        home_ok = _shutdown(ctrl, stream, do_home=not args.no_home)
        stream = None

        # 汇总
        print()
        print("=" * 108)
        print("== task06 总结 ==")
        if tap_ok and home_ok:
            print("[结论] ✓ 全流程成功：起立 → 切模式 → 步态 → 点动前进方向正确 → "
                  + ("趴下 + 归零" if not args.no_home else "趴下（未归零）"))
            print("       下一步 task07 持续前进：")
            print("       python3 task07_axis_stream_forward.py")
        elif not tap_ok:
            print("[结论] ✗ 点动失败，请根据上方分析排查")
            print("       如需重新尝试：python3 task06_axis_tap.py")
        else:
            print("[结论] ~ 点动成功，但收尾未完全确认")
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