#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task07_axis_stream_forward.py —— 起立 → 切移动模式 + 低速步态 → 持续前进 → 停稳 → 趴下 → 归零
兼容 Python 3.8+，用 python3 运行。

【完整流程】
  启动检查：
    (1,0,0) 趴下或 basic_state=98 时继续；其它状态提示并先归零，
    只有确认回到 (1,0,0) 后才继续，否则停止任务。
  阶段 A 起立：
    趴下（四膝 > 130°）→ 发 0x21010202 → 等 (6,0,0) 站立 + 四膝 [60°,90°]
    连续 3 秒。期间只发心跳，不发轴指令。
  阶段 B 切模式 + 步态：
    0x21010D06 移动模式 → 起 30Hz 轴流 → 0x21010300 低速步态
    验收 gait=0 且四膝 [60°,130°]。
  阶段 C 持续前进：
    forward 0 → 9000，持续 --duration 秒（默认 3，0=一直走到 Ctrl-C）
    每 0.5 秒打印 vel_body[0]、pos_world[0]，实时看到速度与位移。
  阶段 D 停车：
    forward 归 0 → 保持轴流继续发 0 共 --cooldown 秒（默认 3.0）
    确认 vel_body[0] 降到 < 0.02 再退出。   
  收尾：
    调用 lite3_common.ensure_home()：
      1) 先 ensure_sit：发 0x21010202 趴下，等四膝 > 130°
      2) 再发 0x21010C05 归零，等 (17,0,0) 回零后回到趴下
      3) 打印归零前后四膝均值变化
  Ctrl-C 中断时同样走“归 0 + 观察减速 → 趴下 → 归零”流程。

【为什么停车要保活】
  前进时轴流持续发 forward=9000，发 0 后机器人靠惯性减速。
  如果立刻停轴流，250ms 轴指令超时会让底层接管，可能突然失去支撑。
  保持轴流发 0 直到 vel 降到 < 0.02，是最平稳的停车方式。

【安全】
  - 默认前进 3 秒，低速步态下约 0.2~0.3 m
  - 前后留 2 m 空间
  - Ctrl-C 会安全停车并进入收尾流程
  - 手边准备软急停：send_simple(0x21020C0E)

【用法】
  python3 task07_axis_stream_forward.py
  python3 task07_axis_stream_forward.py --duration 5
  python3 task07_axis_stream_forward.py --duration 0            # 一直走到 Ctrl-C
  python3 task07_axis_stream_forward.py --value 12000 --duration 5
  python3 task07_axis_stream_forward.py --no-home               # 只趴下不归零
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


def _startup_state_gate(ctrl, timeout=3.0) -> bool:
    """Allow sit/98 at startup; home other states before proceeding."""
    print("[启动检查] 等待状态帧和关节帧...")
    deadline = time.monotonic() + timeout
    st = ctrl.get_state()
    knee_mean = ctrl.knee_mean_deg()
    while time.monotonic() < deadline:
        if st is not None and knee_mean is not None:
            break
        time.sleep(0.05)
        st = ctrl.get_state()
        knee_mean = ctrl.knee_mean_deg()

    if st is None or knee_mean is None:
        print("[启动检查] ✗ 状态/关节数据未就绪，停止任务")
        return False

    state, desc = l3.state_tuple(st)
    if state == (1, 0, 0):
        print("[启动检查] ✓ 当前为趴下状态 (1,0,0)，继续执行")
        return True
    if l3.is_initial_standby(st):
        print("[启动检查] ✓ 当前 basic_state=98，按要求继续执行")
        return True

    print(f"[启动检查] ⚠ 当前状态={l3.format_state_code(st)} {desc}，"
          "先执行归零")
    if not l3.ensure_home(ctrl):
        print("[启动检查] ✗ 归零未确认，停止任务，不启动运动")
        return False

    st = ctrl.get_state()
    state = l3.state_tuple(st)[0] if st is not None else None
    if state != (1, 0, 0):
        print(f"[启动检查] ✗ 归零后状态={l3.format_state_code(st)}，"
              "不是 (1,0,0)，停止任务")
        return False

    print("[启动检查] ✓ 已确认归零并回到 (1,0,0)，继续执行")
    return True


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


# ===================== 阶段 C：持续前进 =====================
def _phase_c_forward(ctrl, stream, value, duration, cooldown=5.0) -> bool:
    """
    持续前进：forward 0 → value，持续 duration 秒（0=无限）。
    每 0.5 秒打印实时速度和位置。
    返回是否方向正确（vel 峰值 > VEL_FWD_EPS）。
    """
    print()
    print("=" * 108)
    print("[阶段C] 持续前进")
    print("=" * 108)

    st0 = ctrl.get_state()
    pos_start = st0.pos_world[0] if st0 is not None else 0.0
    if st0 is not None:
        print(f"[基线] 前进前：vel_body[0]={st0.vel_body[0]:+.3f} m/s  "
              f"pos_world[0]={pos_start:+.4f} m")

    duration_str = f"{duration:.1f}s" if duration > 0 else "无限（Ctrl-C 停）"
    print(f"[前进] forward 0 → {value}，持续 {duration_str}")
    print(f"[前进] 每 0.5 秒打印实时状态，Ctrl-C 安全停车")
    print("-" * 108)

    stream.set_forward(value)
    t0 = time.time()
    last_print = 0.0
    vel_samples = []
    last_st = None

    try:
        while True:
            elapsed = time.time() - t0
            if duration > 0 and elapsed >= duration:
                break
            time.sleep(0.1)
            st = ctrl.get_state()
            if st is None:
                continue
            last_st = st
            vel_samples.append(st.vel_body[0])

            if elapsed - last_print >= 0.5:
                t, desc = l3.state_tuple(st)
                pos_now = st.pos_world[0]
                pos_delta = pos_now - pos_start
                print(f"[前进] {elapsed:5.1f}s  vel={st.vel_body[0]:+.3f} m/s  "
                      f"pos={pos_now:+.3f} m（Δ {pos_delta:+.3f} m）  "
                      f"状态={t} {desc}")
                last_print = elapsed
    except KeyboardInterrupt:
        print("\n[前进] Ctrl-C 中断，进入安全停车流程")
        return _safe_stop(ctrl, stream, vel_samples, pos_start,
                          f"中断时已走 {time.time() - t0:.1f}s",
                          cooldown=cooldown)

    return _safe_stop(ctrl, stream, vel_samples, pos_start,
                      f"前进 {duration:.1f}s 完成", cooldown=cooldown)


# ===================== 阶段 D：停车 =====================
def _safe_stop(ctrl, stream, vel_samples, pos_start, reason,
               cooldown=5.0) -> bool:
    """
    安全停车：forward 归 0，保持轴流发 0 共 cooldown 秒（从 stream 外部给），
    期间每 0.5 秒打印，直到 vel < VEL_STOP_EPS 或超时。
    """
    print()
    print("=" * 108)
    print(f"[停车] {reason}")
    print("=" * 108)
    print(f"[停车] forward 归 0，保持轴流发 0 观察减速")
    stream.set_forward(0)

    t0 = time.time()
    last_print = 0.0
    stop_t0 = None
    stop_vel = None

    while time.time() - t0 < cooldown:
        time.sleep(0.1)
        st = ctrl.get_state()
        if st is None:
            continue
        elapsed = time.time() - t0
        vel_now = st.vel_body[0]

        if elapsed - last_print >= 0.5:
            print(f"[停车] {elapsed:4.1f}s  vel={vel_now:+.3f} m/s")
            last_print = elapsed

        if abs(vel_now) < VEL_STOP_EPS and stop_t0 is None:
            stop_t0 = elapsed
            stop_vel = vel_now
            break

    # 结果统计
    st_end = ctrl.get_state()
    print()
    print("=" * 108)
    print("== 前进 + 停车结果 ==")

    vel_peak = max(vel_samples) if vel_samples else 0.0
    vel_avg = (sum(vel_samples) / len(vel_samples)) if vel_samples else 0.0
    pos_end = st_end.pos_world[0] if st_end is not None else 0.0
    pos_delta = (pos_end - pos_start) if pos_start is not None else 0.0

    print(f"前进期 vel 峰值 : {vel_peak:+.4f} m/s")
    print(f"前进期 vel 均值 : {vel_avg:+.4f} m/s")
    if pos_start is not None:
        print(f"pos_world[0]   : {pos_start:+.4f} → {pos_end:+.4f} m "
              f"（Δ {pos_delta:+.4f} m，IMU 积分，仅参考）")
    if stop_t0 is not None:
        print(f"停稳时刻       : 发 0 后 {stop_t0:.1f}s，vel={stop_vel:+.4f} m/s")
    else:
        if st_end is not None:
            print(f"停稳状态       : {cooldown:.1f}s 内未降到 "
                  f"{VEL_STOP_EPS}，当前 vel={st_end.vel_body[0]:+.4f} m/s")

    vel_ok = vel_peak > VEL_FWD_EPS
    stopped_ok = stop_t0 is not None

    print()
    if vel_ok:
        print(f"[方向] ✓ 正确：前进期 vel 峰值 {vel_peak:+.4f} m/s > "
              f"{VEL_FWD_EPS}，方向为向前")
    else:
        print(f"[方向] ✗ 未前进：vel 峰值 {vel_peak:+.4f} m/s 低于 "
              f"{VEL_FWD_EPS}")
        print(f"       检查是否在移动模式、轴值是否 > 6553、是否被阻挡")

    if stopped_ok:
        print(f"[停稳] ✓ 已停稳：{stop_t0:.1f}s 后 vel < {VEL_STOP_EPS}")
    else:
        print(f"[停稳] ⚠ {cooldown:.1f}s 内未完全停稳")

    if vel_ok and stopped_ok:
        print()
        print(f"[结论] ✓ 成功：方向正确 + 已停稳")
        return True
    elif vel_ok:
        print()
        print(f"[结论] ✗ 前进方向正确，但未在观察窗口内停稳")
        return False
    else:
        print()
        print(f"[结论] ✗ 失败：未检测到前进")
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
        description="起立 → 切移动模式 + 低速步态 → 持续前进 → 停稳 → 趴下 → 归零")
    l3.add_common_args(ap)
    ap.add_argument("--value", type=int, default=9000,
                    help="前进轴值，需 >6553，默认 9000")
    ap.add_argument("--duration", type=float, default=3.0,
                    help="持续前进秒数，默认 3；0=一直走到 Ctrl-C")
    ap.add_argument("--cooldown", type=float, default=3.0,
                    help="停车后观察减速秒数，默认 3")
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
    if args.cooldown <= 0.0:
        print("[错误] --cooldown 必须大于 0")
        return
    if args.duration > 20.0:
        print(f"[警告] --duration={args.duration}s 偏长，请确保前方有足够空间")

    state_port = args.state_port if args.state_port and args.state_port > 0 else None
    if state_port is None:
        ap.error("task07 启动状态检查需要状态监听，不能设置 --state-port 0")

    ctrl = l3.Lite3Controller(args.ip, state_port=state_port)
    ctrl.start_heartbeat(hz=5)
    if state_port:
        ctrl.start_state_receiver()

    duration_str = f"{args.duration:.1f}s" if args.duration > 0 else "无限（Ctrl-C 停）"
    shutdown_str = "趴下 + 归零" if not args.no_home else "仅趴下（--no-home）"

    print(f"[任务7] {args.ip}:43893  心跳 5Hz")
    if state_port:
        print(f"[任务7] 监听 UDP :{state_port} 接收 0x0901 + 0x0902")
    print(f"[任务7] 流程：起立 → 切移动模式 + 低速步态 → 持续前进"
          f"（0x21010130={args.value}，持续 {duration_str}） → 停稳 → {shutdown_str}")
    print(f"[任务7] 起立判据：(6,0,0) 力控站立+平地低速+静止，"
          f"四膝 {l3.STAND_MIN_DEG:.0f}°~{l3.STAND_MAX_DEG:.0f}°，"
          f"连续 {STABLE_SECONDS:.0f}s")
    print(f"[任务7] 停车：forward 归 0，保持轴流 {args.cooldown:.0f}s 观察减速")
    print(f"[任务7] 安全：前后留 2m 空间，Ctrl-C 会安全停车")
    print(f"[任务7] 软急停：python3 -c \"import lite3_common as l3,time; "
          f"c=l3.Lite3Controller('{args.ip}'); c.start_heartbeat(5); "
          f"c.send_simple(l3.CMD_SOFT_ESTOP); time.sleep(1); c.close()\"")
    print(f"[任务7] 运行命令：python3 task07_axis_stream_forward.py")
    print("=" * 108)

    stream = None
    home_ok = False
    try:
        time.sleep(0.5)
        _print_pose(ctrl, tag="当前")
        print()

        if not _startup_state_gate(ctrl):
            return

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

        # 阶段 C：持续前进
        time.sleep(0.5)
        fwd_ok = _phase_c_forward(
            ctrl, stream, args.value, args.duration, cooldown=args.cooldown)

        # 收尾
        home_ok = _shutdown(ctrl, stream, do_home=not args.no_home)
        stream = None

        # 汇总
        print()
        print("=" * 108)
        print("== task07 总结 ==")
        if fwd_ok and home_ok:
            print("[结论] ✓ 全流程成功：起立 → 切模式 → 步态 → 持续前进 → 停稳 → "
                  + ("趴下 + 归零" if not args.no_home else "趴下（未归零）"))
        elif not fwd_ok:
            print("[结论] ✗ 前进失败，请根据上方分析排查")
        else:
            print("[结论] ~ 前进成功，但收尾未完全确认")
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