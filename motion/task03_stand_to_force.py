#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task03_stand_to_force.py —— 起立到稳定站立 → 保持 → 趴下 → 归零
兼容 Python 3.8+，用 python3 运行。

【完整流程】
  启动检查：
    basic_state=98 表示初始待机，不据此判断需要或已完成回零；
    首次起立时要求电量高于 20%。
  阶段 A 起立：
    趴下（四膝 > 130°）→ 发 0x21010202 → 等 (6,0,0) 站立 + 四膝 [60°,90°]
    连续 3 秒。期间只发心跳，不发轴指令。
  阶段 B 保持：
    只发心跳，保持 --hold 秒（默认 10），每秒打印状态三元组 + 四膝。
  阶段 C 收尾：
    调用 lite3_common.ensure_home()：
      1) 先 ensure_sit：发 0x21010202 趴下，等四膝 > 130°
      2) 再发 0x21010C05 归零，等 (17,0,0) 回零后回到趴下
      3) 打印归零前后四膝均值变化

【为什么归零前要先趴下】
  文档状态机：趴下(1) → 回零(17) → 回零完成后 → 趴下(1)。
  没有“站立 → 回零”的路径，所以 ensure_home 内部固定先趴下再归零。

【判据】
  起立成功：l3.is_stand_still(st) 为真（(6,0,0) 力控站立+平地低速+静止）
            且 四膝 60°~90°，连续 3 秒。
  趴下成功：四膝 > 130° 或 (1,0,0)
  归零完成：观察到 (17,0,0) 回零后回到趴下，或超时兜底

【状态日志】
  初始、起立、保持、趴下和归零阶段均打印状态三元组；basic_state=98 用红色显示。

【用法】
  python3 task03_stand_to_force.py
  python3 task03_stand_to_force.py --hold 20
  python3 task03_stand_to_force.py --hold 0            # 一直保持，Ctrl-C 后收尾
  python3 task03_stand_to_force.py --no-home           # 保持后只趴下，不归零
  python3 task03_stand_to_force.py --force             # 跳过起立前检查
"""

import argparse
import math
import time
import lite3_common as l3

STABLE_SECONDS = 3.0
STAND_TIMEOUT  = 30.0


def _print_pose(ctrl, tag="姿态"):
    st = ctrl.get_state()
    if st is not None:
        _, desc = l3.state_tuple(st)
        print(f"[{tag}] 状态={l3.format_state_code(st)} {desc}")
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


def _wait_stand(ctrl, timeout):
    """
    静默等待起立成功：连续 3 秒满足 _is_stand_success。
    每 3 秒打一个点，不逐秒打印。
    返回 (是否成功, 最后一次四膝均值)。
    """
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
            _, desc = l3.state_tuple(st)
            print(f"[起立] ... 等待中（{elapsed:.0f}s，"
                  f"状态={l3.format_state_code(st)} {desc}，"
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


def _phase_a_stand(ctrl, force=False) -> bool:
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
        _, desc = l3.state_tuple(st)
        print(f"[阶段A] 当前状态={l3.format_state_code(st)} {desc}，"
              f"四膝={mean_deg:+.1f}°，按趴下姿态发 0x21010202 起立")
    elif not force:
        _, desc = l3.state_tuple(st)
        print(f"[阶段A] 当前状态={l3.format_state_code(st)} {desc}，"
              f"四膝={mean_deg:+.1f}°")
        print("[阶段A] 不在趴下也不在稳定站立，加 --force 强制发 0x21010202")
        return False
    else:
        print(f"[阶段A] --force：当前四膝={mean_deg:+.1f}°，强制发起立")

    if l3.is_initial_standby(st) and st.battery_level <= 20.0:
        print("[阶段A] ✗ 初始待机(98)下电量不高于20%，禁止首次起立")
        return False

    print("[阶段A] 起立期间只发心跳，不发轴指令"
          "（否则会切到 121.4° 移动就绪态）")
    print("[阶段A] 发送 0x21010202 ...")
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


def _phase_b_hold(ctrl, seconds):
    print()
    print("=" * 108)
    print(f"[阶段B] 保持稳定站立 "
          f"{seconds if seconds > 0 else '无限（Ctrl-C 结束）'}")
    print("=" * 108)
    print("[阶段B] 只发心跳，不发轴指令")

    if seconds <= 0:
        while True:
            time.sleep(1.0)
            st = ctrl.get_state()
            m = ctrl.knee_mean_deg()
            if st is not None and m is not None:
                _, desc = l3.state_tuple(st)
                ok = _is_stand_success(st, m)
                flag = "✓" if ok else "⚠"
                print(f"[保持] {flag} 状态={l3.format_state_code(st)} "
                      f"{desc}  四膝均值={m:+.1f}°")
    else:
        t0 = time.time()
        while time.time() - t0 < seconds:
            time.sleep(1.0)
            elapsed = time.time() - t0
            st = ctrl.get_state()
            m = ctrl.knee_mean_deg()
            if st is not None and m is not None:
                _, desc = l3.state_tuple(st)
                ok = _is_stand_success(st, m)
                flag = "✓" if ok else "⚠"
                print(f"[保持] {elapsed:4.1f}/{seconds:.0f}s  {flag} "
                      f"状态={l3.format_state_code(st)} {desc}  "
                      f"四膝均值={m:+.1f}°")


def _phase_c_shutdown(ctrl, do_home=True) -> bool:
    print()
    print("=" * 108)
    print("[阶段C] 收尾：" + ("趴下 + 归零" if do_home else "仅趴下（--no-home）"))
    print("=" * 108)
    _print_pose(ctrl, tag="收尾前")

    if do_home:
        return l3.ensure_home(ctrl)
    else:
        return l3.ensure_sit(ctrl)


def main():
    ap = argparse.ArgumentParser(
        description="起立到稳定站立 → 保持 → 趴下 → 归零")
    l3.add_common_args(ap)
    ap.add_argument("--force", action="store_true",
                    help="跳过起立前检查，强制发 0x21010202")
    ap.add_argument("--hold", type=float, default=10.0,
                    help="站立保持秒数，默认 10；0=一直保持，Ctrl-C 后收尾")
    ap.add_argument("--no-home", action="store_true",
                    help="保持后只趴下，不做收尾归零（启动前仍会先回零）")
    args = ap.parse_args()

    state_port = args.state_port if args.state_port and args.state_port > 0 else None
    if state_port is None:
        ap.error("task03 启动前回零需要状态监听，不能设置 --state-port 0")

    ctrl = l3.Lite3Controller(args.ip, state_port=state_port)
    ctrl.start_heartbeat(hz=5)
    if state_port:
        ctrl.start_state_receiver()

    hold_str = f"{args.hold:.0f}s" if args.hold > 0 else "无限（Ctrl-C 结束）"
    shutdown_str = "趴下 + 归零" if not args.no_home else "仅趴下（--no-home）"

    print(f"[任务3] {args.ip}:43893  心跳 5Hz")
    if state_port:
        print(f"[任务3] 监听 UDP :{state_port} 接收 0x0901 + 0x0902")
    print(f"[任务3] 流程：起立 → 保持 {hold_str} → {shutdown_str}")
    print(f"[任务3] 起立判据：(6,0,0) 力控站立+平地低速+静止，"
          f"且四膝 {l3.STAND_MIN_DEG:.0f}°~{l3.STAND_MAX_DEG:.0f}°，"
          f"连续 {STABLE_SECONDS:.0f}s")
    print(f"[任务3] 注：归零必须在趴下状态执行，收尾固定先趴下再归零")
    print(f"[任务3] 运行命令：python3 task03_stand_to_force.py")
    print("=" * 108)

    home_ok = False
    try:
        time.sleep(0.5)
        _print_pose(ctrl, tag="当前")
        print()

        print("=" * 108)
        print("[启动检查] 初始状态")
        print("=" * 108)
        print("[启动检查] 等待状态帧和关节帧...")
        telemetry_deadline = time.monotonic() + 3.0
        initial_state = ctrl.get_state()
        knee_mean = ctrl.knee_mean_deg()
        while time.monotonic() < telemetry_deadline:
            if initial_state is not None and knee_mean is not None:
                break
            time.sleep(0.05)
            initial_state = ctrl.get_state()
            knee_mean = ctrl.knee_mean_deg()

        if initial_state is None or knee_mean is None:
            print("[启动检查] ✗ 状态/关节数据未就绪，停止任务")
            return

        if l3.is_initial_standby(initial_state):
            print("[启动检查] 状态码98：初始待机，不表示故障或回零已完成")
        else:
            print(f"[启动检查] 状态码={l3.format_state_code(initial_state)}")
        _print_pose(ctrl, tag="启动检查后")
        print()

        if not _phase_a_stand(ctrl, force=args.force):
            return

        print()
        _print_pose(ctrl, tag="起立后")

        _phase_b_hold(ctrl, args.hold)

        # 阶段 B 结束检查
        st_end = ctrl.get_state()
        m_end = ctrl.knee_mean_deg()
        print()
        print("=" * 108)
        print("== 阶段B 结论 ==")
        if st_end is not None and m_end is not None:
            _, desc = l3.state_tuple(st_end)
            print(f"保持结束状态 = {l3.format_state_code(st_end)} {desc}")
            print(f"保持结束四膝 = {m_end:+.1f}°")
            if _is_stand_success(st_end, m_end):
                print(f"[结论] ✓ 站立稳定，状态=(6,0,0)，"
                      f"四膝 {l3.STAND_MIN_DEG:.0f}°~{l3.STAND_MAX_DEG:.0f}°")
            else:
                print(f"[结论] ⚠ 保持结束时不在稳定站立判据内，可能被干扰")
        else:
            print("[结论] 数据不足")

        # 阶段 C 收尾
        home_ok = _phase_c_shutdown(ctrl, do_home=not args.no_home)
        _print_pose(ctrl, tag="收尾后")

    except KeyboardInterrupt:
        print("\n[中断] 用户 Ctrl-C，进入收尾")
        try:
            home_ok = _phase_c_shutdown(ctrl, do_home=not args.no_home)
            _print_pose(ctrl, tag="收尾后")
        except KeyboardInterrupt:
            print("[中断] 收尾再次被中断，机器人可能仍在站立或中间态")
            print("[中断] 可手动发软急停：")
            print(f"       python3 -c \"import lite3_common as l3,time; "
                  f"c=l3.Lite3Controller('{args.ip}'); c.start_heartbeat(5); "
                  "c.send_simple(l3.CMD_SOFT_ESTOP); time.sleep(1); c.close()\"")
    finally:
        print()
        print("=" * 108)
        print("== task03 总结 ==")
        if home_ok:
            if args.no_home:
                print("[结论] ✓ 起立 → 保持 → 趴下 完成"
                      "（--no-home 跳过收尾归零）")
            else:
                print("[结论] ✓ 起立 → 保持 → 趴下 → 归零 完成")
                print("       机器人停在趴下，关节已归零")
        else:
            print("[结论] ~ 起立/收尾流程未全部完成")
            print("       请查看上方阶段日志")
        print()
        print("[提示] 如需重新起立：python3 task03_stand_to_force.py")
        print("[提示] 如需前进：python3 task06_axis_tap.py 或 task07_axis_stream_forward.py")
        ctrl.close()
        print("[退出] UDP 已关闭")


if __name__ == "__main__":
    main()