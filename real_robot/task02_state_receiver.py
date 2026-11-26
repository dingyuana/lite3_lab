#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task02_state_receiver.py —— 状态帧与关节帧观测（不动机器人）
兼容 Python 3.8+，用 python3 运行。

日志变化（相对上一版）：
  - 增加状态三元组 (basic, gait, motion) 和官方状态表语义。
  - 判据以状态三元组为主，膝关节角度仅作辅助，不再标定膝关节阈值。
  - 只有连接失败时才显示“端口不可达”，正常不显示。

用法：
  python3 task02_state_receiver.py
  python3 task02_state_receiver.py --joints
  python3 task02_state_receiver.py --interval 1 --duration 20
"""

import argparse
import math
import time
import lite3_common as l3


def _fmt_angle(rad: float) -> str:
    return f"{rad:+7.3f}rad({math.degrees(rad):+7.1f}°)"


def _print_state(st, ctrl, show_joints: bool):
    if st is None:
        print("  状态帧：无（0x0901 未到达或已超时）")
        return

    t, desc = l3.state_tuple(st)
    print(f"  状态三元组：(basic, gait, motion) = {t}  →  {desc}")

    print(f"  策略={st.robot_policy_state}  "
          f"动作={st.robot_motion_state}  "
          f"电量={st.battery_level:.2f}%  "
          f"需踏步调整={st.is_robot_need_move}")
    print(f"  姿态角=({st.rpy[0]:+.1f}, {st.rpy[1]:+.1f}, "
          f"{st.rpy[2]:+.1f})°  "
          f"前向速度={st.vel_body[0]:+.3f} m/s  "
          f"侧向速度={st.vel_body[1]:+.3f} m/s  "
          f"偏航角速度={st.vel_body[2]:+.3f} rad/s")
    print(f"  超声波=前 {st.ultrasound[0]:.2f} m / 后 {st.ultrasound[1]:.2f} m")

    ja = ctrl.get_joint()
    if ja is None:
        print("  关节帧：无（0x0902 未到达或已超时）")
        return

    mean_deg = ctrl.knee_mean_deg()
    knees = ctrl.knee_angles()
    if mean_deg is not None and knees is not None:
        degs = [f"{math.degrees(k):+.1f}" for k in knees]
        print(f"  四膝均值={mean_deg:+.1f}°  四膝 deg=[{', '.join(degs)}]")

    if show_joints:
        print("  12 关节角度：")
        for i in range(12):
            mark = " *" if i in l3.KNEE_INDICES else "  "
            print(f"    [{i:2d}]{mark} {l3.JOINT_NAMES[i]:6s}  "
                  f"{_fmt_angle(ja.joint_angle[i])}")


def main():
    ap = argparse.ArgumentParser(description="状态帧与关节帧观测（不动机器人）")
    l3.add_common_args(ap)
    ap.add_argument("--interval", type=float, default=2.0, help="打印间隔（秒）")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="运行秒数；0 表示一直运行直到 Ctrl-C")
    ap.add_argument("--joints", action="store_true", help="额外打印 12 关节明细")
    args = ap.parse_args()

    if not args.state_port or args.state_port <= 0:
        print("[错误] 任务 2 必须接收状态帧，请设置 --state-port（默认 43897）")
        return

    ctrl = l3.Lite3Controller(args.ip, state_port=args.state_port)
    ctrl.start_heartbeat(hz=5)
    ctrl.start_state_receiver()

    print(f"[任务2] {args.ip}:43893  心跳 5Hz  监听 UDP :{args.state_port}")
    print(f"[任务2] 打印间隔 {args.interval}s  "
          f"{'持续运行（Ctrl-C 退出）' if args.duration <= 0 else f'运行 {args.duration}s'}")
    print(f"[任务2] 12 关节明细：{'开启' if args.joints else '关闭（加 --joints 开启）'}")
    print("[任务2] 本任务不发送任何运动指令，机器人保持当前姿态")
    print("=" * 118)

    t0 = time.time()
    next_print = 0.0
    last_st = None

    try:
        while True:
            now = time.time() - t0
            if args.duration > 0 and now >= args.duration:
                break
            if now >= next_print:
                st = ctrl.get_state()
                if st is not None:
                    last_st = st
                s = ctrl.stats()
                icmp_str = "  ⚠端口不可达=True" if s["icmp_unreachable"] else ""
                print(f"\n[时间 {now:6.1f}s]  发送={s['tx_count']:5d}  "
                      f"状态帧={s['rx_count']:6d}  关节帧={s['joint_rx_count']:6d}"
                      f"{icmp_str}")
                _print_state(last_st, ctrl, args.joints)
                next_print += args.interval
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[中断] 用户 Ctrl-C")
    finally:
        s = ctrl.stats()
        final_st = last_st
        final_knee = ctrl.knee_mean_deg()
        final_knees = ctrl.knee_angles()
        ctrl.close()

    print("\n" + "=" * 118)
    print("== 任务 2 验收总结 ==")
    print(f"发送心跳: {s['tx_count']} 包   "
          f"状态帧: {s['rx_count']} 帧   "
          f"关节帧: {s['joint_rx_count']} 帧")

    ok = True
    if s["icmp_unreachable"]:
        print("[失败] 收到 ICMP 端口不可达 → 对端 43893 无进程监听")
        ok = False
    if s["rx_count"] == 0:
        print("[失败] 未收到 0x0901 状态帧 → 检查上报配置 / 端口占用 / 防火墙")
        ok = False
    if s["joint_rx_count"] == 0:
        print("[警告] 未收到 0x0902 关节帧 → task03 将改用状态三元组判断")
    if not ok:
        return

    if final_st is not None:
        t, desc = l3.state_tuple(final_st)
        print("[通过] 状态帧解析正常：")
        print(f"       状态三元组 = {t}  {desc}")
        if l3.is_initial_standby(final_st):
            print("       语义       = 初始待机：尚未触发首次起立；"
                  "不表示故障、起立完成或回零完成")
        elif l3.is_stand_still(final_st):
            print(f"       语义       = 平地低速 + 静止（等价 (6,0,0)）")
        else:
            print(f"       语义       = {desc}")
        print(f"       电量       = {final_st.battery_level:.3f}%")
        print(f"       前向速度   = {final_st.vel_body[0]:+.3f} m/s")
        print(f"       姿态角     = ({final_st.rpy[0]:+.1f}, "
              f"{final_st.rpy[1]:+.1f}, {final_st.rpy[2]:+.1f})")

    if final_knee is not None and final_knees is not None:
        print("[通过] 关节帧解析正常：")
        print(f"       四膝均值 = {final_knee:+.1f}°")
        print(f"       四膝 deg = "
              f"[{', '.join(f'{math.degrees(k):+.1f}' for k in final_knees)}]")

    # ============ 修正后的“下一步” ============
    print()
    print("[下一步] 判据以状态三元组 (basic, gait, motion) 为主，"
          "膝关节角度仅作辅助。")
    if final_st is not None:
        t, desc = l3.state_tuple(final_st)
        print(f"       当前状态三元组 = {t}  →  {desc}")
        if l3.is_initial_standby(final_st):
            print("       语义 = 初始待机；首次起立需电量高于20%，"
                  "task03 会等待实际状态确认")
        elif l3.is_stand_still(final_st):
            print("       语义 = 平地低速 + 静止（等价 (6,0,0)），"
                  "可直接跑 task03")
        else:
            print("       语义 = 非静止站立，跑 task03 时会先起立到 (6,0,0)")
    print("       下一步命令：python3 task03_stand_to_force.py")


if __name__ == "__main__":
    main()