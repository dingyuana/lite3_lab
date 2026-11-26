#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task01_heartbeat.py —— 心跳保活 + 双向连通性验证（中文日志）
兼容 Python 3.8+，用 python3 运行。

日志变化（相对上一版）：
  - 不再每秒显示“端口不可达=False”，只有连接失败（True）时才追加警告。
  - 每秒增加状态三元组 (basic, gait, motion) 和官方状态表语义。
  - 其余列（步态/电量/前向速度/姿态角/四膝均值）保留。

验收：
  - [通过] 双向链路确认：心跳在发 + 状态帧在回
  - [失败] 收到 ICMP 端口不可达
  - [警告] 未收到状态帧
  - [提示] 未启用状态监听，仅证明本机发出
"""

import argparse
import time
import lite3_common as l3


def _line(st, ctrl) -> str:
    """一行状态摘要：状态三元组 + 步态/电量/速度/姿态角/四膝均值"""
    if st is None:
        return ("状态=无  步态=  ?  电量=   ??  前向速度=  ?    "
                "姿态角=(  ?,  ?,  ?)  四膝均值=  ?")

    t, desc = l3.state_tuple(st)
    mean_deg = ctrl.knee_mean_deg()
    knee_str = f"{mean_deg:+6.1f}°" if mean_deg is not None else "   ?  "

    return (f"状态={t} {desc}  "
            f"步态={st.robot_gait_state:3d}  "
            f"电量={st.battery_level:6.2f}  "
            f"前向速度={st.vel_body[0]:+.3f}  "
            f"姿态角=({st.rpy[0]:+.1f},{st.rpy[1]:+.1f},{st.rpy[2]:+.1f})  "
            f"四膝均值={knee_str}")


def main():
    ap = argparse.ArgumentParser(description="心跳保活 + 双向连通性验证")
    l3.add_common_args(ap)
    ap.add_argument("--duration", type=float, default=30.0, help="运行秒数")
    ap.add_argument("--hb-hz", type=float, default=5.0, help="心跳频率，≥2Hz")
    args = ap.parse_args()

    state_port = args.state_port if args.state_port and args.state_port > 0 else None

    ctrl = l3.Lite3Controller(args.ip, state_port=state_port)
    ctrl.start_heartbeat(hz=args.hb_hz)
    if state_port:
        ctrl.start_state_receiver()

    print(f"[心跳] {args.ip}:43893  0x21040001 @ {args.hb_hz}Hz  {args.duration}s")
    if state_port:
        print(f"[状态] 同时监听 UDP :{state_port} 接收 0x0901 + 0x0902")
    else:
        print("[状态] 未启用状态监听（--state-port 0）：只能统计发送")
    print("-" * 150)

    t0 = time.time()
    next_print = 1.0
    last_st = None

    try:
        while time.time() - t0 < args.duration:
            now = time.time() - t0
            if now >= next_print:
                st = ctrl.get_state()
                if st is not None:
                    last_st = st
                s = ctrl.stats()

                # 只有连接失败时才显示“端口不可达”，正常不显示
                icmp_str = "  ⚠端口不可达=True" if s["icmp_unreachable"] else ""

                print(f"时间={now:5.1f}s  "
                      f"发送={s['tx_count']:4d}  "
                      f"接收={s['rx_count']:5d}  "
                      f"关节帧={s['joint_rx_count']:5d}  "
                      f"{_line(last_st, ctrl)}{icmp_str}")
                next_print += 1.0
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[中断] 用户 Ctrl-C")
    finally:
        s = ctrl.stats()
        final_st = last_st
        final_knee = ctrl.knee_mean_deg()
        ctrl.close()

    print("-" * 150)
    print("== 连通性结论 ==")
    print(f"发送心跳: {s['tx_count']} 包   "
          f"收到状态帧: {s['rx_count']} 帧   "
          f"收到关节帧: {s['joint_rx_count']} 帧")

    if s["icmp_unreachable"]:
        print("[失败] 收到 ICMP 端口不可达 → 对端 43893 无进程监听")
        return
    if not state_port:
        print("[提示] 仅证明本机发出；去掉 --state-port 0 并配置上报目标可做双向确认")
        return
    if s["rx_count"] == 0:
        print("[警告] 未收到状态帧 → 检查上报目标 / 端口占用 / 防火墙")
        return

    print("[通过] 双向链路确认：心跳在发 + 状态帧在回")
    if final_st is not None:
        t, desc = l3.state_tuple(final_st)
        print(f"       状态三元组 = {t}  {desc}")
        if l3.is_stand_still(final_st):
            print(f"       语义       = 平地低速 + 静止（等价 (6,0,0)）")
        else:
            print(f"       语义       = 非静止站立"
                  f"（gait={final_st.robot_gait_state} "
                  f"motion={final_st.robot_motion_state}）")
        print(f"       电量       = {final_st.battery_level:.3f}%")
        print(f"       前向速度   = {final_st.vel_body[0]:+.3f} m/s")
        print(f"       姿态角     = ({final_st.rpy[0]:+.1f}, "
              f"{final_st.rpy[1]:+.1f}, {final_st.rpy[2]:+.1f})")
        if final_knee is not None:
            print(f"       四膝均值   = {final_knee:+.1f}°")

    print()
    if s["joint_rx_count"] == 0:
        print("[警告] 未收到 0x0902 关节帧 → 无法用膝关节角度辅助判断姿态")
    else:
        print("[提示] 关节帧正常上报，下一步：")
        print("       python3 task02_state_receiver.py")


if __name__ == "__main__":
    main()