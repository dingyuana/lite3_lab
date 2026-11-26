#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tk keyboard/video dashboard for Lite3 task10."""

import argparse
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from task10_robot import (
    FORWARD_SPEED,
    GAIT_COMMANDS,
    LATERAL_SPEED,
    OBSTACLE_DISTANCE,
    RobotController,
)
from task10_video import VideoStream


class Task10App:
    def __init__(self, root, ip, state_port, video_url,
                 title="Lite3 task10 - 上位机控制", follow_options=None):
        self.root = root
        self.root.title(title)
        self.root.geometry("1180x760")
        self.root.minsize(1000, 660)

        self.follow_options = follow_options or {}
        self.robot = RobotController(
            ip, state_port,
            control_enabled=not self.follow_options.get("offline", False))
        self.video = VideoStream(video_url)
        self.video.start()
        self.follow = None
        self._photo = None
        self._space_down = False
        self._t_down = False

        self._build_ui(ip, state_port, video_url)
        self._bind_keyboard()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(500, lambda: self.robot.speaker_command(2))
        self.root.after(50, self._refresh)

    def _build_ui(self, ip, state_port, video_url):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(0, weight=1)

        video_panel = ttk.LabelFrame(outer, text="机器人视频", padding=8)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        video_panel.columnconfigure(0, weight=1)
        video_panel.rowconfigure(0, weight=1)
        self.video_label = ttk.Label(
            video_panel, text="正在连接 RTSP 视频流…", anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_info = ttk.Label(
            video_panel, text=f"RTSP: {video_url}", wraplength=760)
        self.video_info.grid(row=1, column=0, sticky="w", pady=(8, 0))

        control_panel = ttk.LabelFrame(outer, text="状态与控制", padding=12)
        control_panel.grid(row=0, column=1, sticky="nsew")
        control_panel.columnconfigure(0, weight=1)

        self.connection_label = ttk.Label(
            control_panel, text=f"机器人：{ip}   状态端口：{state_port}")
        self.connection_label.grid(row=0, column=0, sticky="w")
        self.state_label = ttk.Label(control_panel, text="状态：等待状态帧")
        self.state_label.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.posture_label = ttk.Label(control_panel, text="姿态：未知")
        self.posture_label.grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.sensor_label = ttk.Label(control_panel, text="前方：--   后方：--")
        self.sensor_label.grid(row=3, column=0, sticky="w", pady=(5, 0))
        self.speed_label = ttk.Label(control_panel, text="身体速度：--")
        self.speed_label.grid(row=4, column=0, sticky="w", pady=(5, 0))

        self.status_label = ttk.Label(
            control_panel, text="正在初始化", wraplength=390)
        self.status_label.grid(row=5, column=0, sticky="w", pady=(12, 0))
        self.safety_label = ttk.Label(
            control_panel, text="", wraplength=390, foreground="#9a5700")
        self.safety_label.grid(row=6, column=0, sticky="w", pady=(6, 0))

        buttons = ttk.Frame(control_panel)
        buttons.grid(row=7, column=0, sticky="ew", pady=(16, 8))
        ttk.Button(
            buttons, text="空格：起立 / 趴下",
            command=self._toggle_posture).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            buttons, text="停止运动（轴归零）",
            command=self._stop_motion).pack(side=tk.LEFT)
        ttk.Button(
            buttons, text="回零",
            command=self._home).pack(side=tk.LEFT, padx=(8, 0))

        gait_panel = ttk.LabelFrame(control_panel, text="步态切换", padding=6)
        gait_panel.grid(row=8, column=0, sticky="ew", pady=(8, 0))
        gait_panel.columnconfigure(0, weight=1)
        self.gait_selection = ttk.Combobox(
            gait_panel, state="readonly", values=tuple(GAIT_COMMANDS),
            width=22)
        self.gait_selection.set("平地低速步态")
        self.gait_selection.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            gait_panel, text="切换",
            command=self._switch_selected_gait,
        ).grid(row=0, column=1, sticky="e")

        tracking_panel = ttk.LabelFrame(
            control_panel, text="人体跟踪（task12）", padding=6)
        tracking_panel.grid(row=9, column=0, sticky="ew", pady=(8, 0))
        tracking_panel.columnconfigure(0, weight=1)
        self.tracking_status_label = ttk.Label(
            tracking_panel, text="未启动；按 T 开始/停止跟踪",
            wraplength=250)
        self.tracking_status_label.grid(row=0, column=0, sticky="w")
        self.tracking_button = ttk.Button(
            tracking_panel, text="开始跟踪（T）",
            command=self.toggle_tracking)
        self.tracking_button.grid(row=0, column=1, sticky="e", padx=(6, 0))

        speaker_panel = ttk.LabelFrame(
            control_panel, text="扬声器", padding=6)
        speaker_panel.grid(row=10, column=0, sticky="ew", pady=(6, 0))
        self.speaker_status_label = ttk.Label(
            speaker_panel, text="扬声器状态：未知")
        self.speaker_status_label.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            speaker_panel, text="打开",
            command=lambda: self._speaker_command(1),
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            speaker_panel, text="关闭",
            command=lambda: self._speaker_command(0),
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            speaker_panel, text="查询",
            command=lambda: self._speaker_command(2),
        ).pack(side=tk.LEFT)

        ttk.Label(
            control_panel,
            text="方向键：按住移动，松开停止\n"
                 "↑ 前进  ↓ 后退  ← 左移  → 右移\n"
                 "Space 切换起立/趴下；98初始待机时，电量>20%可首次起立\n"
                 "T 开始/停止人体跟踪；Esc 停止运动；“回零”按钮发送 0x21010C05",
            justify=tk.LEFT,
        ).grid(row=11, column=0, sticky="w", pady=(8, 0))
        ttk.Label(
            control_panel,
            text=f"防撞：前进受前方 {OBSTACLE_DISTANCE:.1f}m 阈值保护，"
                 f"后退受后方阈值保护；左右侧向无测距保护，"
                 "操作前请人工确认侧面空间。",
            wraplength=390, justify=tk.LEFT, foreground="#9a2c2c",
        ).grid(row=12, column=0, sticky="w", pady=(8, 0))
        ttk.Label(
            control_panel,
            text=f"控制轴值：前后 ±{FORWARD_SPEED}，"
                 f"左右 ±{LATERAL_SPEED}；轴流 30Hz",
            wraplength=390,
        ).grid(row=13, column=0, sticky="w", pady=(6, 0))

    def _bind_keyboard(self):
        for key in ("Up", "Down", "Left", "Right"):
            direction = key.lower()
            self.root.bind_all(
                f"<KeyPress-{key}>",
                lambda event, d=direction: self._key_down(d))
            self.root.bind_all(
                f"<KeyRelease-{key}>",
                lambda event, d=direction: self._key_up(d))
        self.root.bind_all("<KeyPress-space>", self._space_key_down)
        self.root.bind_all("<KeyRelease-space>", self._space_key_up)
        self.root.bind_all("<KeyPress-t>", self._tracking_key_down)
        self.root.bind_all("<KeyRelease-t>", self._tracking_key_up)
        self.root.bind_all("<KeyPress-T>", self._tracking_key_down)
        self.root.bind_all("<KeyRelease-T>", self._tracking_key_up)
        self.root.bind_all("<KeyPress-Escape>", self._escape_key)
        self.root.bind_all("<FocusOut>", self._focus_lost, add="+")

    def _key_down(self, direction):
        self.robot.set_direction(direction, True)
        return "break"

    def _key_up(self, direction):
        self.robot.set_direction(direction, False)
        return "break"

    def _space_key_down(self, _event):
        if not self._space_down:
            self._space_down = True
            self._toggle_posture()
        return "break"

    def _space_key_up(self, _event):
        self._space_down = False
        return "break"

    def _tracking_key_down(self, _event):
        if not self._t_down:
            self._t_down = True
            self.toggle_tracking()
        return "break"

    def _tracking_key_up(self, _event):
        self._t_down = False
        return "break"

    def _escape_key(self, _event):
        self._stop_tracking()
        self.robot.stop_motion()
        return "break"

    def _focus_lost(self, _event):
        self._stop_tracking()
        self.robot.stop_motion()

    def _toggle_posture(self):
        self._stop_tracking()
        self.robot.toggle_posture()

    def _stop_motion(self):
        self._stop_tracking()
        self.robot.stop_motion()

    def _home(self):
        self._stop_tracking()
        self.robot.home()

    def toggle_tracking(self):
        if self.follow is not None and self.follow.active:
            self._stop_tracking()
            return "break"
        if self.follow is not None:
            self._stop_tracking()

        from task12_follow import FollowSession

        self.tracking_status_label.configure(text="正在加载跟踪模型和标定…")
        self.root.update_idletasks()
        try:
            session = FollowSession(
                self.robot, self.video, **self.follow_options)
            session.start()
        except Exception as exc:
            self.tracking_status_label.configure(text=f"启动失败：{exc}")
            return "break"

        self.follow = session
        self.tracking_button.configure(text="停止跟踪（T）")
        self.tracking_status_label.configure(text="跟踪已启动，正在搜索目标")
        return "break"

    def _stop_tracking(self):
        if self.follow is not None:
            self.follow.stop()
            self.follow = None
        self.tracking_button.configure(text="开始跟踪（T）")
        self.tracking_status_label.configure(text="未启动；按 T 开始/停止跟踪")

    def _switch_selected_gait(self):
        self._stop_tracking()
        self.robot.switch_gait(self.gait_selection.get())

    def _speaker_command(self, value):
        self.robot.speaker_command(value)
        if value in (0, 1):
            self.root.after(300, lambda: self.robot.speaker_command(2))

    @staticmethod
    def _distance_text(distance):
        return f"{distance:.2f} m" if distance is not None else "--"

    def _refresh(self):
        snapshot = self.robot.snapshot()
        state_text = f"状态：{snapshot['state']}"
        if snapshot["state_tuple"] is not None:
            state_text += f"  {snapshot['state_tuple']}"
        self.state_label.configure(text=state_text)
        posture_names = {
            "unknown": "未知",
            "sit": "趴下",
            "stand": "站立",
            "transition": "切换中",
        }
        if snapshot["initial_standby"] and not snapshot["transition"]:
            posture = "初始待机（尚未首次起立）"
        else:
            posture = posture_names.get(
                snapshot["posture"], snapshot["posture"])
        if snapshot["motion_ready"]:
            posture += "（运动就绪）"
        elif snapshot["transition"]:
            posture += f"（{snapshot['transition']}）"
        self.posture_label.configure(text=f"姿态：{posture}")
        self.sensor_label.configure(
            text=f"前方：{self._distance_text(snapshot['front'])}   "
                 f"后方：{self._distance_text(snapshot['back'])}")

        forward = snapshot["forward_speed"]
        lateral = snapshot["lateral_speed"]
        if forward is None or lateral is None:
            speed_text = "身体速度：--"
        else:
            speed_text = (f"身体速度：前后 {forward:+.3f} m/s，"
                          f"左右 {lateral:+.3f} m/s")
        self.speed_label.configure(text=speed_text)
        self.status_label.configure(text=snapshot["status"])
        self.safety_label.configure(text=f"防撞状态：{snapshot['safety']}")
        speaker_enabled = snapshot["speaker_enabled"]
        if speaker_enabled is None:
            speaker_text = snapshot["speaker_command_status"]
        else:
            state = "开启" if speaker_enabled else "关闭"
            speaker_text = f"扬声器状态：{state}"
        self.speaker_status_label.configure(text=speaker_text)

        frame, video_error = self.video.latest()
        if self.follow is not None:
            follow_frame, follow_status = self.follow.latest()
            self.tracking_status_label.configure(text=follow_status)
            if follow_frame is not None:
                frame = follow_frame
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            resampling = getattr(Image, "Resampling", Image)
            image.thumbnail((800, 600), resampling.LANCZOS)
            self._photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=self._photo, text="")
        elif video_error:
            self.video_label.configure(text=video_error, image="")
        self.root.after(50, self._refresh)

    def close(self):
        self._stop_tracking()
        self.robot.stop_motion()
        self.video.close()
        self.robot.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Lite3 task10 键盘与视频控制界面")
    parser.add_argument("--ip", default="192.168.1.120", help="运动主机 IP")
    parser.add_argument("--state-port", type=int, default=43897,
                        help="本地状态帧监听端口")
    parser.add_argument(
        "--video-url", default="rtsp://192.168.1.120:8554/test?tcp",
        help="机器人 RTSP 视频地址")
    args = parser.parse_args()
    if args.state_port <= 0:
        parser.error("--state-port 必须为正数，task10 需要超声波状态帧")

    root = tk.Tk()
    try:
        Task10App(root, args.ip, args.state_port, args.video_url)
    except OSError as exc:
        root.destroy()
        parser.error(f"无法初始化机器人 UDP 状态端口：{exc}")
    root.mainloop()


if __name__ == "__main__":
    main()
