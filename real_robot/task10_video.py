#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Background RTSP reader for the task10 Tk interface."""

import threading
import time

import cv2


class VideoStream:
    def __init__(self, url, reconnect_seconds=2.0):
        self.url = url
        self.reconnect_seconds = reconnect_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._frame_time = 0.0
        self._error = "正在连接视频流"
        self._thread = None
        self._capture = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _set_error(self, message):
        with self._lock:
            self._error = message
            if message:
                self._frame = None
                self._frame_time = 0.0

    def _run(self):
        while not self._stop.is_set():
            try:
                capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            except cv2.error as exc:
                self._set_error(f"无法创建视频流：{exc}")
                self._stop.wait(self.reconnect_seconds)
                continue
            self._capture = capture
            if not capture.isOpened():
                capture.release()
                self._capture = None
                self._set_error("无法打开 RTSP 视频流，检查地址和网络")
                self._stop.wait(self.reconnect_seconds)
                continue

            self._set_error("")
            while not self._stop.is_set():
                try:
                    ok, frame = capture.read()
                except cv2.error as exc:
                    self._set_error(f"视频读取错误，正在重连：{exc}")
                    break
                if not ok:
                    self._set_error("视频流中断，正在重连")
                    break
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.monotonic()
                    self._error = ""

            capture.release()
            self._capture = None
            if not self._stop.is_set():
                self._stop.wait(self.reconnect_seconds)

    def latest(self):
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            return frame, self._error

    def latest_with_timestamp(self):
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            return frame, self._error, self._frame_time

    def close(self):
        self._stop.set()
        capture = self._capture
        if capture is not None:
            capture.release()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
