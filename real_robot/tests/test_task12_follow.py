import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from task10_gui import Task10App
from task10_video import VideoStream
from task12_follow import (
    ConfigurationError,
    FollowSession,
    SafetySupervisor,
    twist_to_axis,
)


class TrackingKeyTests(unittest.TestCase):
    def test_t_key_toggles_once_until_released(self):
        app = Task10App.__new__(Task10App)
        app._t_down = False
        app.toggle_tracking = Mock(return_value="break")

        self.assertEqual(app._tracking_key_down(None), "break")
        app._tracking_key_down(None)
        app.toggle_tracking.assert_called_once_with()

        app._tracking_key_up(None)
        app._tracking_key_down(None)
        self.assertEqual(app.toggle_tracking.call_count, 2)

    def test_stop_tracking_releases_session_and_updates_button(self):
        app = Task10App.__new__(Task10App)
        app.follow = Mock()
        session = app.follow
        app.tracking_button = Mock()
        app.tracking_status_label = Mock()

        app._stop_tracking()

        session.stop.assert_called_once_with()
        self.assertIsNone(app.follow)
        app.tracking_button.configure.assert_called_once_with(text="开始跟踪（T）")


class VideoTimestampTests(unittest.TestCase):
    def test_latest_with_timestamp_preserves_frame_age(self):
        video = VideoStream("")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        video._frame = frame
        video._frame_time = 123.0
        video._error = None

        latest, error, timestamp = video.latest_with_timestamp()

        self.assertTrue(np.array_equal(frame, latest))
        self.assertIsNone(error)
        self.assertEqual(timestamp, 123.0)

    def test_error_invalidates_timestamp(self):
        video = VideoStream("")
        video._frame_time = 123.0

        video._set_error("video disconnected")

        _, error, timestamp = video.latest_with_timestamp()
        self.assertEqual(error, "video disconnected")
        self.assertEqual(timestamp, 0.0)


class FollowSafetyTests(unittest.TestCase):
    def test_ultrasonic_uses_original_frame_age(self):
        supervisor = SafetySupervisor()
        supervisor.update_ultrasonic(2.0, age=0.4)

        allowed, reason, decision = supervisor.check(0.0, 0.0, 2.0)

        self.assertFalse(allowed)
        self.assertEqual(decision, "STOP")
        self.assertIn("超声波数据超时", reason)

    def test_reverse_output_clears_forward_deadzone(self):
        forward, _ = twist_to_axis(-0.2, 0.0, fwd_scale=20000)

        self.assertLess(forward, -6553)

    def test_yaw_output_clears_yaw_deadzone(self):
        _, yaw = twist_to_axis(0.0, -0.05, yaw_scale=65534)

        self.assertLess(yaw, -9553)

    def test_watchdog_stops_robot_and_deactivates_session(self):
        robot = Mock()
        session = FollowSession(robot, Mock(), offline=False)
        session.active = True
        session._last_update = time.monotonic() - session.WATCHDOG_TIMEOUT - 0.1

        triggered = session._check_watchdog()

        self.assertTrue(triggered)
        robot.set_tracking_axes.assert_called_once_with(0, 0)
        robot.stop_tracking.assert_called_once_with()
        self.assertFalse(session.active)

    def test_real_mode_rejects_dummy_homography_file(self):
        robot = Mock()
        video = Mock()
        video.latest.return_value = (np.zeros((8, 8, 3), dtype=np.uint8), None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "H_dummy.npy"
            np.save(path, np.eye(3))
            session = FollowSession(
                robot, video, h_path=path, dummy_h=False, offline=False)

            with self.assertRaisesRegex(ConfigurationError, "占位"):
                session.start()


class FollowSessionTests(unittest.TestCase):
    def test_stop_returns_robot_to_manual_mode(self):
        robot = Mock()
        session = FollowSession(robot, Mock(), offline=False)

        session.stop()

        robot.stop_tracking.assert_called_once_with()
        self.assertFalse(session.active)

    def test_annotation_draws_target_without_modifying_source(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        target = (80.0, 100.0, 60.0, 20.0, 100.0, 110.0)

        display = FollowSession._annotate(
            frame, target, 2.0, "tracking", offline=False)

        self.assertFalse(np.array_equal(frame, display))
        self.assertEqual(frame.shape, display.shape)


if __name__ == "__main__":
    unittest.main()
