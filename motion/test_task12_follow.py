import unittest
from unittest.mock import Mock

import numpy as np

from task10_gui import Task10App
from task12_follow import FollowSession


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
