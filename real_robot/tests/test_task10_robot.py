import threading
import unittest
from unittest.mock import Mock, patch

import lite3_common as l3
from task10_robot import (
    AxisStream,
    RobotController,
    safe_axis_values,
    safe_follow_axes,
)


class SafeAxisValuesTests(unittest.TestCase):
    def test_front_obstacle_blocks_forward_but_not_reverse(self):
        axes, _, warning = safe_axis_values(
            {"up"}, 0.8, 2.0, motion_allowed=True)
        self.assertEqual(axes, 0)
        self.assertIn("前进已拦截", warning)
        self.assertEqual(
            safe_axis_values({"down"}, 0.8, 2.0, True)[0], -9000)

    def test_rear_obstacle_blocks_reverse_but_not_forward(self):
        self.assertEqual(
            safe_axis_values({"up"}, 2.0, 0.8, True)[0], 9000)
        axes, _, warning = safe_axis_values(
            {"down"}, 2.0, 0.8, motion_allowed=True)
        self.assertEqual(axes, 0)
        self.assertIn("后退已拦截", warning)

    def test_lateral_motion_has_no_ultrasonic_gate(self):
        self.assertEqual(
            safe_axis_values({"left"}, 0.4, 0.5, True)[:2], (0, -16000))

    def test_motion_is_zero_until_controller_is_ready(self):
        self.assertEqual(
            safe_axis_values({"up", "right"}, 2.0, 2.0, False)[:2], (0, 0))

    def test_sitting_rejects_direction_key(self):
        robot = RobotController.__new__(RobotController)
        robot._lock = threading.RLock()
        robot._posture = "sit"
        robot._motion_ready = False
        robot._transition = ""
        robot._keys = set()
        robot._status = ""

        robot.set_direction("up", True)

        self.assertNotIn("up", robot._keys)
        self.assertIn("趴下状态禁止移动", robot._status)


class AxisStreamTests(unittest.TestCase):
    def test_tracking_yaw_is_sent_to_robot(self):
        ctrl = Mock()
        stream = AxisStream(ctrl)
        stream._stop = Mock()
        stream._stop.is_set.side_effect = [False, True]
        stream.set_axes(8000, 0, 12000)

        stream._loop()

        ctrl.send_simple.assert_any_call(l3.CMD_AXIS_FORWARD, 8000)
        ctrl.send_simple.assert_any_call(l3.CMD_AXIS_LATERAL, 0)
        ctrl.send_simple.assert_any_call(l3.CMD_AXIS_YAW, 12000)


class FollowAxisTests(unittest.TestCase):
    def test_front_obstacle_blocks_tracking_forward(self):
        axis, warning = safe_follow_axes(8000, 0.8, 2.0)
        self.assertEqual(axis, 0)
        self.assertIn("跟踪前进已拦截", warning)

    def test_rear_obstacle_blocks_tracking_reverse(self):
        axis, warning = safe_follow_axes(-4000, 2.0, 0.8)
        self.assertEqual(axis, 0)
        self.assertIn("跟踪后退已拦截", warning)

    def test_valid_distance_keeps_tracking_axis(self):
        self.assertEqual(safe_follow_axes(8000, 2.0, 2.0), (8000, ""))


class TrackingControlTests(unittest.TestCase):
    @staticmethod
    def _robot():
        robot = RobotController.__new__(RobotController)
        robot._lock = threading.RLock()
        robot._closing = False
        robot._transition = ""
        robot._posture = "stand"
        robot._motion_ready = True
        robot._keys = {"up"}
        robot._follow_active = False
        robot._follow_axes = (0, 0)
        robot._axis_stream = Mock()
        robot._status = ""
        robot.control_enabled = True
        return robot

    def test_tracking_start_requires_ready_robot(self):
        robot = self._robot()
        robot._motion_ready = False
        with self.assertRaisesRegex(RuntimeError, "先按空格"):
            robot.start_tracking()

    def test_tracking_axes_are_clamped_and_stop_at_zero(self):
        robot = self._robot()
        robot.start_tracking()
        robot.set_tracking_axes(40000, -40000)
        self.assertEqual(robot._follow_axes, (32767, -32767))
        self.assertNotIn("up", robot._keys)

        robot.stop_tracking()

        self.assertFalse(robot._follow_active)
        self.assertEqual(robot._follow_axes, (0, 0))
        robot._axis_stream.set_axes.assert_called_once_with(0, 0)

    def test_manual_direction_is_disabled_during_tracking(self):
        robot = self._robot()
        robot.start_tracking()
        robot.set_direction("up", True)
        self.assertNotIn("up", robot._keys)
        self.assertIn("方向键已禁用", robot._status)

    def test_offline_controller_rejects_tracking(self):
        robot = self._robot()
        robot.control_enabled = False
        with self.assertRaisesRegex(RuntimeError, "脱机模式"):
            robot.start_tracking()


class UnhomedStateTests(unittest.TestCase):
    @staticmethod
    def _state(basic, battery=85.0):
        state = l3.RobotStateUpload()
        state.robot_basic_state = basic
        state.battery_level = battery
        return state

    def test_98_is_not_a_confirmed_motion_posture(self):
        self.assertFalse(
            RobotController._is_motion_posture(self._state(98), 80.0))
        self.assertTrue(
            RobotController._is_motion_posture(self._state(6), 80.0))
        self.assertFalse(
            RobotController._is_motion_posture(self._state(98), 150.0))

    def test_98_triggers_first_stand_only_above_20_percent(self):
        robot = RobotController.__new__(RobotController)
        robot._lock = threading.RLock()
        robot._closing = False
        robot._transition = ""
        robot._status = ""
        robot._posture = "unknown"
        robot._motion_ready = False
        robot.ctrl = Mock()
        robot.ctrl.get_state.return_value = self._state(98, battery=20.1)
        robot.ctrl.knee_mean_deg.return_value = 150.0
        robot._start_transition = Mock()

        robot.toggle_posture()

        robot._start_transition.assert_called_once_with(
            "初始待机（98）：首次起立中，请保持周围空间安全",
            robot._stand_and_prepare)

    def test_98_blocks_first_stand_at_20_percent_or_lower(self):
        robot = RobotController.__new__(RobotController)
        robot._lock = threading.RLock()
        robot._closing = False
        robot._transition = ""
        robot._status = ""
        robot._posture = "unknown"
        robot._motion_ready = False
        robot.ctrl = Mock()
        robot.ctrl.get_state.return_value = self._state(98, battery=20.0)
        robot.ctrl.knee_mean_deg.return_value = 150.0
        robot._start_transition = Mock()

        robot.toggle_posture()

        robot._start_transition.assert_not_called()
        self.assertIn("不高于20%", robot._status)

    def test_first_stand_rechecks_battery_before_sending(self):
        robot = RobotController.__new__(RobotController)
        robot.ctrl = Mock()
        robot.ctrl.get_state.return_value = self._state(98, battery=20.0)

        with self.assertRaisesRegex(RuntimeError, "禁止首次起立"):
            robot._stand_and_prepare()

        robot.ctrl.send_simple.assert_not_called()

    def test_home_sends_only_home_command_and_waits_for_sit(self):
        robot = RobotController.__new__(RobotController)
        robot._lock = threading.RLock()
        robot._axis_stream = None
        robot._cancel_transition = threading.Event()
        robot.ctrl = Mock()
        robot.ctrl.get_state.side_effect = [
            self._state(17),
            self._state(1),
        ]

        with patch("task10_robot.time.sleep"):
            result = robot._home()

        self.assertEqual(result[0:2], ("sit", False))
        robot.ctrl.send_simple.assert_called_once_with(l3.CMD_HOME)

    def test_home_button_accepts_98(self):
        robot = RobotController.__new__(RobotController)
        robot._lock = threading.RLock()
        robot._closing = False
        robot._transition = ""
        robot.ctrl = Mock()
        robot.ctrl.get_state.return_value = self._state(98)
        robot._start_transition = Mock()

        robot.home()

        robot._start_transition.assert_called_once_with(
            "正在回零", robot._home)


class DashboardCommandTests(unittest.TestCase):
    @staticmethod
    def _state(basic=6, gait=0):
        state = l3.RobotStateUpload()
        state.robot_basic_state = basic
        state.robot_gait_state = gait
        return state

    def test_gait_switch_requires_ready_non_ai_motion_state(self):
        robot = RobotController.__new__(RobotController)
        robot._lock = threading.RLock()
        robot._closing = False
        robot._transition = ""
        robot._posture = "stand"
        robot._motion_ready = True
        robot._status = ""
        robot.ctrl = Mock()
        robot.ctrl.get_state.return_value = self._state(basic=16)
        robot.ctrl.knee_mean_deg.return_value = 80.0
        robot._start_transition = Mock()

        robot.switch_gait("平地中速步态")

        robot._start_transition.assert_not_called()
        self.assertIn("非 AI", robot._status)

    def test_gait_switch_sends_selected_command_and_waits_for_confirmation(self):
        robot = RobotController.__new__(RobotController)
        robot._lock = threading.RLock()
        robot._closing = False
        robot._transition = ""
        robot._posture = "stand"
        robot._motion_ready = True
        robot._status = ""
        robot._axis_stream = Mock()
        robot._wait_until = Mock(side_effect=[True, True])
        robot.ctrl = Mock()
        robot.ctrl.get_state.return_value = self._state(gait=0)
        robot.ctrl.knee_mean_deg.return_value = 80.0
        robot._start_transition = Mock()

        robot.switch_gait("平地中速步态")

        label, action = robot._start_transition.call_args[0]
        self.assertEqual(label, "切换步态：平地中速步态")
        result = action()

        robot.ctrl.send_simple.assert_called_once_with(l3.CMD_GAIT_MID)
        self.assertEqual(result, ("stand", True, "步态已切换：平地中速步态"))
        robot._axis_stream.set_axes.assert_called_once_with(0, 0)

    def test_speaker_command_sends_protocol_value(self):
        robot = RobotController.__new__(RobotController)
        robot._lock = threading.RLock()
        robot._closing = False
        robot._speaker_command_status = ""
        robot.ctrl = Mock()

        robot.speaker_command(1)

        robot.ctrl.clear_speaker_state.assert_called_once_with()
        robot.ctrl.send_simple.assert_called_once_with(l3.CMD_SPEAKER, 1)


if __name__ == "__main__":
    unittest.main()
