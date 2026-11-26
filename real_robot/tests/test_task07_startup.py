import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import lite3_common as l3
import task07_axis_stream_forward as task07


def _state(basic, gait=0, motion=0):
    state = l3.RobotStateUpload()
    state.robot_basic_state = basic
    state.robot_gait_state = gait
    state.robot_motion_state = motion
    return state


class Task07StartupGateTests(unittest.TestCase):
    def test_sit_state_proceeds_without_homing(self):
        controller = Mock()
        controller.get_state.return_value = _state(1)
        controller.knee_mean_deg.return_value = 150.0

        with contextlib.redirect_stdout(io.StringIO()):
            with patch("task07_axis_stream_forward.l3.ensure_home") as ensure_home:
                self.assertTrue(task07._startup_state_gate(controller))

        ensure_home.assert_not_called()

    def test_98_state_proceeds_without_homing(self):
        controller = Mock()
        controller.get_state.return_value = _state(98)
        controller.knee_mean_deg.return_value = 150.0

        with contextlib.redirect_stdout(io.StringIO()):
            with patch("task07_axis_stream_forward.l3.ensure_home") as ensure_home:
                self.assertTrue(task07._startup_state_gate(controller))

        ensure_home.assert_not_called()

    def test_other_state_homes_and_requires_sit_confirmation(self):
        controller = Mock()
        controller.get_state.side_effect = [_state(6), _state(1)]
        controller.knee_mean_deg.return_value = 80.0

        with contextlib.redirect_stdout(io.StringIO()):
            with patch("task07_axis_stream_forward.l3.ensure_home",
                       return_value=True) as ensure_home:
                self.assertTrue(task07._startup_state_gate(controller))

        ensure_home.assert_called_once_with(controller)

    def test_failed_home_blocks_task_start(self):
        controller = Mock()
        controller.get_state.return_value = _state(9)
        controller.knee_mean_deg.return_value = 80.0

        with contextlib.redirect_stdout(io.StringIO()):
            with patch("task07_axis_stream_forward.l3.ensure_home",
                       return_value=False) as ensure_home:
                self.assertFalse(task07._startup_state_gate(controller))

        ensure_home.assert_called_once_with(controller)


class Task07ForwardTests(unittest.TestCase):
    def test_forward_passes_cooldown_to_safe_stop(self):
        controller = Mock()
        controller.get_state.return_value = None
        stream = Mock()

        with contextlib.redirect_stdout(io.StringIO()):
            with patch("task07_axis_stream_forward.time.time",
                       side_effect=[0.0, 0.2]):
                with patch("task07_axis_stream_forward._safe_stop",
                           return_value=True) as safe_stop:
                    self.assertTrue(task07._phase_c_forward(
                        controller, stream, 9000, 0.1, cooldown=1.5))

        self.assertEqual(safe_stop.call_args.kwargs["cooldown"], 1.5)


if __name__ == "__main__":
    unittest.main()
