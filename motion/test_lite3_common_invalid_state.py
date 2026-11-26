import socket
import struct
import threading
import unittest

import lite3_common as l3


class _FakeReceiver:
    def __init__(self, packets, stop_event):
        self.packets = iter(packets)
        self.stop_event = stop_event

    def recvfrom(self, _max_size):
        try:
            return next(self.packets), ("127.0.0.1", 43897)
        except StopIteration:
            self.stop_event.set()
            raise socket.timeout


def _state_packet(basic_state, front=0.0, back=0.0):
    state = l3.RobotStateUpload()
    state.robot_basic_state = basic_state
    state.ultrasound[0] = front
    state.ultrasound[1] = back
    payload = bytes(state)
    return struct.pack(
        "<III", l3.CODE_ROBOT_STATE, len(payload), 1) + payload


def _speaker_status_packet(enabled):
    return struct.pack(
        "<III", l3.CODE_SPEAKER_STATUS, int(enabled), 0)


class InitialStandbyStateTests(unittest.TestCase):
    def test_initial_standby_is_a_valid_state_frame(self):
        state = l3.RobotStateUpload()
        state.robot_basic_state = l3.ROBOT_STATE_INITIAL_STANDBY

        self.assertTrue(l3.is_valid_robot_state(state))
        self.assertEqual(
            l3.state_tuple(state)[1],
            "初始待机（运控程序启动后尚未触发首次起立）")

    def test_known_robot_state_is_valid(self):
        for basic_state in (1, 6):
            with self.subTest(basic_state=basic_state):
                state = l3.RobotStateUpload()
                state.robot_basic_state = basic_state
                self.assertTrue(l3.is_valid_robot_state(state))

    def test_missing_state_is_invalid(self):
        self.assertFalse(l3.is_valid_robot_state(None))

    def test_initial_standby_is_not_equivalent_to_home(self):
        state = l3.RobotStateUpload()
        self.assertFalse(l3.is_initial_standby(None))

        state.robot_basic_state = l3.ROBOT_STATE_INITIAL_STANDBY
        self.assertTrue(l3.is_initial_standby(state))
        self.assertFalse(l3.needs_home(state))

        state.robot_basic_state = 1
        self.assertFalse(l3.is_initial_standby(state))

    def test_only_98_state_code_is_red(self):
        state = l3.RobotStateUpload()
        state.robot_basic_state = 98
        self.assertEqual(
            l3.format_state_code(state), "\033[31m(98, 0, 0)\033[0m")

        state.robot_basic_state = 6
        self.assertEqual(l3.format_state_code(state), "(6, 0, 0)")

    def test_receiver_retains_98_frame_and_accepts_following_valid_frame(self):
        controller = l3.Lite3Controller.__new__(l3.Lite3Controller)
        controller._stop = threading.Event()
        controller.rx = _FakeReceiver(
            [_state_packet(98, 4.5, 0.86), _state_packet(6)],
            controller._stop)
        controller._state_lock = threading.Lock()
        controller._last_state = None
        controller._last_state_t = 0.0
        controller._stats_lock = threading.Lock()
        controller.tx_count = 0
        controller.rx_count = 0
        controller.invalid_state_rx_count = 0
        controller.joint_rx_count = 0
        controller.icmp_unreachable = False

        controller._rx_loop()

        self.assertEqual(controller.stats()["invalid_state_rx_count"], 0)
        self.assertEqual(controller.stats()["rx_count"], 2)
        self.assertEqual(l3.state_tuple(controller.get_state())[0], (6, 0, 0))

    def test_only_98_frames_still_provide_telemetry(self):
        controller = l3.Lite3Controller.__new__(l3.Lite3Controller)
        controller._stop = threading.Event()
        controller.rx = _FakeReceiver(
            [_state_packet(98, 4.5, 0.86)], controller._stop)
        controller._state_lock = threading.Lock()
        controller._last_state = None
        controller._last_state_t = 0.0
        controller._stats_lock = threading.Lock()
        controller.tx_count = 0
        controller.rx_count = 0
        controller.invalid_state_rx_count = 0
        controller.joint_rx_count = 0
        controller.icmp_unreachable = False

        controller._rx_loop()

        state = controller.get_state()
        self.assertIsNotNone(state)
        self.assertEqual(l3.state_tuple(state)[0], (98, 0, 0))
        self.assertAlmostEqual(state.ultrasound[0], 4.5)
        self.assertAlmostEqual(state.ultrasound[1], 0.86)
        self.assertEqual(controller.stats()["invalid_state_rx_count"], 0)
        self.assertEqual(controller.stats()["rx_count"], 1)

    def test_speaker_status_simple_frames_are_retained(self):
        stop = threading.Event()
        controller = l3.Lite3Controller.__new__(l3.Lite3Controller)
        controller._stop = stop
        controller.rx = _FakeReceiver(
            [_speaker_status_packet(1), _speaker_status_packet(0)], stop)
        controller._speaker_lock = threading.Lock()
        controller._speaker_enabled = None

        controller._rx_loop()

        self.assertIs(controller.get_speaker_state(), False)


if __name__ == "__main__":
    unittest.main()
