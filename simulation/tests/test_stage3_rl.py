import unittest

import numpy as np

from stage3_mujoco_rl import (
    OBS_DIM,
    build_observation,
    default_joint_positions,
    pd_torque_control,
    projected_gravity,
)


class Stage3RLHelpersTest(unittest.TestCase):
    def test_projected_gravity_at_identity_points_down(self):
        gravity = projected_gravity([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(gravity, [0.0, 0.0, -1.0], atol=1e-12)

    def test_projected_gravity_for_x_rotation(self):
        # A +90-degree body-to-world rotation about x.
        quaternion = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
        gravity = projected_gravity(quaternion)
        np.testing.assert_allclose(gravity, [0.0, -1.0, 0.0], atol=1e-12)

    def test_default_joint_positions_match_deploy(self):
        expected = np.tile([0.0, -0.65, 1.3], 4)
        np.testing.assert_allclose(default_joint_positions(), expected)

    def test_pd_torque_direction_and_magnitude(self):
        torque = pd_torque_control(
            target_qpos=[1.0, -1.0],
            qpos=[0.5, 0.0],
            qvel=[2.0, -4.0],
        )
        # tau = 30 * (target - q) - 1 * qvel
        np.testing.assert_allclose(torque, [13.0, -26.0])

    def test_pd_torque_clips_to_actuator_limits(self):
        ctrlrange = np.tile([-5.0, 10.0], (2, 1))
        torque = pd_torque_control(
            target_qpos=[1.0, -1.0],
            qpos=[0.5, 0.0],
            qvel=[2.0, -4.0],
            ctrlrange=ctrlrange,
        )
        np.testing.assert_allclose(torque, [10.0, -5.0])

    def test_observation_dimension_order_and_scaling(self):
        base_angular_velocity = np.array([1.0, 2.0, 3.0])
        gravity = np.array([0.0, 0.0, -1.0])
        command = np.array([0.1, 0.2, 0.3])
        qpos = np.arange(12, dtype=float)
        qvel = np.arange(12, 24, dtype=float)
        last_action = np.arange(24, 36, dtype=float)

        obs = build_observation(
            base_angular_velocity, gravity, command, qpos, qvel, last_action,
        )
        expected = np.concatenate([
            base_angular_velocity * 0.25,
            gravity,
            command,
            qpos - default_joint_positions(),
            qvel * 0.05,
            last_action,
        ]).astype(np.float32)

        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_array_equal(obs, expected)
        np.testing.assert_array_equal(obs[0:3], [0.25, 0.5, 0.75])
        np.testing.assert_array_equal(obs[3:6], [0.0, 0.0, -1.0])
        np.testing.assert_array_equal(obs[6:9], command.astype(np.float32))
        np.testing.assert_array_equal(
            obs[9:21], (qpos - default_joint_positions()).astype(np.float32),
        )
        np.testing.assert_array_equal(obs[21:33], (qvel * 0.05).astype(np.float32))
        np.testing.assert_array_equal(obs[33:45], last_action.astype(np.float32))


if __name__ == '__main__':
    unittest.main()
