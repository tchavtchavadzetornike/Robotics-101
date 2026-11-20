"""Arm inverse kinematics on Unitree's G1 model."""

import numpy as np
from g1.kinematics import ArmIK, load_model

ARM_ANGLES = np.array([0.3, -0.2, 0.1, 0.8, 0.0, 0.0, 0.0])  # a comfortable right-arm pose


def test_the_model_is_parsed_once():
    assert load_model() is load_model()


def test_a_reachable_target_is_reached():
    ik = ArmIK("right")
    target, _ = ik.palm_pose(ARM_ANGLES)
    angles, reached = ik.solve(target, start=np.zeros(7))
    assert reached
    solved, _ = ik.palm_pose(angles)
    assert np.linalg.norm(solved - target) < 0.01


def test_a_reachable_target_with_a_hand_orientation_is_reached():
    ik = ArmIK("right")
    target, rotation = ik.palm_pose(ARM_ANGLES)
    angles, reached = ik.solve(target, rotation, start=np.zeros(7))
    assert reached
    solved, solved_rotation = ik.palm_pose(angles)
    assert np.linalg.norm(solved - target) < 0.01
    assert np.linalg.norm(solved_rotation - rotation) < 0.05


def test_a_target_out_of_reach_is_not_reached_and_stays_within_the_joint_limits():
    ik = ArmIK("right")
    angles, reached = ik.solve(np.array([5.0, 0.0, 0.0]), start=np.zeros(7))
    assert not reached
    assert np.all(angles >= ik.lower) and np.all(angles <= ik.upper)
