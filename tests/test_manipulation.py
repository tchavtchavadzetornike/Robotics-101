"""Arm and hand moves: timing, joint speed, reachability."""

import numpy as np
from fakes import FakeG1
from g1 import joints
from g1.kinematics import ArmIK, hand_pose
from g1.manipulation import GRIP_TIME, MAX_JOINT_SPEED, Manipulator

MAX_STEPS = 1000


def run_until_still(robot, hands):
    """Step until the skills are done. Returns the fastest any upper-body joint moved (rad/s)."""
    fastest, previous = 0.0, hands.targets.copy()
    for _ in range(MAX_STEPS):
        hands.update()
        robot.step()
        fastest = max(fastest, float(np.abs(hands.targets - previous).max()) / robot.dt)
        previous = hands.targets.copy()
        if not hands.busy():
            break
    return fastest


def test_a_hand_move_ends_at_the_target_without_exceeding_the_joint_speed():
    robot = FakeG1()
    hands = Manipulator(robot)
    ik = ArmIK("right")
    palm, rotation = ik.palm_pose(np.zeros(7))
    target = palm + np.array([0.0, 0.0, 0.08])
    assert hands.reach("right", target, rotation)
    assert hands.busy()
    fastest = run_until_still(robot, hands)
    assert not hands.busy()
    assert fastest <= MAX_JOINT_SPEED + 1e-9
    reached, _ = ik.palm_pose(robot.joint_positions(joints.RIGHT_ARM), robot.joint_positions(joints.WAIST))
    assert np.linalg.norm(reached - target) < 0.01


def test_a_target_out_of_reach_does_nothing():
    robot = FakeG1()
    hands = Manipulator(robot)
    assert not hands.reach("right", np.array([5.0, 0.0, 0.0]))
    assert not hands.busy()


def test_closing_the_hand_takes_the_grip_time():
    robot = FakeG1()
    hands = Manipulator(robot)
    hands.grip("right", closed=True)
    steps = 0
    while hands.busy() and steps < MAX_STEPS:
        hands.update()
        robot.step()
        steps += 1
    assert abs(steps * robot.dt - GRIP_TIME) <= 2 * robot.dt
    assert np.allclose(robot.joint_positions(joints.RIGHT_HAND), hand_pose("right", closed=True))
