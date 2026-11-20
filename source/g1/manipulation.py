"""Arm and hand skills: move a hand to a spot smoothly, open or close a hand.

Built only on the robot's interfaces (g1.interfaces), so they work with any
backend. A move goes in joint space from the current angles to the IK
solution (g1.kinematics), each joint following the same smooth S-curve
(smoothstep: starts and ends at zero speed), paced so that no joint goes
faster than MAX_JOINT_SPEED. Unitree's own arm examples also move between
poses by interpolating joint targets over time (g1_arm7_sdk_dds_example.py).
"""

import numpy as np

from . import joints
from .interfaces import Hands, State, UpperBody
from .kinematics import ArmIK, hand_pose

MAX_JOINT_SPEED = 1.0  # rad/s: the fastest any arm joint moves during a move
GRIP_TIME = 1.0  # s to open or close a hand fully


def smoothstep(a):
    """0 -> 1 with zero slope at both ends."""
    a = np.clip(a, 0.0, 1.0)
    return a * a * (3 - 2 * a)


class Manipulator:
    """Moves the arms and hands of a robot that has UpperBody, Hands and State."""

    def __init__(self, robot):
        self.robot: UpperBody | Hands | State = robot
        self.ik = {side: ArmIK(side) for side in ("left", "right")}
        self.targets = robot.joint_positions(joints.UPPER_BODY)  # what the upper body is told
        self.hands = {side: robot.joint_positions(joints.HAND[side]) for side in ("left", "right")}
        self._move = None  # (start time, duration, from, to) of the upper body
        self._grip = {}  # side -> (start time, from, to)

    # ---- starting skills ----

    def reach(self, side, position, rotation=None) -> bool:
        """Move the `side` hand's palm to `position` (pelvis frame), turned like
        `rotation`. Returns False (and does nothing) if it can't be reached."""
        return self.reach_all({side: (position, rotation)})

    def reach_all(self, goals) -> bool:
        """Move several hands at once: goals = {side: (palm position, rotation)}.
        Returns False (and does nothing) if any of them can't be reached."""
        goal = self.targets.copy()
        waist = self.targets[: len(joints.WAIST)]  # the arms are solved with the waist where it is told to be
        for side, (position, rotation) in goals.items():
            arm = slice(*self._arm_slice(side))
            angles, reached = self.ik[side].solve(position, rotation, start=self.targets[arm], waist_angles=waist)
            if not reached:
                return False
            goal[arm] = angles
        self.move_upper_body(goal)
        return True

    def move_upper_body(self, goal):
        """Move the 17 upper-body joints (waist, arms) smoothly to `goal`."""
        largest = float(np.max(np.abs(np.asarray(goal) - self.targets)))
        # smoothstep's top speed is 1.5x its average: pace the largest joint at MAX_JOINT_SPEED
        duration = max(1.5 * largest / MAX_JOINT_SPEED, 0.1)
        self._move = (self.robot.time(), duration, self.targets.copy(), np.asarray(goal, float))

    def grip(self, side, closed: bool):
        """Open or close a hand over GRIP_TIME."""
        self._grip[side] = (self.robot.time(), self.hands[side].copy(), hand_pose(side, closed))

    # ---- every control step ----

    def update(self) -> None:
        """Send this step's targets to the robot."""
        now = self.robot.time()
        if self._move is not None:
            start, duration, a, b = self._move
            self.targets = a + smoothstep((now - start) / duration) * (b - a)
            if now - start >= duration:
                self._move = None
        self.robot.set_upper_body_targets(self.targets)
        for side, (start, a, b) in list(self._grip.items()):
            self.hands[side] = a + smoothstep((now - start) / GRIP_TIME) * (b - a)
            if now - start >= GRIP_TIME:
                del self._grip[side]
        for side in ("left", "right"):
            self.robot.set_hand_targets(side, self.hands[side])

    def busy(self) -> bool:
        return self._move is not None or bool(self._grip)

    @staticmethod
    def _arm_slice(side):
        start = len(joints.WAIST) + (0 if side == "left" else len(joints.LEFT_ARM))
        return start, start + len(joints.ARM[side])
