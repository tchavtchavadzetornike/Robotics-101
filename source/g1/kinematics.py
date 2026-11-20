"""Arm inverse kinematics: where the hand should be -> the 7 arm joint angles.

Unitree's SDK takes joint angles for the arms (rt/arm_sdk), not hand
positions, so turning a hand target into angles is the application's job.
Unitree's own teleoperation software does it the same way: Pinocchio on
Unitree's G1 model (xr_teleoperate, teleop/robot_control/robot_arm_ik.py).
Here: damped least squares (Buss, "Introduction to Inverse Kinematics with
Jacobian Transpose, Pseudoinverse and Damped Least Squares methods", 2004),
the standard iterative method; it stays well-behaved near the arm's reach limit.

The model is Unitree's g1_29dof_with_hand_rev_1_0.urdf (unitree_ros,
BSD-3-Clause, in g1/assets/). Its root is the pelvis, so targets are in the
pelvis frame.
"""

from functools import cache
from pathlib import Path

import numpy as np
import pinocchio as pin

from . import joints

MODEL = Path(__file__).resolve().parent / "assets" / "g1_29dof_with_hand_rev_1_0.urdf"
# The point that grasps: the palm of each hand (frames of Unitree's model)
PALM = {"left": "left_hand_palm_link", "right": "right_hand_palm_link"}

DAMPING = 0.05  # the "lambda" of damped least squares: keeps steps small near singular poses
ROTATION_WEIGHT = 0.3  # 1 rad of hand-orientation error counts like 0.3 m of position error
MAX_ITERATIONS = 200
TOLERANCE = 0.005  # m: close enough (and the rotation error times ROTATION_WEIGHT)


@cache
def load_model():
    """Unitree's G1 model, parsed once. Read-only: every user makes its own Data."""
    return pin.buildModelFromUrdf(str(MODEL))


class ArmIK:
    def __init__(self, side: str):
        self.side = side
        self.model = load_model()
        self.data = self.model.createData()
        self.palm = self.model.getFrameId(PALM[side])
        self.arm_q = [self.model.joints[self.model.getJointId(n)].idx_q for n in joints.ARM[side]]
        self.arm_v = [self.model.joints[self.model.getJointId(n)].idx_v for n in joints.ARM[side]]
        self.lower = self.model.lowerPositionLimit[self.arm_q]
        self.upper = self.model.upperPositionLimit[self.arm_q]
        self.q = pin.neutral(self.model)

    def palm_pose(self, arm_angles, waist_angles=(0.0, 0.0, 0.0)):
        """Where the palm is for these arm angles: position (3) and rotation (3x3), pelvis frame."""
        self._set(arm_angles, waist_angles)
        pin.framesForwardKinematics(self.model, self.data, self.q)
        frame = self.data.oMf[self.palm]
        return frame.translation.copy(), frame.rotation.copy()

    def solve(self, position, rotation=None, start=None, waist_angles=(0.0, 0.0, 0.0)):
        """Arm angles that put the palm at `position` (pelvis frame), with the
        palm turned like `rotation` (3x3) if given.

        start: the arm angles to start from (the current ones), so the result
        is the nearby solution. Returns (angles, reached).
        """
        angles = np.clip(np.zeros(7) if start is None else np.asarray(start, float), self.lower, self.upper)
        target = pin.SE3(np.eye(3) if rotation is None else np.asarray(rotation), np.asarray(position, float))
        for _ in range(MAX_ITERATIONS):
            self._set(angles, waist_angles)
            pin.computeJointJacobians(self.model, self.data, self.q)
            pin.framesForwardKinematics(self.model, self.data, self.q)
            current = self.data.oMf[self.palm]
            error_position = target.translation - current.translation
            # orientation error as a rotation vector, in the world-aligned frame
            error_rotation = current.rotation @ pin.log3(current.rotation.T @ target.rotation)
            jacobian = pin.getFrameJacobian(self.model, self.data, self.palm, pin.LOCAL_WORLD_ALIGNED)[:, self.arm_v]
            if rotation is None:
                error, jacobian = error_position, jacobian[:3]
            else:
                error = np.concatenate([error_position, ROTATION_WEIGHT * error_rotation])
                jacobian = np.vstack([jacobian[:3], ROTATION_WEIGHT * jacobian[3:]])
            if np.linalg.norm(error) < TOLERANCE:
                return angles, True
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + DAMPING**2 * np.eye(len(error)), error)
            angles = np.clip(angles + step, self.lower, self.upper)
        return angles, False

    def _set(self, arm_angles, waist_angles):
        self.q[self.arm_q] = arm_angles
        for name, angle in zip(joints.WAIST, waist_angles):
            self.q[self.model.joints[self.model.getJointId(name)].idx_q] = angle


def grasp_point(side: str) -> np.ndarray:
    """Where a held object's center sits, in the palm frame: the middle of the
    gap between the thumb tip and the two fingers, with the hand open (from
    Unitree's model). The palm is placed so that this point is on the object."""
    model = load_model()
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    point = {
        name: data.oMf[model.getFrameId(f"{side}_hand_{name}_link")].translation
        for name in ("palm", "thumb_2", "index_1", "middle_1")
    }
    fingers = (point["index_1"] + point["middle_1"]) / 2
    middle = (point["thumb_2"] + fingers) / 2
    palm = data.oMf[model.getFrameId(PALM[side])]
    return palm.rotation.T @ (middle - palm.translation)


def hand_pose(side: str, closed: bool) -> np.ndarray:
    """Dex3 finger angles: open (all fingers straight, every joint at 0) or
    closed (every joint at its limit in the closing direction, from Unitree's
    model; a held object stops the fingers earlier, and the finger motors'
    torque limits set how hard they squeeze)."""
    if not closed:
        return np.zeros(7)
    model = load_model()
    angles = []
    for name in joints.HAND[side]:
        i = model.joints[model.getJointId(name)].idx_q
        low, high = model.lowerPositionLimit[i], model.upperPositionLimit[i]
        if "thumb_0" in name:
            angles.append(0.0)  # the thumb's base swivel stays facing the fingers
        else:
            angles.append(high if abs(high) >= abs(low) else low)  # curling = the larger side
    return np.array(angles)
