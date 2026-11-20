"""Speed and separation monitoring: never walk faster than the robot can stop.

The planner chooses where to go; this layer, between the planner and the
robot, only ever slows the command down or stops it. It is the same split as
ROS 2 Nav2's controller and its Collision Monitor, and the rule is the speed
and separation monitoring of ISO/TS 15066 (collaborative robots):

    the free space to a person must stay at least
    S = v_h (T_r + T_s)  +  v_r T_r  +  B  +  Z

    v_h  the person's speed toward the robot (measured by the tracker)
    v_r  the robot's speed toward the person
    T_r  the robot's reaction time: from a lidar scan to a new command
    T_s  the robot's stopping time,    T_s = v_r / a
    B    the robot's braking distance, B = v_r^2 / (2 a)
    Z    measurement uncertainty

For each person (and each raw lidar point, which may be something the tracker
has not recognised, counted as standing still) it solves this for the
largest v_r it allows, and scales the whole command down until the robot's
speed toward everyone is within that. Plain Python + NumPy.
"""

import math

import numpy as np

from .config import PERSON_RADIUS
from .planner import MAX_DECEL, PLAN_DT, ROBOT_RADIUS

# The robot reacts to what it saw up to one lidar turn ago (10 Hz lidar) and
# plans every PLAN_DT: the worst case from seeing to acting is both.
REACTION_TIME = 0.1 + PLAN_DT  # s
# Where people are is known to about 3 cm (the tracker, measured in the
# lesson's tests) and lidar points to about 1 cm; keep 2 standard deviations.
UNCERTAINTY = 0.06  # m
STOP_DISTANCE = 0.05  # m of free space: closer than this, nothing moves toward it
# What the robot knows about its surroundings must be recent: three lidar scans.
MAX_AGE = 0.3  # s of robot time


def sensors_fresh(now, *sensor_times):
    """Every sensor has reported, and none of them longer than MAX_AGE ago
    (`now` and the times are robot time, in seconds)."""
    return all(t is not None and now - t <= MAX_AGE for t in sensor_times)


def allowed_speed(gap, human_speed, decel=MAX_DECEL, reaction=REACTION_TIME):
    """The largest robot speed toward something `gap` meters of free space away
    that comes toward the robot at `human_speed`, from
    gap = v_h (T_r + v/a) + v T_r + v^2 / (2a), solved for v."""
    room = gap - human_speed * reaction
    b = reaction + human_speed / decel
    speed = np.where(room <= 0.0, 0.0, decel * (-b + np.sqrt(b * b + 2.0 * np.maximum(room, 0.0) / decel)))
    return float(speed) if speed.ndim == 0 else speed


class SpeedSeparationMonitor:
    def __init__(self, decel=MAX_DECEL):
        """decel: how fast the robot can brake (m/s^2), the a in the formula."""
        self.decel = decel
        self.limited = False  # did the last command get slowed down?

    def limit(self, cmd, pose, tracks, points=None):
        """The command, slowed down if it isn't safe.

        cmd: (forward, left, turn) in the robot frame. pose: (x, y, heading).
        tracks: rows (x, y, vx, vy), world frame. points: optional raw lidar
        points (x, y), world frame. Turning on the spot is always allowed.
        """
        forward, left, turn = cmd
        x, y, heading = pose
        c, s = math.cos(heading), math.sin(heading)
        velocity = np.array([c * forward - s * left, s * forward + c * left])  # world frame
        speed = float(np.hypot(*velocity))
        self.limited = False
        if speed < 1e-6:
            return (0.0, 0.0, turn)

        tracks = np.asarray(tracks, dtype=float).reshape(-1, 4)
        positions, velocities = tracks[:, :2], tracks[:, 2:4]
        radii = np.full(len(tracks), PERSON_RADIUS)
        if points is not None:
            points = np.asarray(points, dtype=float).reshape(-1, 2)
            positions = np.vstack([positions, points])
            velocities = np.vstack([velocities, np.zeros_like(points)])
            radii = np.concatenate([radii, np.zeros(len(points))])
        if len(positions) == 0:
            return (forward, left, turn)

        offset = positions - (x, y)
        distance = np.hypot(offset[:, 0], offset[:, 1])
        if np.any(distance < 1e-6):
            return (0.0, 0.0, turn)
        toward = offset / distance[:, None]  # unit vectors from the robot to each obstacle
        robot_closing = toward @ velocity  # walking away from one (negative): always fine
        human_closing = np.maximum(0.0, -np.einsum("ij,ij->i", velocities, toward))
        gap = distance - ROBOT_RADIUS - radii - UNCERTAINTY
        allowed = np.where(gap < STOP_DISTANCE, 0.0, allowed_speed(gap, human_closing, self.decel))
        too_fast = robot_closing > allowed
        if not np.any(too_fast):
            return (forward, left, turn)
        self.limited = True
        scale = float(np.min(allowed[too_fast] / robot_closing[too_fast]))
        return (forward * scale, left * scale, turn)
