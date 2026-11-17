"""Turning points between the room (world frame) and the robot (body frame).

world: x, y on the floor, z up, fixed to the room.
body:  the pelvis frame turned only by the heading, x ahead, y left, z up.
       It ignores the small tilt of the pelvis, like mission.Mission.to_body.

to_body and to_world undo each other:
    to_world(to_body(p, position, heading), position, heading) == p
"""

import numpy as np


def to_body(point, position, heading):
    """World point -> body frame: subtract the robot position, turn by -heading."""
    d = np.asarray(point, dtype=float) - np.asarray(position, dtype=float)
    c, s = np.cos(heading), np.sin(heading)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])


def to_world(point, position, heading):
    """Body point -> world frame: turn by +heading, add the robot position."""
    p = np.asarray(point, dtype=float)
    c, s = np.cos(heading), np.sin(heading)
    turned = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])
    return turned + np.asarray(position, dtype=float)


def tilt_of(quat):
    """How far the body leans from upright (rad), any direction, from a (w, x, y, z) quaternion.

    It is the angle between the body's z axis and the room's z axis. 0 = upright.
    """
    _, x, y, _ = quat
    return float(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0)))
