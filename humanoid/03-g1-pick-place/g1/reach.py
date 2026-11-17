"""Where the robot can stand to grasp an object, and where it can put one down.

A table has four sides. Standing at a side, facing the table, the right hand
reaches an object best at GRASP_SPOT (30 cm ahead, 12 cm right of the
pelvis, from mission.py). The further the object is from the table's edge
(its depth), the closer the robot must stand, until its legs hit the table.
MAX_DEPTH is that limit, measured (tell-the-robot/PHASE-4.md).

The open hand comes in from the object's right side (seen from the robot),
closes around it, and later pulls away the same way. Another object in the
area the hand sweeps is in the way: HAND_SWEEP.

Every function here returns options with a reason when there are none, so a
skill can say "failed: the red can is 25 cm from every table edge, the hand
reaches 14 cm" instead of trying and falling.
"""

from dataclasses import dataclass

import numpy as np

from g1 import scenes
from g1.mission import APPROACH, GRASP_SPOT

MIN_DEPTH = 0.05  # m: closer to the edge, the fingers close past the table edge
MAX_DEPTH = 0.17  # m: measured: 18 cm works, 20 cm the legs hit the table; 1 cm margin
HAND_SWEEP = (((-0.16, 0.12), (-APPROACH - 0.04, 0.04)),)  # m around the spot, (ahead, left) in the robot's frame:
# fingertips 12 cm ahead; 16 cm behind: after lifting, the object goes back into the carry pose
# (10 cm behind, 6 cm right of the spot); from 4 cm past the entry point (8 cm right) to 4 cm left
PLACE_SWEEP = (  # placing, measured: the right hand's geoms at every step (spikes/knockover_check.py)
    ((-0.80, -0.04), (-0.24, -0.04)),  # carrying it in from the approach point: wrist and fingers up to 24 cm right
    ((-0.24, 0.12), (-0.14, 0.08)),  # over the spot, down, open, pull straight back, raise
    ((-0.50, -0.13), (-0.33, -0.01)),  # fingers turned in, tucked, stepping back
)
OBJECT_RADIUS = 0.035  # m: the widest object (a can, or a box's corner seen side-on)
PLACE_DEPTH = 0.10  # m from the edge, where objects are put down
PLACE_GAP = 0.13  # m: a put-down object keeps this distance to every other object
NEXT_TO = (0.13, 0.16)  # m between centers for "next to": two rings, the outer one when the hand needs room
MARK = np.array(scenes.PLACE_MARK)


@dataclass
class Option:
    table: str
    side: str
    heading: float  # rad, robot facing the table
    stand: np.ndarray  # (x, y, heading) where the pelvis stands
    spot: np.ndarray  # (x, y) of the object
    depth: float  # m from the edge


def sides(table):
    """(name, heading facing the table, inward unit vector, edge point) for each side."""
    (cx, cy), (hx, hy) = scenes.TABLES[table]
    return [
        ("-x", 0.0, np.array([1.0, 0.0]), np.array([cx - hx, cy])),
        ("+x", np.pi, np.array([-1.0, 0.0]), np.array([cx + hx, cy])),
        ("-y", np.pi / 2, np.array([0.0, 1.0]), np.array([cx, cy - hy])),
        ("+y", -np.pi / 2, np.array([0.0, -1.0]), np.array([cx, cy + hy])),
    ]


def table_of(position):
    for name, ((cx, cy), (hx, hy)) in scenes.TABLES.items():
        if abs(position[0] - cx) <= hx and abs(position[1] - cy) <= hy:
            return name
    return None


def on_table(spot, table, margin=0.05):
    (cx, cy), (hx, hy) = scenes.TABLES[table]
    return abs(spot[0] - cx) <= hx - margin and abs(spot[1] - cy) <= hy - margin


def stand_pose(spot, heading):
    """Where the pelvis stands so that `spot` is at GRASP_SPOT (as Mission.stand_pose_for)."""
    c, s = np.cos(heading), np.sin(heading)
    dx = c * GRASP_SPOT[0] - s * GRASP_SPOT[1]
    dy = s * GRASP_SPOT[0] + c * GRASP_SPOT[1]
    return np.array([spot[0] - dx, spot[1] - dy, heading])


def hand_entry(spot, heading):
    """Where the open hand waits before closing in: APPROACH to the object's right."""
    c, s = np.cos(heading), np.sin(heading)
    return np.asarray(spot[:2]) + APPROACH * np.array([s, -c])


def blocked(spot, heading, others, sweep=None):
    """The other object in the hand's way, or None.

    Not just the point where the hand comes in: the whole area the hand sweeps
    while it comes in, closes and pulls away, as rectangles around the spot in
    the robot's frame (HAND_SWEEP, PLACE_SWEEP), grown by an object's radius.
    Phase 4 found why: the fingers reach 10 cm past the grasp point and pushed
    a box 6 cm. Phase 5 found the carrying wrist 24 cm right of the spot.
    """
    c, s = np.cos(heading), np.sin(heading)
    for other in others:
        d = np.asarray(other[:2]) - np.asarray(spot[:2])
        ahead, left = c * d[0] + s * d[1], -s * d[0] + c * d[1]
        for (ahead_min, ahead_max), (left_min, left_max) in HAND_SWEEP if sweep is None else sweep:
            if ahead_min - OBJECT_RADIUS < ahead < ahead_max + OBJECT_RADIUS and left_min - OBJECT_RADIUS < left < left_max + OBJECT_RADIUS:
                return other
    return None


def grasp_options(position, others=()):
    """Ways to grasp an object at `position`: (options sorted shallowest first, reason if none)."""
    table = table_of(position)
    if table is None:
        return [], "it is not on a table"
    options, reasons = [], []
    for side, heading, inward, edge in sides(table):
        depth = float(np.dot(np.asarray(position[:2]) - edge, inward))
        if depth > MAX_DEPTH:
            reasons.append(f"{100 * depth:.0f} cm from the {side} edge")
            continue
        if depth < MIN_DEPTH:
            reasons.append(f"only {100 * depth:.0f} cm from the {side} edge")
            continue
        if blocked(position, heading, others) is not None:
            reasons.append(f"from the {side} side another object is where the hand comes in")
            continue
        options.append(Option(table, side, heading, stand_pose(position, heading), np.asarray(position[:2]), depth))
    options.sort(key=lambda o: o.depth)
    if options:
        return options, ""
    return [], f"can't reach it on {table}: " + "; ".join(reasons) + f" (the hand reaches {100 * MAX_DEPTH:.0f} cm deep)"


def free_spot(spot, others, gap=PLACE_GAP):
    return all(np.hypot(*(np.asarray(o[:2]) - spot)) >= gap for o in others)


def place_options(target, others=(), anchor=None):
    """Where to put the held object down. target: "table_a", "table_b", "mark", or ("next_to", position).

    others: positions of every other object. Returns (options, reason if none).
    """
    candidates = []
    if target == "mark":
        candidates = [(scenes.PLACE_MARK, "table_b")]
    elif isinstance(target, tuple) and target[0] == "next_to":
        center = np.asarray(target[1][:2])
        table = table_of(center)
        if table is None:
            return [], "that object is not on a table"
        for radius in NEXT_TO:
            for angle in np.radians(np.arange(0, 360, 30)):
                candidates.append((center + radius * np.array([np.cos(angle), np.sin(angle)]), table))
        others = [o for o in others if np.hypot(*(np.asarray(o[:2]) - center)) > 0.01]
        others = list(others) + [center]
    elif target in scenes.TABLES:
        for side, heading, inward, edge in sides(target):
            along = np.array([-inward[1], inward[0]])
            for offset in np.arange(-0.35, 0.36, 0.05):
                candidates.append((edge + PLACE_DEPTH * inward + offset * along, target))
    else:
        return [], f"unknown place {target!r}"

    options = []
    for spot, table in candidates:
        spot = np.asarray(spot, float)
        if not on_table(spot, table):
            continue
        if not free_spot(spot, others):
            continue
        for side, heading, inward, edge in sides(table):
            depth = float(np.dot(spot - edge, inward))
            if MIN_DEPTH <= depth <= MAX_DEPTH and blocked(spot, heading, others, PLACE_SWEEP) is None:
                options.append(Option(table, side, heading, stand_pose(spot, heading), spot, depth))
    if anchor is not None:
        options.sort(key=lambda o: np.hypot(*(o.stand[:2] - np.asarray(anchor[:2]))))
    if options:
        return options, ""
    return [], f"no free spot within reach for {target if isinstance(target, str) else 'next to that object'}"
