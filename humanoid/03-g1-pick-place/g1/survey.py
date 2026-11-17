"""Look around: walk to a spot that sees both tables, face each one, stand still and look.

Why a spot and not just turning on the start position: the head camera looks
11 to 69 degrees down (40 degree tilt, 58 degree field of view). An object's
top, about 1 m high, is only in the image within 1.39 m of the head. From the
start, most of table A is further than that: the detector finds objects there,
but can't see their tops, so it can't tell a bottle from a can. From
SURVEY_SPOT every place an object can stand on either table is 0.56 to 1.35 m away.

Why table A gets a second look from SECOND_SPOT: from one place, an object
can stand exactly behind another (seed 19: bearings 53.4 and 53.9 degrees,
the green can fully hidden behind the blue bottle). No detector finds what
the camera can't see. The two views of table A are about 75 degrees apart,
so an object hidden in both is unlikely.

A plan is a list of steps: ("walk", (x, y, heading)) or ("look", seconds).
Looking means standing still: the detector needs about 0.3 s per image, and
the robot sways less while standing. Phase 4 turns this into the look_around skill.
"""

import numpy as np

from g1 import navigation

SURVEY_SPOT = (0.95, -0.7)
FACE_TABLE_A = 0.71  # rad, toward table A's middle (1.6, 0)
FACE_TABLE_B = -1.47  # rad, toward table B's middle (1.0, -1.7)
SECOND_SPOT = (0.95, 0.45)  # table A from its other side, 0.44 to 1.18 m from every spot on it
SECOND_FACE_A = -0.61  # rad
LOOK_TIME = 2.0  # s per table: about 6 detector images
STOP_WAIT = 0.5  # s: after stopping, wait for a fresh "standing"


def look_around_plan():
    x, y = SURVEY_SPOT
    return [
        ("walk", (*SECOND_SPOT, SECOND_FACE_A)),
        ("look", LOOK_TIME),
        ("walk", (x, y, FACE_TABLE_A)),
        ("look", LOOK_TIME),
        ("walk", (x, y, FACE_TABLE_B)),
        ("look", LOOK_TIME),
    ]


class Plan:
    def __init__(self, steps):
        self.steps = list(steps)
        self.index = -1
        self.navigator = navigation.Navigator()
        self.step_start = 0.0
        self.stopped_at = None
        self.time = 0.0
        self.finished = False

    @property
    def looking(self):
        """True while standing still at a look step: the time to trust what the camera sees."""
        return 0 <= self.index < len(self.steps) and self.steps[self.index][0] == "look"

    @property
    def label(self):
        if not 0 <= self.index < len(self.steps):
            return ""
        kind, value = self.steps[self.index]
        return f"walk to {np.round(value, 2).tolist()}" if kind == "walk" else f"look for {value:.0f} s"

    def _next(self):
        self.index += 1
        self.step_start = self.time
        self.stopped_at = None
        self.navigator.stop()
        if self.index >= len(self.steps):
            self.finished = True

    def update(self, time, pose, standing):
        """pose: (x, y, heading). Returns the walking command (forward, left, turn)."""
        dt = time - self.time
        self.time = time
        stop = (0.0, 0.0, 0.0)
        if self.index < 0:
            self._next()
        if self.finished:
            return stop
        kind, value = self.steps[self.index]
        if kind == "look":
            if time - self.step_start >= value:
                self._next()
            return stop
        if self.stopped_at is None:
            cmd_vel, arrived = self.navigator.update(pose, value, dt)
            if not arrived:
                return cmd_vel
            self.stopped_at = time
        if not standing or time - self.stopped_at < STOP_WAIT:
            return stop
        if not navigation.close_to(pose, value):
            self.stopped_at = None
            return stop
        self._next()
        return stop
