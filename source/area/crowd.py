"""The people on the square: each walks to a goal and steps around the others.

This is the "social force model" (Helbing and Molnar, 1995), the standard
model for simulated crowds. Every person feels two pushes:
  1. toward their goal: they want to walk at their own favourite speed,
  2. away from every other person who is close and in front of them.
The pushes are added up as an acceleration, and the person's velocity and
position follow from it, step by step.

Everything here is 2D (x, y on the ground) and plain NumPy; area.terms.MoveCrowd
moves the people in Isaac Lab to where this model puts them.
"""

import numpy as np

# ---- how people walk: measured values from the pedestrian literature ----
MEAN_SPEED = 1.34  # m/s: average walking speed (Helbing and Molnar 1995)
SPEED_SPREAD = 0.26  # m/s: standard deviation between people (same paper)
RELAX_TIME = 0.5  # s: how quickly a person gets back to their favourite speed
RADIUS = 0.22  # m: a person seen from above, about 44 cm shoulder to shoulder
MAX_SPEED_FACTOR = 1.3  # nobody walks faster than 1.3 x their favourite speed

# ---- how people avoid each other ----
# push = PUSH * exp(-gap / PUSH_RANGE), in m/s^2, where gap is the free space
# between two bodies. At 1 m of free space: 12 * exp(-1 / 0.3) = 0.43 m/s^2,
# a gentle drift aside. At touching distance: 12 m/s^2, a quick step aside.
# Found by trying values on 16 people for 90 s: with these two, nobody ever
# touches anybody, and people still average 1.1 m/s.
PUSH = 12.0  # m/s^2
PUSH_RANGE = 0.3  # m
# People mostly react to what is in front of them. Someone straight ahead
# counts fully, someone straight behind counts only this much:
BEHIND_WEIGHT = 0.3
# Two people walking straight at each other would only brake. Real people
# pass on one side, so part of the push is turned sideways (to the right):
SIDESTEP = 0.6
# People step around the robot like around another person, and give it a bit
# more room than its real size (0.30 m), as people do with a machine they
# don't know. The robot still has to do most of the avoiding: it is half as
# fast as the people.
ROBOT_WEIGHT = 1.0
ROBOT_SPACE = 0.40  # m: the radius people treat the robot as having

GOAL_REACHED = 0.6  # m: close enough to the goal, pick the next one
MAX_PLACING_TRIES = 1000  # random spots tried for one person before the square counts as full


class Crowd:
    def __init__(self, count, half_size, seed, keep_clear=None):
        """count people on a square from -half_size to +half_size.

        keep_clear: (x, y, radius) where nobody is placed at the start
        (the spot where the robot stands).
        """
        self.rng = np.random.default_rng(seed)
        self.half_size = half_size
        self.count = count
        self.speed = np.clip(
            self.rng.normal(MEAN_SPEED, SPEED_SPREAD, count), 0.8, 1.9
        )  # each person's favourite speed
        self.position = np.zeros((count, 2))
        for i in range(count):
            self.position[i] = self._free_spot(i, keep_clear)
        self.side = np.array([self._nearest_side(p) for p in self.position])
        self.goal = np.zeros((count, 2))
        for i in range(count):
            self._new_goal(i)
        direction = self.goal - self.position
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        self.velocity = direction * self.speed[:, None]

    # ---- where people start and where they go ----

    def _free_spot(self, placed, keep_clear):
        """A random spot at least 1.5 m from everyone placed so far."""
        for _ in range(MAX_PLACING_TRIES):
            spot = self.rng.uniform(-self.half_size, self.half_size, 2)
            if keep_clear is not None:
                if np.hypot(*(spot - keep_clear[:2])) < keep_clear[2]:
                    continue
            if placed == 0:
                return spot
            if np.min(np.linalg.norm(self.position[:placed] - spot, axis=1)) > 1.5:
                return spot
        raise ValueError(
            f"no free spot for person {placed + 1} of {self.count} on a {2 * self.half_size:g} m square: "
            "too many people for the space"
        )

    def _nearest_side(self, p):
        """Which side of the square a point is closest to: 0 E, 1 N, 2 W, 3 S."""
        h = self.half_size
        return int(np.argmin([h - p[0], h - p[1], h + p[0], h + p[1]]))

    def _point_on_side(self, side):
        h = self.half_size
        along = self.rng.uniform(-0.8 * h, 0.8 * h)
        return np.array([(h, along), (along, h), (-h, along), (along, -h)][side])

    def _new_goal(self, i):
        """Person i heads for a random point on another side of the square."""
        others = [s for s in range(4) if s != self.side[i]]
        self.side[i] = self.rng.choice(others)
        self.goal[i] = self._point_on_side(self.side[i])

    # ---- the model ----

    def _pushes(self, others, other_radius, weight):
        """Acceleration on every person away from a set of other bodies (m/s^2).

        others: positions, shape (M, 2). Returns shape (people, 2): for each
        person, the pushes from all M bodies added up.
        """
        away = self.position[:, None, :] - others[None, :, :]  # (people, M, 2)
        distance = np.linalg.norm(away, axis=2)
        itself = distance < 1e-6  # a person doesn't push themselves
        away = away / np.where(itself, 1.0, distance)[:, :, None]
        gap = distance - RADIUS - other_radius
        strength = weight * PUSH * np.exp(-gap / PUSH_RANGE)
        # in front or behind? cos = 1 straight ahead, -1 straight behind
        speed = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        heading = self.velocity / np.maximum(speed, 1e-6)
        cos = -np.einsum("ik,ijk->ij", heading, away)
        strength = strength * (BEHIND_WEIGHT + (1 - BEHIND_WEIGHT) * (1 + cos) / 2)
        strength = np.where(itself, 0.0, strength)
        right = np.stack([away[:, :, 1], -away[:, :, 0]], axis=2)
        sideways = SIDESTEP * np.maximum(cos, 0.0)
        push = strength[:, :, None] * (away + sideways[:, :, None] * right)
        return push.sum(axis=1)

    def step(self, dt, robot_position):
        """Move everyone forward by dt seconds. robot_position: (x, y)."""
        robot = np.asarray(robot_position, dtype=float).reshape(1, 2)
        to_goal = self.goal - self.position
        for i in np.flatnonzero(np.linalg.norm(to_goal, axis=1) < GOAL_REACHED):
            self._new_goal(i)
        to_goal = self.goal - self.position
        distance = np.linalg.norm(to_goal, axis=1, keepdims=True)
        wanted = self.speed[:, None] * to_goal / distance

        acceleration = (wanted - self.velocity) / RELAX_TIME  # toward the goal
        acceleration += self._pushes(self.position, RADIUS, 1.0)  # around people
        acceleration += self._pushes(robot, ROBOT_SPACE, ROBOT_WEIGHT)  # and the robot

        self.velocity += acceleration * dt
        speed = np.linalg.norm(self.velocity, axis=1)
        too_fast = np.maximum(speed / (MAX_SPEED_FACTOR * self.speed), 1.0)
        self.velocity /= too_fast[:, None]
        self.position += self.velocity * dt
