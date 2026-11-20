"""Walk to the goal without bumping into anyone.

The method is "try many, keep the best" (the idea of the Dynamic Window
Approach, a standard local planner for mobile robots):
  1. List the velocities the robot could walk with next: a few speeds in
     many directions, and standing still.
  2. For each one, look ahead a few seconds: the robot keeps that velocity,
     every person keeps theirs. How close do they get?
  3. Give each one a cost: getting close to people is bad, not getting
     nearer to the goal is bad, changing your mind all the time is bad.
  4. Walk with the cheapest one. Do it all again 10 times per second.

The chosen velocity is in the world frame. `to_cmd_vel` turns it into what
the walking controller takes: forward, sideways and turn speed.
"""

import numpy as np

from .config import PERSON_RADIUS

# ---- what the robot can do ----
# Speed changes are smoothed with separate limits for speeding up and for
# braking, like ROS 2 Nav2's velocity smoother.
MAX_ACCEL = 0.5  # m/s^2: 0 to 1 m/s in 2 s
# Braking. The safety layer (safety.py) relies on this value, so it must be
# what the robot really achieves: ISO 10218-1 asks for the stopping distance
# to be measured (scripts/measure_walking.py). Measured: at least 0.95 m/s^2
# (README, section 2), so 0.5 is on the safe side.
MAX_DECEL = 0.5  # m/s^2
MAX_TURN_ACCEL = 1.0  # rad/s^2
# As fast as the walking policy allows: NVIDIA's AGILE G1 walking policy is
# trained for forward commands up to 1.5 m/s (WBC-AGILE, locomotion_height/g1/
# velocity_height_env_cfg.py: lin_vel_x=(-0.5, 1.5)). To be confirmed on the
# released policy file with scripts/measure_walking.py.
MAX_SPEED = 1.5  # m/s
SPEEDS = (1.5, 1.0, 0.5)  # m/s: the speeds it tries, plus standing still
# The directions it tries, relative to the straight line to the goal: every
# 15 degrees, up to 90 degrees to each side. It never walks away from the
# goal: people are twice as fast, so running away doesn't help. When no
# direction is good, it slows down or stands still and lets people pass.
DIRECTIONS = np.radians(np.arange(-90, 91, 15))
MAX_SIDEWAYS = 0.25  # m/s
MAX_TURN = 1.0  # rad/s
CRUISE_TURN_GAIN = 2.0  # 1/s: 0.5 rad off the wanted direction -> turn at 1 rad/s

# ---- looking ahead ----
HORIZON = 4.0  # s: how far ahead it looks
LOOK_DT = 0.2  # s: it checks every 0.2 s of that future
PLAN_DT = 0.1  # s between two plans (it plans on every lidar scan, 10 Hz)
# Sizes seen from above. Together: centers closer than 0.52 m = a bump.
ROBOT_RADIUS = 0.30  # m: the G1 with its arms (PERSON_RADIUS is in config.py)
COMFORT = 0.8  # m of free space the robot likes to keep to a person
# People change their minds, and the tracker's idea of their velocity is a bit
# off. So for every second it looks ahead, the planner treats each person as
# this much bigger:
GROWTH = 0.05  # m per second

# ---- the costs (the unit is "meters of progress": 1 = losing 1 m toward the goal) ----
NEAR_COST = 4.0  # for coming as close as touching (less for staying farther away)
BUMP_COST = 60.0  # extra, for overlapping a person by COMFORT: far worse than any detour
CHANGE_COST = 0.5  # per radian of direction change: don't flip-flop
ARRIVED = 0.3  # m: close enough to the goal
SLOW_DOWN_GAIN = 0.8  # 1/s: near the goal, speed = 0.8 x distance
MIN_SPEED = 0.15  # m/s: the walking network ignores smaller commands


def wrap(angle):
    """Angle in [-pi, pi): 350 deg and -10 deg are the same heading."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def limit_change(value, previous, max_change):
    """`value`, but at most `max_change` away from `previous`."""
    return previous + np.clip(value - previous, -max_change, max_change)


def smooth_speed(value, previous, dt):
    """Change a speed by at most MAX_ACCEL*dt when it grows and MAX_DECEL*dt
    when it shrinks (toward zero)."""
    speeding_up = abs(value) > abs(previous) and value * previous >= 0
    return limit_change(value, previous, (MAX_ACCEL if speeding_up else MAX_DECEL) * dt)


class Planner:
    def __init__(self, goal):
        self.goal = np.asarray(goal, dtype=float)
        self.velocity = np.zeros(2)  # the world-frame velocity chosen last
        self.moving = np.zeros(2)  # the velocity the robot has by now (see plan)
        self.cmd = np.zeros(3)  # the last cmd_vel sent
        self.arrived = False
        self.times = np.arange(LOOK_DT, HORIZON + 1e-9, LOOK_DT)

    def plan(self, position, people):
        """Choose the walking velocity. people: rows (x, y, vx, vy), world frame.

        Returns the chosen world-frame velocity (vx, vy).
        """
        position = np.asarray(position[:2], dtype=float)
        to_goal = self.goal - position
        distance = np.hypot(*to_goal)
        if distance < ARRIVED:
            self.arrived = True
        if self.arrived:
            self.velocity = np.zeros(2)
            self.moving = np.zeros(2)
            return self.velocity

        goal_direction = np.arctan2(to_goal[1], to_goal[0])
        top_speed = min(MAX_SPEED, max(MIN_SPEED, SLOW_DOWN_GAIN * distance))
        candidates = [np.zeros(2)]
        for speed in SPEEDS:
            speed = min(speed, top_speed)
            for offset in DIRECTIONS:
                angle = goal_direction + offset
                candidates.append(speed * np.array([np.cos(angle), np.sin(angle)]))
        candidates = np.array(candidates)  # (C, 2)

        # Where the robot would be: (C, T, 2). A walking robot can't change
        # its velocity at once: it gets from what it does now (self.moving) to
        # the candidate at MAX_ACCEL, and only then keeps the candidate.
        change = candidates - self.moving  # (C, 2)
        size = np.linalg.norm(change, axis=1, keepdims=True)  # (C, 1)
        done = np.minimum(1.0, MAX_ACCEL * self.times[None, :] / np.maximum(size, 1e-9))
        velocity = self.moving + change[:, None, :] * done[:, :, None]  # (C, T, 2)
        robot_path = position + np.cumsum(velocity * LOOK_DT, axis=1)
        cost = np.zeros(len(candidates))

        # 1. progress: how much nearer to the goal it gets within the horizon,
        # in meters. The nearest point of the path counts, so that close to the
        # goal, walking "through" it isn't judged as walking away from it.
        nearest = np.linalg.norm(self.goal - robot_path, axis=2).min(axis=1)
        cost += nearest - distance

        # 2. people: predicted free space at every looked-at moment
        if len(people):
            people = np.asarray(people, dtype=float)
            people_path = people[:, None, 0:2] + people[:, None, 2:4] * self.times[None, :, None]  # (P, T, 2)
            gap = (
                np.linalg.norm(robot_path[:, None, :, :] - people_path[None, :, :, :], axis=3)
                - ROBOT_RADIUS
                - PERSON_RADIUS
                - GROWTH * self.times
            )  # (C, P, T)
            # the far future is less certain, so it counts less: 1 now, 0.25 at the horizon
            weight = 1.0 - 0.75 * self.times / HORIZON
            # closeness: 0 = COMFORT away or more, 1 = touching, above 1 = overlapping
            closeness = np.clip((COMFORT - gap) / COMFORT, 0.0, 2.0)
            danger = NEAR_COST * closeness + BUMP_COST * np.clip(closeness - 1.0, 0.0, 1.0)
            # for each person the worst moment counts; then all people add up
            cost += (danger * weight).max(axis=2).sum(axis=1)

        # 3. steadiness: prefer the direction the robot is walking in now
        if np.hypot(*self.moving) > 0.1:
            now = np.arctan2(self.moving[1], self.moving[0])
            angles = np.arctan2(candidates[:, 1], candidates[:, 0])
            walking = np.linalg.norm(candidates, axis=1) > 1e-3
            cost += CHANGE_COST * np.abs(wrap(angles - now)) * walking

        self.velocity = candidates[np.argmin(cost)]
        # what the robot will really be doing when we plan again
        step = self.velocity - self.moving
        size = np.hypot(*step)
        self.moving += step * min(1.0, MAX_ACCEL * PLAN_DT / max(size, 1e-9))
        return self.velocity

    def to_cmd_vel(self, heading, dt):
        """The chosen world velocity -> (forward, left, turn) for the legs.

        The robot turns to face the way it walks, like a person. While it is
        still turning, the part of the velocity that isn't "forward" is done
        by stepping sideways (up to MAX_SIDEWAYS).
        """
        speed = np.hypot(*self.velocity)
        if speed < 1e-3:
            wanted = np.zeros(3)
        else:
            off = wrap(np.arctan2(self.velocity[1], self.velocity[0]) - heading)
            wanted = np.array(
                [
                    max(speed * np.cos(off), 0.0),  # never walk backwards blind
                    np.clip(speed * np.sin(off), -MAX_SIDEWAYS, MAX_SIDEWAYS),
                    np.clip(CRUISE_TURN_GAIN * off, -MAX_TURN, MAX_TURN),
                ]
            )
        self.cmd[0] = smooth_speed(wanted[0], self.cmd[0], dt)
        self.cmd[1] = smooth_speed(wanted[1], self.cmd[1], dt)
        self.cmd[2] = limit_change(wanted[2], self.cmd[2], MAX_TURN_ACCEL * dt)
        return tuple(float(v) for v in self.cmd)
