"""Walking skill: get to a spot without bumping into anyone, then face a direction.

The whole navigation pipeline in one object: lidar scan -> people (tracker)
-> plan (planner) -> smoothed command -> safety check (safety). Any task that
walks uses it: give it a goal, call update() every control step, and send the
command to the robot's Locomotion interface.

    navigator = Navigator()
    navigator.set_goal((5.0, 0.0))
    every step:  robot.move(*navigator.update(robot.pose(), scan_or_None, dt))

Give it a scan whenever the lidar has a new one (10 Hz); in between pass None
and it keeps smoothing the command. The dora graphs run the same three parts
as separate nodes (g1/nodes: tracker, planner, safety).
"""

import numpy as np

from .planner import MAX_ACCEL, MAX_TURN, MAX_TURN_ACCEL, Planner, limit_change, wrap
from .safety import SpeedSeparationMonitor
from .tracker import Tracker, scan_points

DOCK_TURN_GAIN = 1.5  # 1/s: 0.2 rad off -> turn at 0.3 rad/s
HEADING_TOLERANCE = 0.05  # rad (3 degrees): facing the right way
# Docking: the last part to an exact spot (e.g. in front of a table), slowly,
# with a proportional controller on the error in the robot's own frame.
# Docking: NVIDIA's AGILE walking policy barely follows small commands
# (measured, scripts/measure_walking.py and the README: 0.10 m/s -> 0.04 m/s,
# 0.20 -> 0.14, 0.30 -> 0.26), so docking walks at a speed it follows and stops
# inside a tolerance the arm makes up for (hand targets use the measured pose)
DOCK_SPEED = 0.2  # m/s: slow near objects, still followed by the walking policy
DOCK_TOLERANCE = 0.05  # m: there
# Things on the map (tables, shelves): whatever the lidar sees inside them
# plus this margin is the map, not a person. As in ROS 2 Nav2, where the
# costmap's static layer holds the map and its obstacle layer what moves.
MAP_MARGIN = 0.15  # m


class Navigator:
    def __init__(self, goal=None, static_map=()):
        """static_map: rectangles (x_min, y_min, x_max, y_max) of things that
        never move, like tables."""
        self.static_map = [np.asarray(r, float) for r in static_map]
        self.tracker = Tracker()
        self.safety = SpeedSeparationMonitor()
        self.tracks = np.zeros((0, 4))
        self.points = np.zeros((0, 2))  # raw lidar points off the map: whatever the tracker doesn't call a person
        self.planner = Planner(goal if goal is not None else (0.0, 0.0))
        self.planner.arrived = goal is None

    def set_goal(self, goal) -> None:
        """Walk to a new spot (x, y); the people seen so far are kept."""
        cmd = self.planner.cmd.copy()
        self.planner = Planner(goal)
        self.planner.cmd[:] = cmd  # continue smoothly from the current command

    @property
    def arrived(self) -> bool:
        """At the goal (the robot may still be braking)."""
        return self.planner.arrived

    def update(self, pose, scan, dt):
        """The walking command (forward, left, turn) for the next dt seconds.

        pose: (x, y, heading). scan: a new lidar scan (ranges), or None.
        """
        x, y, _ = pose
        if scan is not None:
            self.tracks = self._not_on_map(self.tracker.update(scan, pose))
            self.points = self._not_on_map(scan_points(scan, pose)[0])
            self.planner.plan(np.array([x, y]), self.tracks)
        cmd = self.safety.limit(self.planner.to_cmd_vel(pose[2], dt), pose, self.tracks, self.points)
        self.planner.cmd[:] = cmd  # the smoother continues from what was really sent
        return cmd

    def dock(self, pose, goal_pose, dt):
        """Walk slowly to the exact spot goal_pose = (x, y, heading).
        Returns (command, docked): docked is True once the robot is on the
        spot, facing goal_pose's heading, and no longer turning. The safety
        layer still checks for people."""
        x, y, heading = pose
        c, s = np.cos(heading), np.sin(heading)
        dx, dy = goal_pose[0] - x, goal_pose[1] - y
        ahead, left = c * dx + s * dy, -s * dx + c * dy  # the error in the robot's frame
        distance = float(np.hypot(ahead, left))
        if distance < DOCK_TOLERANCE:
            return self.turn_to(heading, goal_pose[2], dt)
        speed = DOCK_SPEED
        wanted = np.array([speed * ahead / distance, speed * left / distance])
        turn = float(np.clip(DOCK_TURN_GAIN * wrap(goal_pose[2] - heading), -MAX_TURN, MAX_TURN))
        for i, value in enumerate(wanted):
            self.planner.cmd[i] = limit_change(value, self.planner.cmd[i], MAX_ACCEL * dt)
        self.planner.cmd[2] = limit_change(turn, self.planner.cmd[2], MAX_TURN_ACCEL * dt)
        cmd = self.safety.limit(tuple(self.planner.cmd), pose, self.tracks, self.points)
        self.planner.cmd[:] = cmd
        return cmd, False

    def turn_to(self, heading_now, heading_goal, dt):
        """Turn on the spot toward `heading_goal`. Returns (command, facing):
        facing is True once the heading is right and the turning has stopped."""
        error = wrap(heading_goal - heading_now)
        facing = abs(error) < HEADING_TOLERANCE
        wanted = 0.0 if facing else float(np.clip(DOCK_TURN_GAIN * error, -MAX_TURN, MAX_TURN))
        self.planner.cmd[:2] = 0.0
        self.planner.cmd[2] = limit_change(wanted, self.planner.cmd[2], MAX_TURN_ACCEL * dt)
        return (0.0, 0.0, float(self.planner.cmd[2])), facing and abs(self.planner.cmd[2]) < 1e-3

    def _not_on_map(self, rows):
        """The rows (x, y, ...) that are not part of the static map."""
        rows = np.asarray(rows, dtype=float)
        inside = np.zeros(len(rows), dtype=bool)
        for r in self.static_map:
            inside |= (
                (r[0] - MAP_MARGIN <= rows[:, 0])
                & (rows[:, 0] <= r[2] + MAP_MARGIN)
                & (r[1] - MAP_MARGIN <= rows[:, 1])
                & (rows[:, 1] <= r[3] + MAP_MARGIN)
            )
        return rows[~inside]
