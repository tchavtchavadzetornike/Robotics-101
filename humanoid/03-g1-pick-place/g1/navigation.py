"""Go to a pose: turns "be there, facing that way" into a walking velocity.

A pose is (x, y, heading) on the floor, in the world frame.
The walking velocity is (forward m/s, left m/s, turn rad/s) in the robot's
own frame: what the locomotion node takes (like the G1's
LocoClient.Move(vx, vy, vyaw) and ROS's cmd_vel).

Two rules, both standard in mobile robots:
  1. Proportional control: speed = gain x distance still to go, capped at a
     maximum. Far away: full speed. Close by: slower. Inside the tolerance: stop.
     The forward and sideways speeds are capped together, so the robot
     keeps heading straight at the goal.
  2. Smoothing: the velocity may change only by a limited amount per second
     (like ROS Nav2's velocity smoother). The walking policy needs time to
     speed up and slow down, like a person does.

There is no obstacle avoidance: the room is laid out so straight paths are
free. A real robot would add a path planner here.
"""

import numpy as np

GAIN = 0.8  # 1/s: 10 cm away -> 0.08 m/s
TURN_GAIN = 1.0  # 1/s: 0.1 rad off -> 0.1 rad/s
MAX_FORWARD = 0.4  # m/s (the policy was trained up to 1 m/s; slower is steadier)
MAX_BACKWARD = 0.15  # m/s: backing away from a table slowly rocks less, like a person does
MAX_SIDEWAYS = 0.25  # m/s
MAX_TURN = 0.6  # rad/s
# The walking policy ignores very small commands (it keeps stepping in place),
# so close to the goal we still ask for at least this much:
MIN_SPEED = 0.08  # m/s
MIN_TURN = 0.15  # rad/s
# Sideways the dead zone is wider: a 7 cm/s sideways correction made the robot step in place while
# it drifted 10-15 cm (tell-the-robot phase 4). Above a small sideways error, ask for at least this:
MIN_SIDEWAYS = 0.10  # m/s
SIDEWAYS_ERROR = 0.03  # m
MAX_ACCEL = 0.5  # m/s^2: from 0 to 0.4 m/s in 0.8 s
MAX_TURN_ACCEL = 1.0  # rad/s^2
POSITION_TOLERANCE = 0.04  # m: arrived
HEADING_TOLERANCE = 0.06  # rad (3.4 deg)
# After stopping, the robot may have slid a little. Up to this much is fine
# (the arm adapts to where the robot really stands); more means walk again.
CLOSE_POSITION = 0.08  # m
CLOSE_HEADING = 0.12  # rad (7 deg)


def wrap(angle):
    """Angle in [-pi, pi): 350 deg and -10 deg are the same heading."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def error_in_robot_frame(pose, goal):
    """How far the goal is: (ahead m, left m, turn rad), seen from the robot."""
    x, y, heading = pose
    dx, dy = goal[0] - x, goal[1] - y
    ahead = np.cos(heading) * dx + np.sin(heading) * dy
    left = -np.sin(heading) * dx + np.cos(heading) * dy
    return ahead, left, wrap(goal[2] - heading)


def close_to(pose, goal):
    """True if `pose` is within the "good enough" distance of `goal`."""
    ahead, left, turn = error_in_robot_frame(pose, goal)
    return np.hypot(ahead, left) < CLOSE_POSITION and abs(turn) < CLOSE_HEADING


def limit_change(value, previous, max_change):
    return previous + np.clip(value - previous, -max_change, max_change)


class Navigator:
    def __init__(self):
        self.cmd = np.zeros(3)  # last velocity sent

    def stop(self):
        self.cmd = np.zeros(3)

    def update(self, pose, goal, dt):
        """Velocity toward `goal` for the next dt seconds.

        Returns ((forward, left, turn), arrived).
        """
        ahead, left, turn = error_in_robot_frame(pose, goal)
        if np.hypot(ahead, left) < POSITION_TOLERANCE and abs(turn) < HEADING_TOLERANCE:
            self.stop()
            return (0.0, 0.0, 0.0), True

        distance = np.hypot(ahead, left)
        vx, vy = GAIN * ahead, GAIN * left
        if distance > POSITION_TOLERANCE and GAIN * distance < MIN_SPEED:
            vx, vy = MIN_SPEED * ahead / distance, MIN_SPEED * left / distance
        max_ahead = MAX_FORWARD if vx >= 0 else MAX_BACKWARD
        too_fast = max(abs(vx) / max_ahead, abs(vy) / MAX_SIDEWAYS, 1.0)
        vx, vy = vx / too_fast, vy / too_fast  # same direction, slower
        if abs(left) > SIDEWAYS_ERROR and abs(vy) < MIN_SIDEWAYS:
            vy = MIN_SIDEWAYS if left > 0 else -MIN_SIDEWAYS
        vturn = np.clip(TURN_GAIN * turn, -MAX_TURN, MAX_TURN)
        if abs(turn) > HEADING_TOLERANCE and abs(vturn) < MIN_TURN:
            vturn = MIN_TURN if turn > 0 else -MIN_TURN

        self.cmd[0] = limit_change(vx, self.cmd[0], MAX_ACCEL * dt)
        self.cmd[1] = limit_change(vy, self.cmd[1], MAX_ACCEL * dt)
        self.cmd[2] = limit_change(vturn, self.cmd[2], MAX_TURN_ACCEL * dt)
        return tuple(float(v) for v in self.cmd), False
