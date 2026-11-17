"""The messages the nodes send each other, like ROS .msg files.

dora sends arrays of numbers. Each message here is a flat list of float64
with a fixed layout. pack_* builds one, unpack_* splits it into named parts.
Joint lists are always in robot.JOINTS order.

  low_state     sim -> locomotion, arm       joint angles + speeds, IMU   (500 Hz)
  joint_command locomotion, arm -> sim       per joint: target, kp, kd, torque
  pose          sim -> task                  robot pelvis / bottle pose   (50 Hz)
  cmd_vel       task -> locomotion           walk: forward, left, turn
  standing      locomotion -> task           1 = standing still and steady
  world_state   sim -> camera                every position in the scene (MuJoCo qpos)  (15 Hz)
  arm_command   task -> arm                  one step for the arm (see kinds below)
  arm_status    arm -> task                  which arm command is finished, ok or not
"""

import numpy as np

from g1.robot import NUM_JOINTS

# ---- low_state: [time, q x43, dq x43, quat x4 (w x y z), gyro x3] ----


def pack_low_state(time, q, dq, quat, gyro):
    return np.concatenate([[time], q, dq, quat, gyro])


def unpack_low_state(msg):
    n = NUM_JOINTS
    return (
        msg[0],
        msg[1 : 1 + n],
        msg[1 + n : 1 + 2 * n],
        msg[1 + 2 * n : 5 + 2 * n],
        msg[5 + 2 * n : 8 + 2 * n],
    )


# ---- joint_command: [target x n, kp x n, kd x n, tau x n] for a group of n joints ----


def pack_joint_command(target, kp, kd, tau):
    return np.concatenate([target, kp, kd, tau])


def unpack_joint_command(msg):
    return np.split(np.asarray(msg), 4)  # target, kp, kd, tau


# ---- pose: [time, x, y, z, qw, qx, qy, qz] (world frame; time = when measured) ----


def pack_pose(time, position, quat):
    return np.concatenate([[time], position, quat])


def unpack_pose(msg):
    return msg[0], msg[1:4], msg[4:8]


# ---- world_state: [time, qpos x model.nq] ----
# Everything that moves: the robot (pelvis + 43 joints) and every free object.
# A camera node copies it into its own MuJoCo world to draw what the robot sees.


def pack_world_state(time, qpos):
    return np.concatenate([[time], qpos])


def unpack_world_state(msg):
    return msg[0], msg[1:]


# ---- cmd_vel: [forward m/s, left m/s, turn rad/s] (robot frame) ----
# ---- standing: [1.0 or 0.0] ----

# ---- arm_command: [id, kind, x, y, z, yaw] ----
# The hand target is the right hand's grasp point in the pelvis frame (meters),
# fingers pointing horizontally at heading `yaw` (0 = straight ahead).
REST = 0  # joint move back to the walking pose
MOVE_JOINTS = 1  # joint move to the hand target (big moves, free space)
MOVE_LINE = 2  # straight-line move to the hand target (near objects)
OPEN = 3  # open the hand
CLOSE = 4  # close the hand


def pack_arm_command(command_id, kind, position=(0.0, 0.0, 0.0), yaw=0.0):
    return np.array([command_id, kind, *position, yaw], dtype=float)


def unpack_arm_command(msg):
    return int(msg[0]), int(msg[1]), np.asarray(msg[2:5]), float(msg[5])


# ---- arm_status: [id of the last finished command, 1 = ok / 0 = failed] ----


def pack_arm_status(command_id, ok):
    return np.array([command_id, 1.0 if ok else 0.0])


def unpack_arm_status(msg):
    return int(msg[0]), bool(msg[1])
