"""dora node: the safety layer between the planner and the robot.

Every walking command passes through here; it is only ever slowed down or
stopped, never made faster (speed and separation monitoring, ISO/TS 15066;
the same place in the graph as ROS 2 Nav2's Collision Monitor).

Inputs:  cmd_vel_nav  the planner's walking command (50 Hz)
         tracks       the people the robot knows about, from tracker (10 Hz)
         scan         one turn of the lidar, from sim (10 Hz): its points are obstacles too
         base_pose    where the robot is, from sim (50 Hz)
Outputs: cmd_vel      the command the robot may carry out (to sim and back to the planner)

The logic is g1.safety.SpeedSeparationMonitor; this node only passes messages.
If the tracks or the scan are older than MAX_AGE (a node died or stalled), the
robot is told to stand still: what it knows about its surroundings is no longer true.
"""

import numpy as np
import pyarrow as pa
from dora import Node

from g1 import messages
from g1.safety import SpeedSeparationMonitor, sensors_fresh
from g1.tracker import scan_points

safety = SpeedSeparationMonitor()
pose = None
pose_time = None
tracks = np.zeros((0, 4))
tracks_time = None
points = np.zeros((0, 2))
scan_time = None
node = Node()

for event in node:
    if event["type"] == "STOP":
        break
    if event["type"] != "INPUT":
        continue
    if event["id"] == "base_pose":
        pose_time, (x, y), heading = messages.unpack_pose(event["value"].to_numpy())
        pose = (x, y, heading)
    elif event["id"] == "tracks":
        tracks_time, tracks = messages.unpack_tracks(event["value"].to_numpy())
    elif event["id"] == "scan" and pose is not None:
        scan_time, ranges = messages.unpack_scan(event["value"].to_numpy())
        points, _ = scan_points(ranges, pose)
    elif event["id"] == "cmd_vel_nav":
        wanted = messages.unpack_cmd_vel(event["value"].to_numpy())
        known = pose is not None and sensors_fresh(pose_time, tracks_time, scan_time)
        cmd = safety.limit(wanted, pose, tracks, points) if known else (0.0, 0.0, 0.0)
        node.send_output("cmd_vel", pa.array(messages.pack_cmd_vel(cmd)))
