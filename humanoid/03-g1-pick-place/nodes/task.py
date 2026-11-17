"""The task: walk to table A, pick up the bottle, walk to table B, put it on the mark.

Inputs:  base_pose    where the robot is (50 Hz), from sim (real robot: localization)
         bottle_pose  where the bottle is (50 Hz), from sim (real robot: camera)
         standing     from locomotion
         arm_status   from arm
Outputs: cmd_vel      walking velocity, to locomotion
         arm_command  one arm step at a time, to arm
         step         the step that just started, as text (for the dashboard)

The steps themselves are in g1/mission.py. Every time a new base_pose arrives,
this node gives the mission what the robot sees and sends out what it decides.
"""

import math
import os

import numpy as np
import pyarrow as pa
from dora import Node

from g1 import messages
from g1.mission import Mission

PLACE_POINT = [float(v) for v in os.environ["PLACE_POINT"].split(",")]
mission = Mission(
    pick_heading=math.radians(float(os.environ["PICK_HEADING_DEG"])),
    place_point=PLACE_POINT,
    place_heading=math.radians(float(os.environ["PLACE_HEADING_DEG"])),
    table_height=float(os.environ["TABLE_HEIGHT"]),
)
bottle = None
standing = False
arm_status = None
label = ""
reported = False
node = Node()

for event in node:
    if event["type"] == "STOP":
        break
    if event["type"] != "INPUT":
        continue

    if event["id"] == "bottle_pose":
        _, bottle, _ = messages.unpack_pose(event["value"].to_numpy())

    elif event["id"] == "standing":
        standing = event["value"].to_numpy()[0] > 0.5

    elif event["id"] == "arm_status":
        arm_status = messages.unpack_arm_status(event["value"].to_numpy())

    elif event["id"] == "base_pose" and bottle is not None:
        time, position, quat = messages.unpack_pose(event["value"].to_numpy())
        cmd_vel, command = mission.update(
            time, position, quat, bottle, standing, arm_status
        )
        node.send_output("cmd_vel", pa.array(cmd_vel))
        if command is not None:
            node.send_output("arm_command", pa.array(command))

        if mission.label != label and mission.label:
            label = mission.label
            step = f"{mission.index + 1}/{len(mission.steps)}"
            node.log_info(f"t={time:5.1f}s  step {step}: {label}")
            node.send_output("step", pa.array([f"step {step}: {label}"]))
        if mission.finished and not reported:
            miss = 100 * np.hypot(
                bottle[0] - PLACE_POINT[0], bottle[1] - PLACE_POINT[1]
            )
            node.log_info(
                f"t={time:5.1f}s  done: the bottle stands {miss:.1f} cm from the mark"
            )
            reported = True
        if mission.failed and not reported:
            node.log_error(f"t={time:5.1f}s  mission stopped: {mission.failed}")
            reported = True
