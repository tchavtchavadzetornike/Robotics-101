"""The robot's memory of the objects on the tables (g1/world.py).

Inputs:  seen          what the detector found in one image (detector)
         depth         head camera depth images, 10 Hz (camera): kept for a few
         cam_pose      seconds, to check what the camera should have seen in the
                       image the detector used
         arm_area      [x, y] where the arm works, [nan, nan] when done (skills)
         held          [id, x, y, z] picked up / [-1, x, y, z] put down at x, y, z (skills)
Outputs: objects       JSON text: [{id, kind, color, position, last_seen, sightings}],
                       after every update

The depth images arrive 10 times a second, the detector's results about 0.4 s
later. KEEP is how many seconds of depth images to hold.
"""

import json

import numpy as np
import pyarrow as pa
from dora import Node

from g1 import objects, world

KEEP = 3.0  # s
memory = world.World()
depths = {}  # time -> (depth image, metadata)
poses = {}  # time -> cam_pose array
node = Node()

for event in node:
    if event["type"] == "STOP":
        break
    if event["type"] != "INPUT":
        continue
    name = event["id"]

    if name == "depth":
        meta = event["metadata"]
        depths[meta["time"]] = (event["value"].to_numpy(), meta)
    elif name == "cam_pose":
        p = event["value"].to_numpy()
        poses[p[0]] = p
    elif name == "arm_area":
        area = event["value"].to_numpy()
        memory.arm_working(None if np.isnan(area[0]) else area)
    elif name == "held":
        held = event["value"].to_numpy()
        if held[0] >= 0:
            memory.hold(int(held[0]))
        else:
            memory.release(None if np.isnan(held[1]) else held[1:4])
        node.send_output("objects", pa.array([json.dumps(memory.as_list())]), {"time": 0.0})
    elif name == "seen":
        report = json.loads(event["value"][0].as_py())
        time = report["time"]
        if time not in depths or time not in poses:
            node.log_warn(f"no depth image for t={time:.2f}, skipping this detection")
            continue
        flat, meta = depths[time]
        depth = flat.reshape(meta["height"], meta["width"])
        p = poses[time]
        position, rotation = p[1:4], p[4:13].reshape(3, 3)
        fx, fy, cx, cy = meta["fx"], meta["fy"], meta["cx"], meta["cy"]
        memory.update(
            time,
            [objects.from_dict(d) for d in report["objects"]],
            lambda q: world.should_see(q, depth, fx, fy, cx, cy, position, rotation),
        )
        node.send_output("objects", pa.array([json.dumps(memory.as_list())]), {"time": time})

    for store in (depths, poses):
        if store:
            newest = max(store)
            for t in [t for t in store if t < newest - KEEP]:
                del store[t]
