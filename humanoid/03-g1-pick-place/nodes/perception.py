"""Where the bottle is, from the head camera: color, depth and geometry (g1/perception.py).

This replaces sim/bottle_pose for the task. The simulator still knows the true
position, but only the dashboard looks at it, to show how far off the camera is.

Inputs:  rgb, depth   head camera images, 10 Hz (camera)
         cam_pose     where the camera was for those images (camera)
Outputs: bottle_pose  [time, x, y, z, 1, 0, 0, 0]: the bottle center in the room,
                      time = when the image was taken. Only sent when the bottle is seen.

The camera sends rgb, depth, then cam_pose. When cam_pose arrives, the two
images of the same moment are here: that is when this node works.
"""

import pyarrow as pa
from dora import Node

from g1 import messages, perception
from g1.mission import BOTTLE_HALF_HEIGHT

BOTTLE_RADIUS = 0.03  # m, from robot/scene.xml
UPRIGHT = (1.0, 0.0, 0.0, 0.0)  # the camera can't see the bottle's turn; it stands upright

rgb = depth = None
last = None
node = Node()

for event in node:
    if event["type"] == "STOP":
        break
    if event["type"] != "INPUT":
        continue

    if event["id"] == "rgb":
        rgb = event
    elif event["id"] == "depth":
        depth = event
    elif event["id"] == "cam_pose" and rgb is not None and depth is not None:
        pose = event["value"].to_numpy()
        time = pose[0]
        if rgb["metadata"]["time"] != time or depth["metadata"]["time"] != time:
            continue  # images from another moment: wait for the next set
        meta = rgb["metadata"]
        height, width = meta["height"], meta["width"]
        seen, points = perception.find_bottle(
            rgb["value"].to_numpy().reshape(height, width, 3),
            depth["value"].to_numpy().reshape(height, width),
            meta["fx"],
            meta["fy"],
            meta["cx"],
            meta["cy"],
            pose[1:4],
            pose[4:13].reshape(3, 3),
            BOTTLE_RADIUS,
            BOTTLE_HALF_HEIGHT,
        )
        if seen is None:
            continue
        seen = perception.keep_height(seen, last)
        if seen is None:
            continue
        last = seen
        node.send_output(
            "bottle_pose", pa.array(messages.pack_pose(time, seen, UPRIGHT)), {"points": int(points)}
        )
