"""What is on the tables in the head camera's image: OWLv2 boxes, then kind, color, position.

Slow on purpose: OWLv2 takes about 0.3 s per image, so this node looks at most
DETECT_HZ times a second, and only while the robot stands still (sharper
images, and less work for the computer). The camera keeps sending 15 images a
second; this node only ever takes the newest (queue_size 1).

Inputs:  rgb, depth, cam_pose  head camera (camera)
         standing              1 = legs still (locomotion)
         tick                  how often to look
Outputs: seen                  JSON text: {"time", "objects": [{kind, color, position, points, box}]}
                               metadata: time, and the camera pose + intrinsics of that image,
                               so the world node can check what it should have seen
"""

import json

import pyarrow as pa
from dora import Node

from g1 import objects
from g1.detector import Detector

detector = Detector()
rgb = depth = pose = None
standing = False
node = Node()

for event in node:
    if event["type"] == "STOP":
        break
    if event["type"] != "INPUT":
        continue
    name = event["id"]
    if name == "rgb":
        rgb = event
    elif name == "depth":
        depth = event
    elif name == "cam_pose":
        pose = event
    elif name == "standing":
        standing = event["value"].to_numpy()[0] > 0.5
    elif name == "tick" and standing and rgb is not None and depth is not None and pose is not None:
        p = pose["value"].to_numpy()
        time = p[0]
        if rgb["metadata"]["time"] != time or depth["metadata"]["time"] != time:
            continue
        meta = rgb["metadata"]
        height, width = meta["height"], meta["width"]
        image = rgb["value"].to_numpy().reshape(height, width, 3)
        seen = objects.look(
            detector,
            image,
            depth["value"].to_numpy().reshape(height, width),
            meta["fx"],
            meta["fy"],
            meta["cx"],
            meta["cy"],
            p[1:4],
            p[4:13].reshape(3, 3),
        )
        text = json.dumps({"time": float(time), "objects": [objects.to_dict(s) for s in seen]})
        node.send_output("seen", pa.array([text]), {"time": float(time)})
