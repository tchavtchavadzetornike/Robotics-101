"""The head camera: color and depth images of what the robot sees, 10 times a second (CAMERA_HZ in sim).

It keeps its own copy of the MuJoCo world. For every world_state from sim it
moves everything into place and draws the scene from the head camera
(g1/camera.py). Drawing takes 10-20 ms, so it runs here and not in sim,
whose physics step has 2 ms.

Inputs:  world_state  every position in the scene, 10 Hz (sim)
Outputs: rgb          uint8, height x width x 3, flattened
         depth        float32 meters along the viewing axis, height x width, flattened
         cam_pose     [time, x, y, z, r11 .. r33]: where the camera is, image convention

Every output carries the same metadata: time (the sim time the image shows),
width, height, encoding, and the intrinsics fx, fy, cx, cy. That is all a
later node needs to turn a pixel and its depth into a 3D point.
"""

import os

import numpy as np
import pyarrow as pa
from dora import Node

from g1 import messages
from g1.camera import Camera

camera = Camera(
    os.environ["SCENE"],
    os.environ["CAMERA"],
    int(os.environ["WIDTH"]),
    int(os.environ["HEIGHT"]),
)
node = Node()

for event in node:
    if event["type"] == "STOP":
        break
    if event["type"] != "INPUT" or event["id"] != "world_state":
        continue

    time, qpos = messages.unpack_world_state(event["value"].to_numpy())
    camera.set_state(qpos)
    shared = {
        "time": float(time),
        "width": camera.width,
        "height": camera.height,
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
    }
    node.send_output(
        "rgb", pa.array(camera.color().ravel()), {**shared, "encoding": "rgb8"}
    )
    node.send_output(
        "depth", pa.array(camera.depth().ravel()), {**shared, "encoding": "32FC1"}
    )
    position, rotation = camera.pose()
    pose = np.concatenate([[time], position, rotation.ravel()])
    node.send_output("cam_pose", pa.array(pose), {**shared, "encoding": "pose"})
