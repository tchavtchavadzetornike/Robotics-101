"""The robot and the room: MuJoCo physics at 500 Hz, shown in a 3D window.

On the real robot this node is the G1 itself: its motor drivers take the
joint commands, its encoders and IMU produce low_state. Here the Simulator
class (g1/simulator.py) plays that part. The window belongs to the
simulator, so you can push the robot or the bottle with the mouse
(double-click a part, then Ctrl + right-drag).

Inputs:  tick            2 ms timer: one physics step
         lower_body_cmd  joint commands for legs + waist (from locomotion)
         upper_body_cmd  joint commands for arms + hands (from arm)
Outputs: low_state       joint angles/speeds + IMU, every step (500 Hz)
         base_pose       where the robot is, every 10 steps (50 Hz)
         bottle_pose     where the bottle is, every 10 steps (50 Hz)
         world_state     every position in the scene, CAMERA_HZ times a second (default 10), for the camera

Physics starts when both controllers have sent their first command, like
a real robot that waits for its controller before it moves.
macOS: the MuJoCo window needs mjpython (see dataflow.yml).
"""

import os
import time

import mujoco.viewer
import pyarrow as pa
from dora import Node

from g1 import messages, robot
from g1.simulator import Simulator

POSE_EVERY = 10  # steps per pose message: 500 Hz / 10 = 50 Hz
# Lighter by default, the Mac overheated with live windows (tell-the-robot phase 5):
CAMERA_HZ = float(os.environ.get("CAMERA_HZ", "10"))  # head camera images per second (was 15)
DRAW_FPS = float(os.environ.get("DRAW_FPS", "30"))  # 3D window frames per second (was 60)
STATE_EVERY = round(500 / CAMERA_HZ)  # steps per world_state: 500 Hz / 50 = 10 Hz
DRAW_EVERY = round(500 / DRAW_FPS)  # ticks per window frame: 500 / 17 = about 30 frames per second
REPORT_EVERY = 5000  # steps between speed reports: every 10 s of robot time

sim = Simulator(os.environ["SCENE"])
have_lower = have_upper = False
ticks = 0
steps = 0
started_at = None


def send_pose(name, position, quat):
    node.send_output(name, pa.array(messages.pack_pose(sim.time, position, quat)))


with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
    viewer.cam.lookat[:] = [0.8, -0.6, 0.7]  # between the two tables
    viewer.cam.distance = 3.6
    viewer.cam.azimuth = 140
    viewer.cam.elevation = -25
    viewer.sync()  # draw the first (slow) frame before connecting
    node = Node()

    for event in node:
        if event["type"] == "STOP" or not viewer.is_running():
            break
        if event["type"] != "INPUT":
            continue

        if event["id"] == "lower_body_cmd":
            command = messages.unpack_joint_command(event["value"].to_numpy())
            sim.set_command(robot.LOWER_BODY, *command)
            have_lower = True

        elif event["id"] == "upper_body_cmd":
            command = messages.unpack_joint_command(event["value"].to_numpy())
            sim.set_command(robot.UPPER_BODY, *command)
            have_upper = True

        elif event["id"] == "tick":
            ticks += 1
            if have_lower and have_upper:
                sim.step()
                steps += 1
                if steps % POSE_EVERY == 0:
                    send_pose("base_pose", *sim.base_pose())
                    send_pose("bottle_pose", *sim.bottle_pose())
                if steps % STATE_EVERY == 0:
                    state = messages.pack_world_state(sim.time, sim.world_state())
                    node.send_output("world_state", pa.array(state))

            q, dq = sim.joint_state()
            quat, gyro = sim.imu()
            state = messages.pack_low_state(sim.time, q, dq, quat, gyro)
            node.send_output("low_state", pa.array(state))

            if ticks % DRAW_EVERY == 0:
                viewer.sync()
            if steps == 1:
                started_at = time.monotonic()
            elif steps > 0 and steps % REPORT_EVERY == 0:
                speed = sim.time / (time.monotonic() - started_at)
                node.log_info(
                    f"t={sim.time:5.1f}s  running at {100 * speed:.0f}% of real time"
                )
