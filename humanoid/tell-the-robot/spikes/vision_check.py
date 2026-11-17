"""Phase 2 spike: perception next to the truth, over a whole mission.

The mission runs on the simulator's bottle position as before. At 15 Hz the
head camera renders and g1.perception estimates the bottle. Per step: how many
bottle points were seen and how far the estimate is from the truth.

    uv run python humanoid/tell-the-robot/spikes/vision_check.py
"""

import math
import sys
from itertools import groupby
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LESSON = HERE.parents[1] / "03-g1-pick-place"
sys.path.insert(0, str(LESSON))

from g1 import messages, perception, robot  # noqa: E402
from g1.arm import ArmController  # noqa: E402
from g1.camera import Camera  # noqa: E402
from g1.locomotion import POLICY_DT, Locomotion  # noqa: E402
from g1.mission import BOTTLE_HALF_HEIGHT, Mission  # noqa: E402
from g1.simulator import Simulator  # noqa: E402

SCENE = str(LESSON / "robot/scene.xml")
CAMERA_DT = 1 / 15

sim = Simulator(SCENE)
locomotion = Locomotion(str(LESSON / "policies/g1_walk.onnx"))
arm = ArmController(str(LESSON / "robot/g1_with_hands.xml"))
mission = Mission(0.0, (1.0, -1.52), math.radians(-90), 0.80)
camera = Camera(SCENE, "head_cam", 640, 480)

status = None
command_id = None
next_control = next_frame = 0.0
rows = []
while sim.time < 60 and not mission.finished and not mission.failed:
    t = sim.time
    if t >= next_control - 1e-9:
        next_control += POLICY_DT
        q, dq = sim.joint_state()
        quat, gyro = sim.imu()
        base, base_quat = sim.base_pose()
        bottle, _ = sim.bottle_pose()
        cmd_vel, command = mission.update(t, base, base_quat, bottle, locomotion.is_standing(t), status)
        if command is not None:
            command_id, kind, position, yaw = messages.unpack_arm_command(command)
            if not arm.execute(t, kind, position, yaw):
                status, command_id = (command_id, False), None
        sim.set_command(robot.LOWER_BODY, *locomotion.update(t, cmd_vel, q[robot.LEGS], dq[robot.LEGS], quat, gyro))
        sim.set_command(robot.UPPER_BODY, *arm.update(t, q[robot.UPPER_BODY]))
        if command_id is not None and not arm.busy():
            status, command_id = (command_id, True), None
    if t >= next_frame - 1e-9:
        next_frame += CAMERA_DT
        camera.set_state(sim.world_state())
        position, rotation = camera.pose()
        seen, points = perception.find_bottle(
            camera.color(), camera.depth(), camera.fx, camera.fy, camera.cx, camera.cy,
            position, rotation, 0.03, BOTTLE_HALF_HEIGHT,
        )
        truth, _ = sim.bottle_pose()
        error = None if seen is None else seen - truth
        rows.append((t, mission.index, mission.label, points, error))
    sim.step()

print(f"mission finished={mission.finished} at t={sim.time:.1f}s\n")
print(f"{'step':40s} {'frames':>6s} {'seen':>5s} {'z?':>3s} {'points':>11s} {'xy error cm':>16s} {'z error cm':>14s}")
for (index, label), group in groupby(rows, key=lambda r: (r[1], r[2])):
    group = list(group)
    errors = np.array([r[4] for r in group if r[4] is not None])
    points = [r[3] for r in group]
    seen = len(errors)
    if seen:
        xy = 100 * np.hypot(errors[:, 0], errors[:, 1])
        z = 100 * errors[:, 2]
        z = z[~np.isnan(z)] if (~np.isnan(z)).any() else np.array([np.nan])
        xy_text = f"{np.median(xy):5.2f} / {xy.max():5.2f}"
        z_text = f"{np.median(z):5.2f} / {np.abs(z).max():4.2f}"
    else:
        xy_text = z_text = "-"
    unknown = int(np.isnan(errors[:, 2]).sum()) if seen else 0
    print(f"{index + 1:2d} {label[:37]:37s} {len(group):6d} {seen:5d} {unknown:3d} {min(points):5d}-{max(points):5d} {xy_text:>16s} {z_text:>14s}")
print("\nerror columns: median / worst")
