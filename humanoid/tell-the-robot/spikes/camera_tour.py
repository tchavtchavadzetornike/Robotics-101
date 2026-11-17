"""Phase 1 spike: run the mission headless and look through the head camera.

Saves a color and a depth image when chosen steps start, and checks the depth
images against exact ray casts (mj_ray): how far off is the rendered depth?

    uv run python humanoid/tell-the-robot/spikes/camera_tour.py
"""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import PIL.Image as Image

HERE = Path(__file__).resolve().parent
LESSON = HERE.parents[1] / "03-g1-pick-place"
sys.path.insert(0, str(LESSON))

from g1 import messages, robot  # noqa: E402
from g1.arm import ArmController  # noqa: E402
from g1.locomotion import POLICY_DT, Locomotion  # noqa: E402
from g1.mission import Mission  # noqa: E402
from g1.simulator import Simulator  # noqa: E402

WIDTH, HEIGHT = 640, 480
SNAP_AT = {0: "start", 1: "at_table_a", 3: "hand_at_bottle", 9: "at_table_b"}
OUT = HERE / "out"


def intrinsics(model, camera, width, height):
    fovy = math.radians(model.cam_fovy[model.camera(camera).id])
    fy = (height / 2) / math.tan(fovy / 2)
    return fy, fy, width / 2, height / 2  # fx, fy, cx, cy


def depth_errors(model, data, depth, camera, samples=400, rng=np.random.default_rng(0)):
    """Rendered depth minus ray-cast depth, at random pixels (meters)."""
    fx, fy, cx, cy = intrinsics(model, camera, WIDTH, HEIGHT)
    cam = model.camera(camera).id
    origin = data.cam_xpos[cam]
    rotation = data.cam_xmat[cam].reshape(3, 3)
    errors = []
    for _ in range(samples):
        u, v = rng.integers(0, WIDTH), rng.integers(0, HEIGHT)
        d_image = np.array([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, 1.0])
        d_mujoco = d_image * [1, -1, -1]  # image: y down, z ahead. MuJoCo camera: y up, looks along -z
        direction = rotation @ d_mujoco
        norm = np.linalg.norm(direction)
        geom = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(model, data, origin, direction / norm, np.array([1, 1, 1, 0, 0, 0], np.uint8), 1, -1, geom)
        if distance < 0 or distance > 6:
            continue
        errors.append(depth[v, u] - distance / norm)
    return np.array(errors)


def save(name, rgb, depth):
    Image.fromarray(rgb).save(OUT / f"{name}_rgb.png")
    gray = (255 * (1 - np.clip((depth - 0.2) / 3.0, 0, 1))).astype(np.uint8)
    Image.fromarray(gray).save(OUT / f"{name}_depth.png")


sim = Simulator(str(LESSON / "robot/scene.xml"))
locomotion = Locomotion(str(LESSON / "policies/g1_walk.onnx"))
arm = ArmController(str(LESSON / "robot/g1_with_hands.xml"))
mission = Mission(0.0, (1.0, -1.52), math.radians(-90), 0.80)
renderer = mujoco.Renderer(sim.model, HEIGHT, WIDTH)
OUT.mkdir(exist_ok=True)
print("intrinsics fx, fy, cx, cy:", np.round(intrinsics(sim.model, "head_cam", WIDTH, HEIGHT), 1))

status = None
command_id = None
next_control = 0.0
snapped = set()
while sim.time < 60 and not mission.finished and not mission.failed:
    if sim.time >= next_control - 1e-9:
        next_control += POLICY_DT
        t = sim.time
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

        step = mission.index
        if step in SNAP_AT and step not in snapped:
            snapped.add(step)
            renderer.update_scene(sim.data, camera="head_cam")
            rgb = renderer.render()
            renderer.enable_depth_rendering()
            renderer.update_scene(sim.data, camera="head_cam")
            depth = renderer.render()
            renderer.disable_depth_rendering()
            save(SNAP_AT[step], rgb, depth)
            e = depth_errors(sim.model, sim.data, depth, "head_cam")
            near = e[np.abs(e) < 1]
            print(f"t={t:5.1f} {SNAP_AT[step]:15s} depth error vs ray: median {1000*np.median(near):6.2f} mm, "
                  f"|err| 95% {1000*np.percentile(np.abs(near), 95):6.2f} mm, outliers {np.sum(np.abs(e) >= 0.05)}/{len(e)}")
    sim.step()
