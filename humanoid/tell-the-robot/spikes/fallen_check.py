"""Fix after phase 5: what does the camera see when an object lies on its side?

Room 0 seen from the survey spot, every object standing, then every object laid on
its side (turned 90 degrees about a horizontal axis, resting on the table). For each:
the detector's box, the measured top above the table, and the extent of its points.

    humanoid/tell-the-robot/light.sh uv run python humanoid/tell-the-robot/spikes/fallen_check.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

LESSON = Path(__file__).resolve().parents[2] / "03-g1-pick-place"
sys.path.insert(0, str(LESSON))

from g1 import objects, perception, scenes, survey  # noqa: E402
from g1.camera import Camera  # noqa: E402
from g1.detector import Detector  # noqa: E402
from g1.simulator import Simulator  # noqa: E402

SEED = 0
detector = Detector()
scene = scenes.make_scene(SEED)
path = str(scenes.write_scene(scene, LESSON / "robot"))


def lay_down(sim, obj):
    """Turn the object 90 degrees about the robot-facing horizontal axis, resting on the table."""
    joint = sim.model.joint(obj.name)
    a = sim.model.jnt_qposadr[joint.id]
    radius, _ = scenes.SHAPES[obj.kind]
    sim.data.qpos[a + 2] = scenes.TABLE_TOP + radius + 0.002
    sim.data.qpos[a + 3 : a + 7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]


for posture in ("standing", "lying"):
    for name, start in (("view A", (survey.SECOND_SPOT[0], survey.SECOND_SPOT[1], survey.SECOND_FACE_A)), ("view B", (survey.SURVEY_SPOT[0], survey.SURVEY_SPOT[1], survey.FACE_TABLE_B))):
        sim = Simulator(path, start=start)
        if posture == "lying":
            for o in scene.objects:
                lay_down(sim, o)
        mujoco.mj_forward(sim.model, sim.data)
        camera = Camera(path, "head_cam", 640, 480)
        camera.set_state(sim.world_state())
        rgb, depth = camera.color(), camera.depth()
        position, rotation = camera.pose()
        boxes, _ = detector(rgb)
        print(f"\n{posture}, {name}: {len(boxes)} boxes")
        for box in boxes:
            u1, v1, u2, v2 = [int(round(b)) for b in box]
            v, u = np.mgrid[v1:v2, u1:u2]
            v, u = v.ravel(), u.ravel()
            pts = perception.to_world(perception.back_project(u, v, depth[v, u], camera.fx, camera.fy, camera.cx, camera.cy), position, rotation)
            pts = pts[(pts[:, 2] - scenes.TABLE_TOP > 0.008) & (pts[:, 2] - scenes.TABLE_TOP < 0.3)]
            if len(pts) < 40:
                continue
            pts = pts[objects.main_cluster(pts)]
            if not objects.on_a_table(pts.mean(axis=0)):
                continue
            top = np.percentile(pts[:, 2], 99) - scenes.TABLE_TOP
            xy = pts[:, :2] - pts[:, :2].mean(axis=0)
            spread = np.sqrt(np.linalg.eigvalsh(np.cov(xy.T)))[::-1]
            truth = min(scene.objects, key=lambda o: np.hypot(*(sim.data.xpos[sim.model.body(o.name).id][:2] - pts[:, :2].mean(axis=0))))
            seen = objects.describe(box, rgb, depth, camera.fx, camera.fy, camera.cx, camera.cy, position, rotation)
            print(f"   {truth.label:14s} top {100 * top:5.1f} cm, points spread {100 * spread[0]:4.1f} x {100 * spread[1]:4.1f} cm, {len(pts):5d} points -> described as {seen.label if seen else None}")
