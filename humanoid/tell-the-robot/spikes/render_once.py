"""Phase 1 spike: can MuJoCo render the scene offscreen, in color and depth, in plain python?

Run from the repository root:
    uv run python humanoid/tell-the-robot/spikes/render_once.py
Writes out/spike_rgb.png and out/spike_depth.png next to this file, prints timings.
"""

import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SCENE = HERE.parents[1] / "03-g1-pick-place" / "robot" / "scene.xml"
OUT = HERE / "out"
WIDTH, HEIGHT = 640, 480

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
data.qpos[2] = 0.79  # pelvis height while standing
mujoco.mj_forward(model, data)

torso = model.body("torso_link").id
head = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == torso and model.geom_group[g] == 2 and model.geom_dataid[g] == model.mesh("head_link").id)
center = data.geom_xpos[head] + data.geom_xmat[head].reshape(3, 3) @ model.geom_aabb[head][:3]
half = model.geom_aabb[head][3:]
print(f"head mesh: center {center.round(3)} (world), half size {half.round(3)}")
print(f"torso origin {data.xpos[torso].round(3)}")

camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = [1.42, 0.0, 0.85]  # the bottle
camera.distance, camera.azimuth, camera.elevation = 1.4, 180, -35

renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
start = time.perf_counter()
renderer.update_scene(data, camera)
rgb = renderer.render()
t_rgb = time.perf_counter() - start
renderer.enable_depth_rendering()
start = time.perf_counter()
renderer.update_scene(data, camera)
depth = renderer.render()
t_depth = time.perf_counter() - start
renderer.disable_depth_rendering()

times = []
for _ in range(50):
    start = time.perf_counter()
    renderer.update_scene(data, camera)
    renderer.render()
    times.append(time.perf_counter() - start)

OUT.mkdir(exist_ok=True)
import PIL.Image as Image  # noqa: E402

Image.fromarray(rgb).save(OUT / "spike_rgb.png")
near, far = depth.min(), np.percentile(depth, 99)
gray = (255 * (1 - np.clip((depth - near) / (far - near), 0, 1))).astype(np.uint8)
Image.fromarray(gray).save(OUT / "spike_depth.png")
print(f"rgb {rgb.shape} {rgb.dtype}, depth {depth.shape} {depth.dtype} range {depth.min():.3f}-{depth.max():.3f} m")
print(f"first rgb {1000*t_rgb:.1f} ms, first depth {1000*t_depth:.1f} ms, steady rgb {1000*np.median(times):.1f} ms")
