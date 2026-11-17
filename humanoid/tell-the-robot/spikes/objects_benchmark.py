"""Phase 3 spike: the whole chain against the truth. Detector boxes -> g1.objects.describe.

Same 10 scenes x 4 views as detector_benchmark.py. Every true object with at
least 150 visible pixels should be found. A seen object matches the nearest
true object within 6 cm. Reported: found, kind right, color right, position
error, and seen objects that match nothing (false).

    uv run python humanoid/tell-the-robot/spikes/objects_benchmark.py [scenes] [threshold]
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LESSON = HERE.parents[1] / "03-g1-pick-place"
sys.path.insert(0, str(LESSON))
sys.path.insert(0, str(HERE))

from detector_benchmark import MIN_PIXELS, VIEWS  # noqa: E402
VIEWS = {**VIEWS, "survey_a": (0.95, -0.7, 0.71), "survey_b": (0.95, -0.7, -1.47)}
from g1 import objects, scenes  # noqa: E402
from g1.camera import Camera  # noqa: E402
from g1.detector import Detector  # noqa: E402
from g1.simulator import Simulator  # noqa: E402

MATCH = 0.06


def main(scene_count, threshold):
    detector = Detector(threshold=threshold)
    found = kind_right = color_right = false = total = 0
    errors, times, misses = [], [], []
    for seed in range(scene_count):
        scene = scenes.make_scene(seed)
        path = str(scenes.write_scene(scene, LESSON / "robot"))
        camera = Camera(path, "head_cam", 640, 480)
        for view, start in VIEWS.items():
            sim = Simulator(path, start=start)
            camera.set_state(sim.world_state())
            rgb, depth, seg = camera.color(), camera.depth(), camera.segmentation()
            position, rotation = camera.pose()
            visible = [o for o in scene.objects if (seg == camera.model.body(o.name).id).sum() >= MIN_PIXELS]
            truth = {o.name: sim.data.xpos[camera.model.body(o.name).id].copy() for o in scene.objects}
            start_time = time.perf_counter()
            seen = objects.look(detector, rgb, depth, camera.fx, camera.fy, camera.cx, camera.cy, position, rotation)
            times.append(time.perf_counter() - start_time)
            total += len(visible)
            used = set()
            for s in seen:
                best = min(scene.objects, key=lambda o: np.hypot(*(truth[o.name][:2] - s.position[:2])))
                distance = np.hypot(*(truth[best.name][:2] - s.position[:2]))
                if distance > MATCH or best.name in used:
                    false += 1
                    why = "same object again" if distance <= MATCH else f"{100 * distance:.0f} cm from {best.label}"
                    misses.append(f"seed {seed} {view}: false {s.label} at {s.position.round(2)} ({s.points} points), {why}")
                    continue
                used.add(best.name)
                if best in visible:
                    found += 1
                    errors.append(distance)
                    kind_right += s.kind == best.kind
                    color_right += s.color == best.color
                    if s.kind != best.kind or s.color != best.color:
                        misses.append(f"seed {seed} {view}: {best.label} seen as {s.label}")
            for o in visible:
                if o.name not in used:
                    misses.append(f"seed {seed} {view}: {o.label} not found")
    e = 100 * np.array(errors)
    print(f"threshold {threshold}: found {found}/{total}, kind right {kind_right}/{found}, color right {color_right}/{found}, false {false}")
    print(f"  position error cm: median {np.median(e):.2f}, 90% {np.percentile(e, 90):.2f}, worst {e.max():.2f}")
    print(f"  time per image (detector + describe): median {1000 * np.median(times):.0f} ms")
    for m in misses:
        print("   ", m)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10, float(sys.argv[2]) if len(sys.argv) > 2 else 0.2)
