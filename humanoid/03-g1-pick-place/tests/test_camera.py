"""Check the head camera's geometry: does a known 3D point land on the right pixel?

The bottle's center is known from the simulator. Project it with the camera
pose and the intrinsics (g1/camera.py), then look at that pixel in the
rendered images: it must be bottle blue, and the depth there must be the
depth of the point where that pixel's ray enters the bottle.

  uv run python tests/test_camera.py      (prints the numbers, PASS or FAIL)
  uv run --with pytest pytest tests/
"""

import sys
from pathlib import Path

import numpy as np

LESSON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LESSON))

from g1.camera import Camera  # noqa: E402
from g1.simulator import Simulator  # noqa: E402

SCENE = str(LESSON / "robot/scene.xml")
BOTTLE_RADIUS = 0.03


def project(camera, point):
    """World point -> (u, v) pixel and its depth Z, by the pinhole model."""
    position, rotation = camera.pose()
    x, y, z = rotation.T @ (np.asarray(point) - position)
    return camera.fx * x / z + camera.cx, camera.fy * y / z + camera.cy, z


def surface_point(camera, center, radius):
    """Where the ray from the camera to the bottle's center enters its side.

    The bottle is an upright cylinder. Seen from above at an angle, the ray
    runs radius / (horizontal part of the ray) inside it before it reaches the axis,
    more than the radius itself: 3 cm at 0 deg down, 6 cm at 60 deg down.
    """
    position, _ = camera.pose()
    ray = np.asarray(center) - position
    ray /= np.linalg.norm(ray)
    horizontal = np.hypot(ray[0], ray[1])
    return np.asarray(center) - ray * radius / horizontal


def check(start):
    sim = Simulator(SCENE, start=start)
    camera = Camera(SCENE, "head_cam", 640, 480)
    camera.set_state(sim.world_state())
    rgb, depth = camera.color(), camera.depth()
    bottle, _ = sim.bottle_pose()
    u, v, z = project(camera, bottle)
    r, g, b = rgb[round(v), round(u)].astype(int)
    measured = depth[round(v), round(u)]
    _, _, expected = project(camera, surface_point(camera, bottle, BOTTLE_RADIUS))
    print(
        f"start {start}: bottle center at pixel ({u:.0f}, {v:.0f}), Z {z:.3f} m; "
        f"color ({r}, {g}, {b}); depth {measured:.3f} m, expected about {expected:.3f} m"
    )
    inside = 0 <= u < 640 and 0 <= v < 480
    blue = b > r + 60 and b > g
    return inside and blue and abs(measured - expected) < 0.01


def test_camera():
    assert check((0.0, 0.0, 0.0))  # the start: bottle 1.4 m ahead
    assert check((1.12, 0.12, 0.0))  # where the robot stands to grasp it


if __name__ == "__main__":
    ok = check((0.0, 0.0, 0.0)) and check((1.12, 0.12, 0.0))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
