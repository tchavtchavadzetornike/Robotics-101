# Tell the robot, phase 1: give the robot eyes

Goal: a head camera that renders color and depth, live in dora, without slowing the 500 Hz
physics loop. All code in `humanoid/03-g1-pick-place/`.

## What was added

| File | What it is |
| --- | --- |
| `robot/g1_with_hands.xml` | `head_cam` in `torso_link`: 8 cm ahead, 42 cm up, 40° down, `fovy` 58° |
| `g1/camera.py` | `Camera`: own MuJoCo world, `set_state(qpos)`, `color()`, `depth()`, `pose()`, `intrinsics()` |
| `g1/messages.py`, `g1/simulator.py` | `world_state`: `[time, qpos]`, every position in the scene |
| `nodes/sim.py` | sends `world_state` every 33 steps (15 Hz) |
| `nodes/camera.py` | renders `rgb` (uint8) and `depth` (float32 m) and sends `cam_pose`, intrinsics in the metadata |
| `nodes/dashboard.py` | head camera image, depth image, camera pyramid in the 3D room, 5 Hz, JPEG + 16-bit mm depth |
| `tests/test_camera.py` | projects the bottle center onto a pixel: must be blue, depth must match |
| `tell-the-robot/spikes/` | `render_once.py`, `camera_tour.py`: the experiments, images in `spikes/out/` |

## Results

| Check | Result |
| --- | --- |
| Offscreen rendering in plain `python` on the Mac | works, 13.7 ms per 640×480 color frame |
| Intrinsics | fx = fy = 433.0, cx = 320, cy = 240 |
| Rendered depth vs exact ray cast (`mj_ray`), 4 views, ~390 pixels each | 95% within 0.04 mm |
| Bottle center projected to a pixel | (320, 41) from the start, (447, 386) at the table, both blue |
| Depth at that pixel vs geometry | 1.239 / 1.238 m, 0.360 / 0.360 m |
| Live dataflow | 100% real time, mission done 0.7 cm, 5 Hz images in Rerun, 26.6 MB per minute saved |

The macOS warning `ARB_clip_control unavailable ... depth accuracy will be limited` does not
matter at these distances, measured above.

## Things learned the hard way

- **A cylinder seen from above looks closer than its radius suggests.** At the table the
  camera looks 61° down. The ray toward the bottle's center enters the side
  `radius / cos(61°)` = 6 cm before the center, not 3 cm. Phase 2 must use this to go from the
  visible surface to the center.
- **Two camera conventions.** MuJoCo: y up, looks along −z. Image, OpenCV and Rerun: y down,
  z ahead. One `diag(1, -1, -1)` between them (`IMAGE_TO_MUJOCO`). `Camera.pose()` already
  returns the image convention.
- **Throttling by time needs a margin.** Frames every 1/15 s: three frames are 0.198 s, so a
  0.2 s threshold waited for a fourth (3.8 Hz instead of 5). 0.19 s gives every third frame.
- **Depth as 16-bit millimeters** (what the RealSense sends) is half the size of float32 meters
  and loses nothing that matters (1 mm steps).

## Exercises

1. From `fovy` = 58° and a 480 pixel tall image, compute fy by hand. Check against 433.
2. The bottle is 6 cm wide. How many pixels wide is it from the start (Z = 1.27 m) and at the
   table (Z = 0.41 m)? Measure it in `spikes/out/start_rgb.png` and `at_table_a_rgb.png`.
3. In `test_camera.py`, `project()` does `rotation.T @ (point - position)`. Why the transpose?
4. Change the camera tilt to 30° and 50° in `g1_with_hands.xml`, run `camera_tour.py`. At which
   tilt does the bottle leave the image at the start, and at which during the grasp?
5. Why does the camera node take `world_state` with `queue_size: 1`? What would happen at 0 or
   at 100 if rendering got slower than 66 ms per frame?
6. `sim` sends `world_state`, which a real robot cannot. What replaces it on the real G1?
7. Look at `hand_at_bottle_rgb.png`. Which part of the bottle can the camera not see once
   the fingers close? What does that mean for phase 2?

## Next: phase 2, pixels to 3D

Find the bottle by color, back-project its pixels with the depth to 3D, move the result into
the world frame, and run the mission from the camera instead of `sim/bottle_pose`.
Start with exercise 1 and 3, phase 2 is those two formulas run backwards.
