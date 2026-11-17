# Tell the robot, phase 2: pixels to 3D

Goal: the mission runs from the head camera alone. `task` no longer sees the simulator's
bottle position. All code in `humanoid/03-g1-pick-place/`.

## What was added

| File | What it is |
| --- | --- |
| `g1/perception.py` | `hsv`, `color_mask`, `back_project`, `to_world`, `cylinder_center`, `find_bottle`, `keep_height` |
| `nodes/perception.py` | images + camera pose in, `bottle_pose` out (15 Hz) |
| `dataflow.yml` | `task` takes `bottle_pose` from `perception`; `sim`'s own `bottle_pose` goes only to the dashboard |
| `nodes/dashboard.py` | the seen bottle (yellow) next to the true one (blue), a `camera error` chart in cm |
| `tests/test_mission.py` | `--vision` and `test_mission_vision`; `bottle_shift` to move the bottle |
| `spikes/vision_check.py` | perception next to the truth for every step of the mission |

## How it finds the bottle

1. **Color mask:** hue 190–225°, saturation ≥ 0.7, brightness ≥ 0.4
2. **Back-projection:** `X = (u + 0.5 − cx)·Z/fx`, `Y = (v + 0.5 − cy)·Z/fy`, Z from the depth image
3. **To the room:** `p_world = R · p_camera + t`, with `R, t` from `Camera.pose()`
4. **Height check:** drop every point below 0.5 m
5. **Center, sideways:** fit a circle of radius 3 cm to the points seen from above (Gauss-Newton,
   10 iterations at most), start one radius behind the points
6. **Center, height:** 98th percentile of the point heights (the top rim) minus 7.5 cm.
   Unknown if the bottle touches the top edge of the image, then the last known height

## Results

Per step, over a full mission (`spikes/vision_check.py`), median / worst:

| Phase of the mission | xy error | z error |
| --- | --- | --- |
| walk to table A, grasp | 0.05–0.18 / 0.23 cm | 0.00 / 0.03 cm |
| lift, carry, walk to table B | 0.4–0.5 / 1.0 cm | 0.1–0.2 / 0.3 cm |
| put down, open the hand | 0.6 / 0.7 cm | 0.2–0.25 / 0.26 cm |

Seen in 725 of 725 frames, fingers around it included. Height unknown in 9 frames (first walk).

| Run | Truth | Camera |
| --- | --- | --- |
| default position | 0.9 cm | 1.4 cm |
| bottle 15 cm left | 1.0 cm | 1.3 cm |
| bottle 15 cm right | 0.7 cm | 1.5 cm |
| bottle 8 cm nearer | 0.8 cm | 1.6 cm |
| bottle 10 cm further back | FAIL | FAIL |
| live dora run | | 1.2 cm, 100% real time |

Perception: 10.6 ms per frame, 427 of 429 frames answered in a live run.

## Things learned the hard way

- **Color alone fails.** The floor tiles have exactly the bottle's hue (210°). Geometry after
  back-projection (height above 0.5 m) separates them. Detectors in phase 3 need the same check.
- **The occlusion problem was not one.** The plan expected the hand to hide the bottle. At a
  40° camera tilt enough of it stays visible, measured over every frame.
- **The cut-off top was the real problem.** Walking far away, the head sways and the bottle's
  top leaves the image: "top minus half a bottle" was 5.5 cm too low. Now: height unknown.
- **Further back fails for a reason that has nothing to do with seeing.** 10 cm deeper on the
  table, the robot would have to stand 8 cm from the edge, its legs hit the table, it never
  arrives. Phase 4: `pick` must check reachability before walking.
- **The dashboard drops messages, the task doesn't.** The dashboard logged 299 of ~900 seen
  positions (busy compressing images). Only the chart is coarser.

## Exercises

1. In `cylinder_center`, why start one radius *behind* the points, away from the camera? Try
   starting at the mean of the points: does it still converge?
2. The Jacobian is `-(p − c)/|p − c|`. Derive it from `residual = |p − c| − r`.
3. Why the 98th percentile for the top and not the maximum?
4. Set `MIN_HEIGHT = 0` and run `vision_check.py`. What happens to the xy error, and why?
5. The xy error grows from 0.1 cm to 0.5 cm once the bottle is in the hand. Two possible
   reasons: fewer visible points, or the circle fit gets points from the light-blue top. Find out
   which by printing the number of points and their heights.
6. The robot's own position still comes from `sim/base_pose`. If localization had 2 cm error,
   how big would the bottle error in the room be? And in the robot frame, where the arm needs it?

## Next: phase 3, many objects, found by name

Random scenes with 3 to 6 objects, an open-vocabulary detector (YOLO-World or OWLv2) instead of
the color mask, colors labelled from the mask, and a `world` node that remembers every object.
