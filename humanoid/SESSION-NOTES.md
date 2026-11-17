# Humanoid track: where we are

Goal: learn dora and robotics step by step, towards a simulated Unitree G1 humanoid in MuJoCo, on the Mac (M4 Pro).

## Done: Lesson 1, `01-mujoco-basics/`

A robot arm with one motor (shoulder). You give a target angle, and it goes there and holds.

| File | What it is |
| --- | --- |
| `pendulum.xml` | the robot: shapes, shoulder joint, 20 Nm motor, green target ghost, target slider |
| `trajectory.py` | plans smooth motion: speed up, cruise (max 86°/s), slow down, stop on target |
| `controller.py` | follows the plan: `weight + inertia*ref_accel + KP*(ref_angle-angle) + KD*(ref_speed-speed)` |
| `watch.py` | runs it in a 3D window: `uv run mjpython watch.py` |
| `swing.py` | runs it without a window, prints numbers: `uv run python swing.py --target 45` |

`watch.py` and `swing.py` are independent. Both use `pendulum.xml` and `controller.py`.

What you learned:
- **model / data:** `model` is the fixed description, `data` the changing state
- **The state values:** `qpos` = angle, `qvel` = speed, `ctrl` = motor command
- **Torque:** in Nm. Holding the 2.06 kg arm straight out takes 6.44 Nm (2.06 × 9.81 × 0.319 m)
- **Radians:** 1 rad = 57.3°
- **Smooth motion:** real robots plan a trajectory first, then the controller follows it (like MoveIt + ros2_control)
- **The control loop:** read → decide → act → repeat, 500 times per second

## Done: Lesson 2, `02-dora-nodes/`

The same arm as 7 dora nodes: `sim` (MuJoCo) → `planner` → `controller` → back to `sim`, plus `goals`, `safety` (e-stop), `plots` (Rerun) and `viewer3d` (MuJoCo window). Run: `./run.sh arm-nodes`.

YAML features used: timers, top-level and node `env`, `module`, `output_types` + `strict_types`, `input_timeout`, `queue_size`, `restart_policy`. Tested: real time at 500 Hz, tracking error under 0.3°, e-stop brakes the arm.

Presentation: https://claude.ai/artifact/NBK238P1gtejg9Yix6fYA8

## Done: Lesson 3, `03-g1-pick-place/`

The full Unitree G1 (43 motors, Dex3 hands) walks to table A, picks up a bottle, walks to table B and puts it on the green mark, in about 45 s. Run: `./run.sh g1-pick-place`. Check without a window: `uv run python tests/test_mission.py` (PASS/FAIL).

| Part | How |
| --- | --- |
| robot model | MuJoCo Menagerie G1 with hands; motors changed to torque motors (like the real G1's motor drivers) |
| walking | Unitree's trained policy (`unitree_rl_gym`), converted to ONNX; arm pose and waist gains matched to its training |
| standing | fixed pose with the weight over the middle of the feet, stiff ankles, ankle balance from the IMU |
| navigation | speed = 0.8 x distance, smoothed, with a minimum speed |
| arm | inverse kinematics, joint and line moves with a speed profile, gravity compensation |
| grasp | open hand from the side, fingers curl past the bottle and keep squeezing |
| mission | 17 steps; each computes its target when it starts |
| dora | 4 nodes: `sim`, `locomotion`, `arm`, `task`; the logic is in the `g1/` library |

Tested: the mission passes with the bottle moved up to 15 cm and from different start spots (0.5-1.0 cm from the mark). Through dora: real time, done in 43 s, 1.0 cm from the mark.

Presentation (19 slides, with a video of the run): https://claude.ai/artifact/EQd9Su9Q83nUjgNQpjqcVU

## Next

1. Fall detection: go limp safely when the robot falls (real robots do this)
2. A camera node: find the bottle in an image instead of reading its position from the simulator

## Setup reminders

- `uv sync --extra sim` installs mujoco. Run it again if you get "No module named mujoco" (a plain `uv sync` or `./run.sh` removes it)
- macOS: the 3D window needs `mjpython` instead of `python`
- zsh: paste only command lines, not lines starting with `#`
