# Tell the robot, phase 0: own the starting point

Plan: https://claude.ai/artifact/DdS7HDAMjuHDou7pkupu7K

Goal of this phase: understand every message in the G1 pick-and-place demo before changing it,
and add a dashboard that shows what the robot does. All work is in
`humanoid/03-g1-pick-place/`, branch `tell-the-robot`.

## What was added

| File | What it is |
| --- | --- |
| `g1/frames.py` | `to_body` and `to_world`: turn a point between the room and the robot frame. `Mission.to_body` now uses it |
| `nodes/dashboard.py` | Rerun dashboard, a listener only: 3D map of tables, mark, bottle, robot heading and path, hand targets; mission log; walk command and standing charts |
| `nodes/task.py` | sends a new output `step`: the step label as text. First non-number message in this dataflow |
| `dataflow.yml` | the `dashboard` node, and `step` on `task` |

Checked: `tests/test_mission.py` PASS (0.9 cm), `dora validate` OK, full dataflow run done in
42.4 s, 1.2 cm from the mark, 99% real time with the dashboard attached.

## Run it

```bash
./run.sh g1-pick-place                 # MuJoCo window + Rerun window
VIEWER=save ./run.sh g1-pick-place --stop-after 55s   # writes humanoid/03-g1-pick-place/out/g1.rrd
uv run rerun humanoid/03-g1-pick-place/out/g1.rrd      # open a saved run, scrub the timeline
```

## Setup fix on this Mac

`sim` crashed with `Library not loaded: @executable_path/../lib/libpython3.13.dylib`.
MuJoCo's `mjpython` looks for the Python library next to the venv, uv keeps it elsewhere.
Fix, once per fresh `.venv` (local only, nothing in the repo):

```bash
ln -sf "$(uv run --no-sync python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')/libpython3.13.dylib" .venv/lib/
```

## Exercises

Do these in order, write the answers below each one in your own words.

1. Draw the dataflow from memory: 5 nodes, every arrow, every message name, and its rate.
   Check against the comment at the top of `dataflow.yml`.

2. In `g1/messages.py`, `low_state` is one flat array of float64. How many numbers is it for
   43 joints? Which index holds the IMU gyro z (turn rate)?

3. `Mission` computes each step's target when the step starts, not when the mission starts.
   What would go wrong if it computed all 16 targets at t = 0? Test your answer: in the
   MuJoCo window, push the bottle 5 cm while the robot walks to table A.

4. Push the robot sideways (double-click the pelvis, Ctrl + right-drag) while it walks.
   Watch `cmd_vel` in the dashboard. What does the navigator do?

5. Knock the bottle out of the hand right after "close the hand". What does the mission do?
   What would it need to notice the failure? (This is phase 4.)

6. In the dashboard, the pink hand target is computed with `frames.to_world` from a target in
   the body frame. Why does `to_body` ignore the pelvis tilt, and when could that cause
   a centimeter-sized error?

7. `standing` and `cmd_vel` run at 50 Hz, `arm_command` only once per step. Why is the
   difference right for each one?

## What phase 1 needs from this

- The robot state that a camera node would need to render the scene is the full MuJoCo
  `qpos`, joints plus the bottle's free joint. `sim` has it, nothing sends it yet.
- Every message carries a time. The camera images will need the same.

## Finding: the near-fall at table B (fixed)

Seen live: after putting the bottle down, the robot almost fell and kept stepping.

Measured with a headless trace (contacts, pelvis tilt and height, `cmd_vel` every 20 ms):

- In the walking pose (elbows at 1.2 rad) the hands reach forward. At the 15 cm standing
  distance from the table edge, that pose puts the right hand inside the table top
- The old order lowered the arm ("arm down") while still at table B. The fingers
  (`right_hand_middle_1_link`) hit the table and pushed the robot 10 cm back with
  `cmd_vel` = 0: tilt 16.4°, pelvis down from 0.78 to 0.70 m
- "Step back" then started from a stumble and walked to a goal computed from the pushed position

Tried first: step back with the hand raised, then arm down. Failed: with the hand in the
ready pose the walking policy overshoots by 16 cm, then steps in place and ignores the
small correction commands. The policy is sensitive to arm pose, as the README says.

Fix in `g1/mission.py`: at table B, move the empty hand into the carry pose (the pose that
already walks well at table A), step back, then lower the arm. 17 steps now.
Result: max tilt 7.5° (normal walking), pelvis never below 0.772 m, no right-hand contact,
test PASS 0.9 cm, dora run 0.8 cm.

Still open: the left hand's fingers graze the table edges at both tables, harmless today.
Phase 4 skills should check arm poses against the table, not only the hand target.

## Finding: the wobbly step back (tuned)

Seen live after the fix above: walking backward at the end still looked close to falling.

Measured in a live dora run (temporary CSV recorder on `base_pose`) and headless:
final step back max tilt 6.9°, pelvis never below 0.770 m. Same range as forward walking
(7.0–7.1°). Not a loss of balance: the walking policy rocks 5–6° side to side every step,
and the robot backed up at 0.37 m/s, the same cap as forward.

Speed sweep, final step back (headless): 0.40 m/s → 7.5°, 0.25 → 7.2°, 0.15 → 7.0°, 0.10 → 6.9°.
The rocking belongs to the gait; speed changes it little.

Change in `g1/navigation.py`: `MAX_BACKWARD = 0.15` m/s, separate from `MAX_FORWARD`.
Mission 48.3 s instead of 45.1 s, test PASS 0.9 cm.
The dashboard now has a `balance` chart (pelvis tilt from `base_pose`), so you can see this yourself.
