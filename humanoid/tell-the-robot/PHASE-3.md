# Tell the robot, phase 3: many objects, found by name, remembered

Goal: random rooms with 3 to 6 objects; the robot finds every one, knows what it is and what
color, remembers where it is with an id that never changes, and forgets it only when it is
really gone. All code in `humanoid/03-g1-pick-place/`.

## What was added

| File | What it is |
| --- | --- |
| `g1/scenes.py` | random rooms from a seed: the blue target bottle on table A plus 2–5 bottles, cans, boxes in 4 colors |
| `g1/camera.py` | `segmentation()`: which object each pixel shows, the truth for scoring |
| `g1/detector.py` | OWLv2, open-vocabulary detector, shape and name prompts, boxes merged (NMS) |
| `g1/objects.py` | box → points above the table → kind from height, color from hue, center; duplicates merged in 3D |
| `g1/world.py` | the memory: matching, ids, votes for kind and color, forgetting only what should be visible |
| `g1/survey.py` | look around: second spot for table A, survey spot for table A and table B |
| `nodes/detector.py`, `nodes/world.py` | the dora nodes; detector 1.5 Hz, only while standing |
| `nodes/dashboard.py` | remembered objects as labelled boxes, a `memory` text panel |
| `tests/test_world.py` | 7 rules of the memory, no simulator |
| `spikes/detector_benchmark.py`, `objects_benchmark.py`, `world_eval.py` | the measurements below |
| `light.sh` | runs heavy jobs gently, with a heat watchdog (see the end) |

Run a random room: `SEED=7 ./run.sh g1-pick-place`.

## The detector: measured, not assumed

40 images (10 rooms × 4 places), 96 visible objects, truth from the segmentation image.

| Detector, prompts | Found at all | Kind right | Time per image |
| --- | --- | --- | --- |
| YOLO-World v2 s, names | 46% | bottles only | 12 ms |
| YOLO-World v2 s, shapes | 72% | rarely: cans called bottles | 11 ms |
| OWLv2 base, names | 74% | 66 of 71 | 330 ms |
| **OWLv2 base, shapes + names** | **97%** | boxes yes, every bottle a "cylinder" | 330 ms |

Decision: OWLv2 for *where*, geometry and color for *what*.

**Why the detectors struggle:** our objects are plain, single-color shapes with no labels or texture.
Detectors learn from photos, where a can has a printed label. Sim-to-real in reverse: the simulation
is too clean for a model trained on reality.

## The whole chain against the truth

Detector boxes → `objects.look` (60 images incl. the two survey views):

| Measure | Result |
| --- | --- |
| Found | 145 of 147 |
| Color right | 145 of 145 |
| Kind | right whenever the top is in the image; "unknown" otherwise, never wrong |
| Position error | median 0.13 cm, worst 1.9 cm |
| False objects | 0 (were 28, all duplicates: one box for the bottle, one for its top) |

## The memory, 20 random rooms

Per room: look around, compare with the truth, walk around for 60 s, remove one object the camera
can see the spot of, look again.

| Check | Result |
| --- | --- |
| Rooms passing every check | **20 of 20** |
| Every object remembered, kind and color right | 20 of 20 rooms |
| Worst position error per room | 0.1–1.6 cm |
| Ids unchanged after 60 s of walking | 20 of 20 |
| Removed object forgotten | 20 of 20 |
| Time per room | 14–15 s, under `light.sh`, 15 s pause between rooms |
| Mac | peak 0.9 cores for the job, no thermal warning in 187 samples |

Before the second viewpoint for table A: 19 of 20 (seed 19, the hidden green can).

## Things learned the hard way

- **Measure the detector on your data before building on it.** Benchmarks on photos said nothing
  about plain shapes. Two hours of measuring changed the design.
- **Use each part for what it is good at.** OWLv2 localizes well and names badly here; height and
  hue name perfectly once you have the points.
- **The camera can't see far tops.** Tilted 40° down, it sees a 1 m high top only within 1.39 m.
  From the start, far objects are "unknown". Hence a survey spot.
- **One viewpoint is not enough.** Seed 19: the green can stood exactly behind the blue bottle
  (bearings 53.4° and 53.9°). No threshold fixes that; a second viewpoint does.
- **Forgetting needs care.** "Not detected" is not "gone". Only count a miss when the camera should
  see the spot: in the image, in range, and not covered by something nearer, checked at 5 points up
  the object's axis. Seed 2: the removed bottle's spot was behind the blue bottle, so keeping it was right.
- **dora nodes leave the process group.** The first watchdog measured 0% CPU while the Mac ran at
  load 20. It now follows the process tree.
- **A slow node must drop, not queue.** The detector takes 0.3 s; with `queue_size` 2 on images it
  always works on the newest one. `standing` at 50 Hz needs a queue that covers 0.6 s, or dora
  warns on every message.

## Running gently: `light.sh`

`humanoid/tell-the-robot/light.sh <command>` runs any heavy job at low priority with 4 threads
and watches it every 3 s: it stops the whole job at once if macOS reports a thermal or performance
warning or limits CPU speed, or if the job uses more than 6 cores for 60 s. Log in `/tmp/light-*.log`
with the 5 busiest processes every 30 s. Containers would not help on a Mac: Podman and Docker run
in a VM on the same chip, without the GPU.

Live run, room 7: the job used 1.7–1.9 cores, load 5–7, no heat warning. `sim` 1.1 cores, `camera`
0.3, the detector 6% CPU (its work runs on the GPU).

## Exercises

1. In `g1/world.py`, matching is closest-pair-first. Build two objects 10 cm apart and two sightings
   where closest-first gives the wrong pairing. How would the Hungarian algorithm pair them?
2. Why do kind and color use votes (`Counter`) and position a moving average? What would go wrong
   the other way round?
3. Why does `should_see` treat a covered spot as "can't know" instead of "missing"? Construct a
   case where the other choice loses a real object.
4. Change `SAME_OBJECT` in `g1/objects.py` from 5 cm to 15 cm. Which rooms break, and why?
5. Run `detector_benchmark.py 10 owl names` and `... shapes`. Why does adding the word "cylinder"
   make OWLv2 find more objects but name bottles worse?
6. The memory is only updated while the robot stands. What goes wrong when the robot carries an
   object to the other table? What does phase 4 need to tell the memory?

## Next: phase 4, skills

`go_to`, `pick(id)`, `place(target)`, `look_around` with a check before and a check after, so the
language model can call them. The memory needs to know which object is in the hand.
