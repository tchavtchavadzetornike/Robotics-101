# Tell the robot, phase 4: skills

Goal: replace the fixed 17-step script with reusable skills that a language model can call in
phase 5: each takes a parameter, checks before, during and after, and ends with an honest result.
All code in `humanoid/03-g1-pick-place/`.

```bash
./run.sh g1-skills                                     # look around, pick the blue bottle, put it on the mark
SEED=3 ROBOT_SCRIPT="look_around; pick red can; place table_a" ./run.sh g1-skills
```

## What was added

| File | What it is |
| --- | --- |
| `g1/skills.py` | `LookAround`, `GoTo(place)`, `Pick(id)`, `Place(target)`; steps, retries, retreats, `Result` |
| `g1/reach.py` | from which table side an object can be grasped, where an object can be put down, what is in the hand's way |
| `g1/paths.py` | walking around tables: grown obstacles, visibility graph, Dijkstra |
| `g1/world.py` | the held object follows the hand; no misses counted next to the working arm |
| `g1/navigation.py` | `MAX_BACKWARD` stays, `MIN_SIDEWAYS` 0.10 m/s for small sideways corrections |
| `nodes/skills.py` | runs one skill at a time from JSON commands, sends results; replaces `task` |
| `nodes/commander.py` | sends a script of skills; phase 5 replaces it with the language model |
| `skills.yml` | the skills graph; `dataflow.yml` keeps the original mission |
| `tests/harness.py` | `Bench`: skills headless with physics, arm, camera, detector, memory |
| `tests/test_reach.py`, `tests/test_paths.py` | pure tests, 22 with `test_world.py` |
| `spikes/skills_demo.py`, `skills_eval.py`, `skills_failures.py` | the measurements below |

## The skills

| Skill | Checks before | Checks during | Check after |
| --- | --- | --- | --- |
| `look_around` | | | |
| `go_to(place)` | the place exists | | arrived |
| `pick(id)` | hand empty, object known, a table side within reach, nothing in the hand's way | still there up close, in reach from where the robot really stands, fingers not closed on nothing (twice) | |
| `place(target)` | holding something, a free spot within reach | still in the hand (twice), spot in reach from where the robot really stands | seen at the spot *after* letting go, neighbors didn't move |

A skill that can't line up or reach tries the next table side or spot (up to 3). One that fails
near a table retreats: stand still, hand back, step back, only then arm down. One that fails while
still holding the object keeps holding it and says so.

## Measured numbers

| Measure | Result |
| --- | --- |
| Deepest grasp from a table edge | 18 cm works, 20 cm the legs hit the table: `MAX_DEPTH` 17 cm |
| Fingers closed on an object vs on nothing | 0.74 rad vs 0.03–0.18 rad from the closed pose: `GRIP_EMPTY` 0.4 |
| Original room: look, pick, place on the mark | 0.9 cm from the mark |
| 10 random rooms, every kind | **17 of 18** moves right, 0.6–1.9 cm from the spot, upright; 11 objects refused with a reason |
| Per kind | bottles 8/8, cans 5/5, boxes 4/5 |
| Objects knocked over by accident | 0 (checked against the simulator's truth after every step) |
| Deliberate failures (pulled out while closing, pulled out while carrying, unreachable, unknown id) | 4 of 4: right message, robot standing, arm down |
| Live dora run, `./run.sh g1-skills` | 4 of 4 skills OK, 100% real time, under 2 cores, no heat warning |
| Original mission (`test_mission.py`) | still passes, 0.5 cm |

The one failure: a box put down tipped over (8.9 cm off, on its side). The robot reported it
correctly: "not where I put it: I don't see it there".

## Things learned the hard way

Most of this phase was finding out why objects got knocked over. A check against the simulator's
truth after every step was what found each cause; the robot's own reports only found some.

- **Check the whole sweep, not the target point.** The fingers reach 12 cm past the grasp point;
  pulling the hand away sideways pushed a box 6 cm. Pick sweeps sideways; place now lowers from
  above and pulls straight back.
- **The fingers forward over the table.** After placing, the arm swung into the carry pose with the
  fingertips 32 cm ahead and knocked a can off. Stepping back now uses the tuck pose: hand 5 cm
  nearer, fingers turned 90° along the table edge.
- **The walking policy dislikes new arm poses.** Walking backward with the hand raised got stuck;
  carrying an object tucked in drifted 7–12 cm and fell twice in 10 rooms. Carrying stays in the
  pose the policy was trained near; only an empty hand gets tucked.
- **Small sideways commands do nothing.** 7 cm/s sideways: the robot steps in place. The minimum
  is now 10 cm/s, a stuck robot within 10 cm counts as arrived, and the arm adapts to where it
  really stands.
- **Don't cut the corner into a table.** Passing the approach point within 15 cm let the robot
  come in diagonally, its wrist sweeping the table edge. The approach point must be reached.
- **A check must not trust its own note.** "Is it where I put it?" read the memory, which had just
  been told where it was put. It now needs a sighting after letting go.
- **A retreat can do damage.** Opening the hand away from the table dropped the bottle on the
  floor; moving the arm while still stepping toppled the robot. Retreats stand still first, and
  keep holding what is still in the hand.
- **The robot's own hand blinds the memory.** Next to the working arm, objects are missed and
  would be forgotten. The memory doesn't count misses there.

## Still open

- Boxes tip over sometimes when put down (1 of 5).
- "but #7 is gone" can appear for a short-lived duplicate in the memory: a false alarm.
- 11 of 29 objects in random rooms are out of reach from every side: the arm reaches 17 cm deep.
  Pushing an object closer, or a second arm, would be the next step for those.
- The detector runs only while standing, so the memory doesn't see the held object while walking.

## Exercises

1. In `g1/paths.py`, why grow the tables by the robot's radius instead of checking the robot's
   shape against the table directly?
2. Why does `reach.HAND_SWEEP` reach 16 cm *behind* the grasp spot?
3. `Pick` checks the grip twice. What happens between the two checks that the first can't catch?
4. Find in `skills.py` every place a step uses where the robot *really* stands instead of where it
   planned to stand. What would go wrong in each if it used the plan?
5. A skill result is plain English. Write the three results you would want to read if you were the
   language model in phase 5 and the robot dropped the bottle on the way to the mark.
6. Run `skills_failures.py`, then add a fifth case: an object pushed 10 cm while the robot walks
   to it. What should `pick` do?

## Next: phase 5, the brain

The `commander` node is replaced by an `agent` node: your sentence goes to Claude with the skills
as tools; Claude calls them one at a time, reads each result, and decides the next. The results
above are written for it.
