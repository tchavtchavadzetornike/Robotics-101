# Tell the robot what to do

You type a sentence, for example "Put the blue bottle on the green mark", and the Unitree G1 from
[lesson 3](../03-g1-pick-place/README.md) does it: it looks around, finds the bottle with its head
camera, walks over, picks it up, carries it to the mark and puts it down. Then it tells you, in a
sentence, what it did, or what went wrong.

Claude (Anthropic's language model) is the robot's brain. It never moves a joint. It chooses from
five skills, `look_around`, `get_objects`, `go_to`, `pick` and `place`, reads each result and
decides the next step. Everything below the brain is ordinary robot software, built in six phases
on top of lesson 3, each one measured against the simulator's truth before the next one started.

Built by Elene Kvirikashvili on lesson 3 of this repository. The plan the phases follow:
https://claude.ai/artifact/DdS7HDAMjuHDou7pkupu7K

## Run it

Two terminals, both from the top folder of the repository.

Terminal 1, the robot, its 3D window and the Rerun dashboard. Claude needs an Anthropic API key in
the environment (`ANTHROPIC_API_KEY`):

```bash
humanoid/tell-the-robot/demo.sh g1-agent --stop-after 600s
```

Terminal 2, once the windows are open, to talk to it:

```bash
cd humanoid/03-g1-pick-place && uv run --no-sync python nodes/chat.py
```

Good first sentences: `What do you see?`, `Put the blue bottle on the green mark.`,
`Put the red can on table A.` (there is none, it says so), `I'm thirsty.` Typing `stop` stops the
robot at once, without asking the model.

A room with more objects: put `SEED=6` in front of the first command (any number gives a
different random room). Skills without a model, from a script:
`SEED=3 ROBOT_SCRIPT="look_around; pick red can; place mark" ./run.sh g1-skills`.

`demo.sh` runs the demo gently: low priority, windows at 30 frames per second, camera at 10 images
per second, and it stops everything if the demo uses more than 3 CPU cores for 20 s or macOS
reports heat. `light.sh` does the same for long headless evaluations.

## What the screen shows

| Window | What you see |
| --- | --- |
| MuJoCo | the robot and the room: walking, grasping, carrying, placing |
| Rerun | the head camera with the detector's boxes, the depth image, a top-down map of the memory (every object with its id, standing or fallen over), and the conversation: your sentence, every skill Claude calls, each result, the reply |

## The six phases

| Phase | What it added | Measured result | Notes |
| --- | --- | --- | --- |
| 0 | Rerun dashboard; a safer mission: the bottle is carried close and the robot steps back before lowering the arm | the near-fall at table B is gone; 17 steps instead of 16 | [PHASE-0.md](PHASE-0.md) |
| 1 | a head camera: color and depth images, 10 per second, in its own dora node | depth within 0.04 mm of an exact ray cast; 100% real time | [PHASE-1.md](PHASE-1.md) |
| 2 | the mission runs from the camera alone: color mask, pixels to 3D points, a cylinder fit | bottle placed 1.2 to 1.6 cm from the mark (0.7 to 1.0 cm with the simulator's truth) | [PHASE-2.md](PHASE-2.md) |
| 3 | random rooms with bottles, cans and boxes; an open-vocabulary detector (OWLv2, chosen over YOLO-World by measurement); a memory with ids that don't change | 145 of 147 objects found, color always right; 20 of 20 rooms remembered and updated correctly | [PHASE-3.md](PHASE-3.md) |
| 4 | skills with checks before, during and after: path planning around tables, reach limits, retries, honest failure messages | 17 of 18 pick-and-place moves right in 10 rooms; unreachable objects refused with the reason | [PHASE-4.md](PHASE-4.md) |
| 5 | Claude as the brain: skills as tools, a safety check on every call, a chat node | 12 of 12 test sentences end correctly, from direct orders to ambiguous ones and deliberate sabotage | [PHASE-5.md](PHASE-5.md) |

Every phase note has what was added, the measurements, the things learned the hard way, and
exercises.

## How a sentence becomes movement

```
you type a sentence
  -> chat node -> agent node: Claude reads it with the 5 skill descriptions
       -> get_objects: the memory as text, each object with whether the hand can reach it
       -> pick #1: checked first (does #1 exist? is the hand empty?), then run by the skills node
            -> walk (path planner, locomotion policy), arm (inverse kinematics), hand (grip check)
            -> result: "holding #1 blue bottle", or why not
       -> place mark -> result: "put #1 blue bottle on the mark", checked by a fresh camera look
  -> Claude answers in a sentence
```

Below the brain, every result comes from what the camera sees, not from the simulator. The
simulator's truth is used only to judge the evaluations.

## What changed in lesson 3

All code lives in `humanoid/03-g1-pick-place/`. The original mission still runs
(`./run.sh g1-pick-place`, `tests/test_mission.py` passes), with three differences from the
lesson 3 README:

- 17 steps: after placing, the hand goes back to the carry pose, the robot steps back, then the
  arm goes down (lowering it at the table hit the table)
- `task` gets `bottle_pose` from the camera (`perception` node), not from `sim`
- `dataflow.yml` has 9 nodes: `camera`, `perception`, `detector`, `world` and `dashboard` were added

New files:

| File | What it is |
| --- | --- |
| `g1/camera.py`, `g1/frames.py`, `g1/perception.py` | rendering, camera geometry, the bottle from the images |
| `g1/detector.py`, `g1/objects.py`, `g1/world.py`, `g1/survey.py` | finding objects, naming them, remembering them, looking around |
| `g1/scenes.py` | random rooms (`SEED=<n>`) |
| `g1/paths.py`, `g1/reach.py`, `g1/skills.py` | path planning, what the hand can reach, the skills |
| `g1/agent.py` | the system prompt, the tools, the safety check, the conversation loop |
| `nodes/camera.py`, `perception.py`, `detector.py`, `world.py`, `skills.py`, `agent.py`, `chat.py`, `commander.py`, `dashboard.py` | the dora nodes |
| `skills.yml`, `agent.yml` | the graphs for `./run.sh g1-skills` and `./run.sh g1-agent` |
| `tests/test_camera.py`, `test_world.py`, `test_paths.py`, `test_reach.py`, `test_agent.py`, `harness.py` | 34 tests without a simulator, plus a headless bench for skills |

`humanoid/tell-the-robot/spikes/` holds the measurements and evaluations behind every number above
(`agent_eval.py` costs a few cents per run: it calls the real model).

## Tests

```bash
cd humanoid/03-g1-pick-place
uv run --with pytest pytest tests/test_camera.py tests/test_world.py tests/test_paths.py tests/test_reach.py tests/test_agent.py
uv run python tests/test_mission.py
```

## Still open

- picking can still nudge a neighboring object about 5 cm
- an object hidden behind another can be reported as "can't see it anymore"
- in one live run, looking around missed a yellow can that the headless evaluation found
- phase 6: voice (push-to-talk, speech to text on the Mac, spoken replies)
