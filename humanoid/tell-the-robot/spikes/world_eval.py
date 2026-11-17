"""Phase 3 evaluation: does the memory hold every object, right, after looking around?

Per random scene, headless:
  1. look around (g1/survey.py): walk to the survey spot, face table A, face table B
  2. check the memory against the truth: every object, kind, color, < 2 cm, no extras
  3. keep moving for 60 s (walk toward table A, back, look again), detector running
  4. remove one object on table A, look at table A again: it must be forgotten
  5. at the end: every other object still has the id it got in step 1

The detector runs only while standing still at a look step (3 images per second of robot time).

    humanoid/tell-the-robot/light.sh uv run python humanoid/tell-the-robot/spikes/world_eval.py [scenes] [first seed] [pause s]

The pause between scenes (default 20 s) lets the Mac cool down.
"""

import sys
import time as wall
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
LESSON = HERE.parents[1] / "03-g1-pick-place"
sys.path.insert(0, str(LESSON))

from g1 import objects, robot, scenes, survey, world  # noqa: E402
from g1.camera import Camera  # noqa: E402
from g1.detector import Detector  # noqa: E402
from g1.locomotion import POLICY_DT, Locomotion  # noqa: E402
from g1.simulator import Simulator, yaw_of  # noqa: E402

DETECT_DT = 1 / 3
MAX_ERROR = 0.02
FAR_AWAY = (-3.0, 3.0)  # where a removed object goes


class Run:
    def __init__(self, path, detector):
        self.sim = Simulator(path)
        self.locomotion = Locomotion(str(LESSON / "policies/g1_walk.onnx"))
        self.camera = Camera(path, "head_cam", 640, 480)
        self.detector = detector
        self.world = world.World()
        self.next_control = self.next_detect = 0.0
        self.arm_target = None

    def run(self, plan, limit=60.0):
        sim, start = self.sim, self.sim.time
        while not plan.finished and sim.time - start < limit:
            t = sim.time
            if t >= self.next_control - 1e-9:
                self.next_control += POLICY_DT
                q, dq = sim.joint_state()
                quat, gyro = sim.imu()
                base, base_quat = sim.base_pose()
                pose = (base[0], base[1], yaw_of(base_quat))
                cmd_vel = plan.update(t, pose, self.locomotion.is_standing(t))
                sim.set_command(robot.LOWER_BODY, *self.locomotion.update(t, cmd_vel, q[robot.LEGS], dq[robot.LEGS], quat, gyro))
            if plan.looking and t >= self.next_detect - 1e-9:
                self.next_detect = t + DETECT_DT
                self.detect(t)
            sim.step()
        return plan.finished

    def detect(self, t):
        c = self.camera
        c.set_state(self.sim.world_state())
        rgb, depth = c.color(), c.depth()
        position, rotation = c.pose()
        seen = objects.look(self.detector, rgb, depth, c.fx, c.fy, c.cx, c.cy, position, rotation)
        self.world.update(t, seen, lambda p: world.should_see(p, depth, c.fx, c.fy, c.cx, c.cy, position, rotation))

    def truth(self, scene):
        return {o.name: self.sim.data.xpos[self.sim.model.body(o.name).id].copy() for o in scene.objects}


def compare(memory, scene, truth, gone=()):
    """Match remembered objects to the truth. Returns (problems, {object name: id})."""
    problems, ids = [], {}
    remembered = list(memory.objects.values())
    for o in scene.objects:
        if o.name in gone:
            continue
        near = [r for r in remembered if np.hypot(*(r.position[:2] - truth[o.name][:2])) < 0.06]
        if not near:
            problems.append(f"{o.label} missing")
            continue
        r = min(near, key=lambda r: np.hypot(*(r.position[:2] - truth[o.name][:2])))
        error = np.hypot(*(r.position[:2] - truth[o.name][:2]))
        ids[o.name] = r.id
        if r.kind != o.kind or r.color != o.color:
            problems.append(f"{o.label} remembered as {r.label}")
        if error > MAX_ERROR:
            problems.append(f"{o.label} {100 * error:.1f} cm off")
    extra = len(remembered) - len(ids)
    if extra:
        problems.append(f"{extra} extra object(s): " + ", ".join(r.label for r in remembered if r.id not in ids.values()))
    return problems, ids


def remove_one_in_view(run, scene, known):
    """Take away an object whose spot the camera can then see (else it can't know it's gone)."""
    sim, c = run.sim, run.camera
    for o in reversed(scene.objects):
        if o.name not in known:
            continue
        address = sim.model.jnt_qposadr[sim.model.joint(o.name).id]
        before = sim.data.qpos[address : address + 7].copy()
        sim.data.qpos[address : address + 2] = FAR_AWAY
        mujoco.mj_forward(sim.model, sim.data)
        c.set_state(sim.world_state())
        position, rotation = c.pose()
        if world.should_see(o.position, c.depth(), c.fx, c.fy, c.cx, c.cy, position, rotation):
            return o.name
        sim.data.qpos[address : address + 7] = before
        mujoco.mj_forward(sim.model, sim.data)
    return None


def evaluate(seed, detector):
    scene = scenes.make_scene(seed)
    path = str(scenes.write_scene(scene, LESSON / "robot"))
    run = Run(path, detector)
    report = {"seed": seed, "objects": len(scene.objects)}

    if not run.run(survey.Plan(survey.look_around_plan())):
        return {**report, "ok": False, "problems": ["look around did not finish"]}
    truth = run.truth(scene)
    problems, ids_first = compare(run.world, scene, truth)
    errors = [np.hypot(*(run.world.objects[i].position[:2] - truth[n][:2])) for n, i in ids_first.items()]
    report["after_look"] = problems

    x, y = survey.SURVEY_SPOT
    moving = survey.Plan([
        ("walk", (1.0, -0.25, 0.0)),
        ("walk", (x, y, survey.FACE_TABLE_B)),
        ("look", 2.0),
        ("walk", (0.4, -0.4, 0.71)),
        ("walk", (x, y, survey.FACE_TABLE_A)),
        ("look", 1.0),
    ])
    run.run(moving, limit=60.0)

    removed = remove_one_in_view(run, scene, ids_first)
    if removed:
        run.run(survey.Plan([("look", 2.0)]))
    truth = run.truth(scene)
    problems_end, ids_end = compare(run.world, scene, truth, gone={removed} if removed else set())
    changed = [n for n in ids_end if n in ids_first and ids_end[n] != ids_first[n]]
    still_there = removed is not None and any(
        np.hypot(*(r.position[:2] - scene.object(removed).position[:2])) < 0.06 for r in run.world.objects.values()
    )
    report.update(
        after_moving=problems_end,
        ids_changed=changed,
        removed=scene.object(removed).label if removed else None,
        forgotten=not still_there,
        error_cm=100 * float(np.max(errors)) if errors else 0.0,
    )
    report["ok"] = not problems and not problems_end and not changed and (removed is None or not still_there)
    return report


def main(count, first=0, pause=20.0):
    detector = Detector()
    passed = 0
    for n, seed in enumerate(range(first, first + count)):
        if n:
            wall.sleep(pause)
        start = wall.perf_counter()
        r = evaluate(seed, detector)
        passed += r["ok"]
        print(
            f"seed {seed:2d} {r['objects']} objects: {'PASS' if r['ok'] else 'FAIL'}"
            f"  worst {r.get('error_cm', 0):.1f} cm, removed {r.get('removed')} forgotten={r.get('forgotten')}"
            f"  ({wall.perf_counter() - start:.0f} s)",
            flush=True,
        )
        for key in ("problems", "after_look", "after_moving", "ids_changed"):
            if r.get(key):
                print(f"    {key}: {r[key]}", flush=True)
    print(f"\n{passed}/{count} scenes pass")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[0]) if args else 20, int(args[1]) if len(args) > 1 else 0, float(args[2]) if len(args) > 2 else 20.0)
