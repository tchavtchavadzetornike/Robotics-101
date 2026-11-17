"""Run skills headless: physics, walking, arm, head camera, detector and memory, no window.

The same parts as the dora graph, in one loop (like test_mission.py), so a skill
can be checked in a minute without opening anything:

    bench = Bench(seed=3)            # random room 3 (g1/scenes.py)
    bench.run(skills.LookAround())   # -> Result(ok, message)
    bench.find("red", "can")         # -> its id in the memory
    bench.run(skills.Pick(4))
    bench.run(skills.Place("mark"))
    bench.truth("obj_2")             # where an object really is

The detector runs 1.5 times a second whenever the legs stand still, as in dora.
"""

import sys
from pathlib import Path

import numpy as np

LESSON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LESSON))

from g1 import messages, objects, robot, scenes, skills, world  # noqa: E402
from g1.arm import ArmController  # noqa: E402
from g1.camera import Camera  # noqa: E402
from g1.locomotion import POLICY_DT, Locomotion  # noqa: E402
from g1.simulator import Simulator, yaw_of  # noqa: E402

DETECT_DT = 1 / 1.5
_detector = None


def detector():
    global _detector
    if _detector is None:
        from g1.detector import Detector

        _detector = Detector()
    return _detector


class Bench:
    def __init__(self, seed=None, start=(0.0, 0.0, 0.0)):
        self.scene = scenes.make_scene(seed) if seed is not None else None
        path = str(scenes.write_scene(self.scene, LESSON / "robot")) if self.scene else str(LESSON / "robot/scene.xml")
        self.sim = Simulator(path, start=start)
        self.locomotion = Locomotion(str(LESSON / "policies/g1_walk.onnx"))
        self.arm = ArmController(str(LESSON / "robot/g1_with_hands.xml"))
        self.camera = Camera(path, "head_cam", 640, 480)
        self.memory = world.World()
        self.robot = skills.Robot()
        self.arm_command_id = None
        self.next_control = self.next_detect = 0.0
        self.log = []

    def observe(self):
        r, sim = self.robot, self.sim
        base, quat = sim.base_pose()
        q, _ = sim.joint_state()
        r.time = sim.time
        r.position, r.heading = base, yaw_of(quat)
        r.standing = self.locomotion.is_standing(sim.time)
        r.hand_q = q[robot.RIGHT_HAND]
        r.objects = self.memory.as_list()

    def detect(self):
        c = self.camera
        c.set_state(self.sim.world_state())
        rgb, depth = c.color(), c.depth()
        position, rotation = c.pose()
        seen = objects.look(detector(), rgb, depth, c.fx, c.fy, c.cx, c.cy, position, rotation)
        self.memory.update(self.sim.time, seen, lambda p: world.should_see(p, depth, c.fx, c.fy, c.cx, c.cy, position, rotation))

    def run(self, skill, limit=180.0, verbose=True, on_step=None):
        """Run one skill to its end. on_step(label, bench) is called when a step starts."""
        sim, r = self.sim, self.robot
        start, label = sim.time, ""
        while not skill.done and sim.time - start < limit:
            t = sim.time
            if t >= self.next_control - 1e-9:
                self.next_control += POLICY_DT
                self.observe()
                held_before = r.held
                cmd_vel, command = skill.update(r)
                self.memory.arm_working(r.arm_area)
                if r.held != held_before:
                    if r.held is not None:
                        self.memory.hold(r.held)
                    else:
                        self.memory.release(r.released_at)
                        r.released_at = None
                if skill.label != label:
                    label = skill.label
                    self.log.append(f"t={t:6.2f}  {skill.name}: {label}")
                    if verbose:
                        print(self.log[-1], flush=True)
                    if on_step is not None:
                        on_step(label, self)
                if command is not None:
                    cid, kind, position, yaw = messages.unpack_arm_command(command)
                    if self.arm.execute(t, kind, position, yaw):
                        self.arm_command_id = cid
                    else:
                        r.arm_status = (cid, False)
                q, dq = sim.joint_state()
                quat, gyro = sim.imu()
                sim.set_command(robot.LOWER_BODY, *self.locomotion.update(t, cmd_vel, q[robot.LEGS], dq[robot.LEGS], quat, gyro))
                sim.set_command(robot.UPPER_BODY, *self.arm.update(t, q[robot.UPPER_BODY]))
                if self.arm_command_id is not None and not self.arm.busy():
                    r.arm_status = (self.arm_command_id, True)
                    self.arm_command_id = None
            if self.robot.standing and t >= self.next_detect - 1e-9:
                self.next_detect = t + DETECT_DT
                self.detect()
            sim.step()
            if sim.data.qpos[2] < 0.5:
                return skills.Result(False, "the robot fell")
        self.observe()
        result = skill.result or skills.Result(False, f"{skill.name} did not finish in {limit:.0f} s")
        if verbose:
            print(f"t={sim.time:6.2f}  {skill.name}: {'OK' if result.ok else 'FAILED'}: {result.message}", flush=True)
        return result

    def find(self, color, kind):
        found = self.memory.find(kind=kind, color=color)
        return found[0].id if found else None

    def knock_away(self, name, to=(-3.0, 3.0)):
        """Take an object away, as if someone pulled it out of the robot's hand."""
        import mujoco

        address = self.sim.model.jnt_qposadr[self.sim.model.joint(name).id]
        self.sim.data.qpos[address : address + 3] = (*to, 0.2)
        self.sim.data.qvel[self.sim.model.jnt_dofadr[self.sim.model.joint(name).id] : self.sim.model.jnt_dofadr[self.sim.model.joint(name).id] + 6] = 0
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def truth(self, name):
        return self.sim.data.xpos[self.sim.model.body(name).id].copy()
