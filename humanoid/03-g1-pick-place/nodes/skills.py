"""Runs skills (g1/skills.py) one at a time: the robot's hands and feet for whoever gives commands.

A command is JSON text: {"skill": "look_around"} | {"skill": "go_to", "place": "table_a"}
| {"skill": "pick", "object": 3} or {"skill": "pick", "label": "red can"}
| {"skill": "place", "target": "mark"} or {"target": {"next_to": 5}} or {"target": {"next_to": "blue bottle"}}
| {"skill": "stop"}: stop walking and moving the arm now.
Labels are looked up in the memory ("red can" -> the remembered red can's id).

Inputs:  command      JSON text, one skill (commander now, the language model in phase 5)
         base_pose    50 Hz, the robot's position: one skill update per pose
         low_state    500 Hz, only the hand's finger angles are used (grasp check)
         standing     legs still (locomotion)
         arm_status   which arm command finished (arm)
         objects      the memory (world)
Outputs: cmd_vel, arm_command   to locomotion and arm, as task did
         held                    [object id, x, y, z] when picked up, [-1, x, y, z] when put down (to world)
         arm_area                [x, y] where the arm works at a table, [nan, nan] when done (to world)
         status                  text: the skill and step now running
         result                  JSON text: {"skill", "ok", "message"} when a skill ends
"""

import json

import numpy as np
import pyarrow as pa
from dora import Node

from g1 import messages, robot, skills
from g1.simulator import yaw_of

r = skills.Robot()
skill = None
label = ""
arm_area_sent = None
node = Node()


def find(text):
    """A label like "red can" -> the id of a remembered object, or None."""
    words = text.lower().split()
    for o in r.objects:
        if all(w in (o["color"], o["kind"], f"#{o['id']}", str(o["id"])) for w in words):
            return o["id"]
    return None


def make(command):
    """JSON command -> a Skill, or a Result if it can't even start."""
    name = command.get("skill")
    if name == "look_around":
        return skills.LookAround()
    if name == "go_to":
        return skills.GoTo(command.get("place", ""))
    if name == "pick":
        object_id = command.get("object")
        if object_id is None and "label" in command:
            object_id = find(command["label"])
            if object_id is None:
                return skills.Result(False, f"I don't remember a {command['label']}; look around first")
        return skills.Pick(int(object_id))
    if name == "place":
        target = command.get("target")
        if isinstance(target, dict) and "next_to" in target:
            anchor = target["next_to"]
            anchor = find(anchor) if isinstance(anchor, str) else anchor
            if anchor is None:
                return skills.Result(False, f"I don't remember a {target['next_to']}")
            target = ("next_to", int(anchor))
        return skills.Place(target)
    return skills.Result(False, f"unknown skill {name!r}")


def finish(name, result):
    node.send_output("result", pa.array([json.dumps({"skill": name, "ok": result.ok, "message": result.message})]))
    node.log_info(f"{name}: {'OK' if result.ok else 'FAILED'}: {result.message}")


for event in node:
    if event["type"] == "STOP":
        break
    if event["type"] != "INPUT":
        continue
    name = event["id"]

    if name == "low_state":
        _, q, _, _, _ = messages.unpack_low_state(event["value"].to_numpy())
        r.hand_q = q[robot.RIGHT_HAND]
    elif name == "standing":
        r.standing = event["value"].to_numpy()[0] > 0.5
    elif name == "arm_status":
        r.arm_status = messages.unpack_arm_status(event["value"].to_numpy())
    elif name == "objects":
        r.objects = json.loads(event["value"][0].as_py())
    elif name == "command":
        command = json.loads(event["value"][0].as_py())
        if command.get("skill") == "stop":
            if skill is not None:
                finish(skill.name, skills.Result(False, "stopped"))
            skill = None
            node.send_output("cmd_vel", pa.array([0.0, 0.0, 0.0]))
            continue
        if skill is not None and not skill.done:
            finish(command.get("skill", "?"), skills.Result(False, f"busy with {skill.name}; wait for its result or send stop"))
            continue
        made = make(command)
        if isinstance(made, skills.Result):
            finish(command.get("skill", "?"), made)
        else:
            skill = made
    elif name == "base_pose":
        time, position, quat = messages.unpack_pose(event["value"].to_numpy())
        r.time, r.position, r.heading = time, position, yaw_of(quat)
        if skill is None or skill.done:
            node.send_output("cmd_vel", pa.array([0.0, 0.0, 0.0]))
            continue
        held_before = r.held
        cmd_vel, command = skill.update(r)
        node.send_output("cmd_vel", pa.array(cmd_vel))
        if command is not None:
            node.send_output("arm_command", pa.array(command))
        area = None if r.arm_area is None else tuple(float(v) for v in r.arm_area[:2])
        if area != arm_area_sent:
            node.send_output("arm_area", pa.array([*(area or (np.nan, np.nan))]))
            arm_area_sent = area
        if r.held != held_before:
            where = r.released_at if r.held is None and r.released_at is not None else [np.nan] * 3
            node.send_output("held", pa.array([float(r.held if r.held is not None else -1), *where]))
            r.released_at = None
        if skill.label != label:
            label = skill.label
            node.send_output("status", pa.array([f"{skill.name}: {label}"]))
        if skill.done:
            finish(skill.name, skill.result)
