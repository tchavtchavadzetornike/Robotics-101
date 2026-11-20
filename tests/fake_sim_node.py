"""A stand-in for the sim node (g1/nodes/sim.py) without Isaac Sim, to check a
task's dora graph on any computer. Same inputs and outputs; the robot is the
stand-in G1 of tests/fakes.py (exact walking, real arm kinematics, no physics).

Use it in place of the sim node: path python, args tests/fake_sim_node.py,
with TASK = G1-Crowd-Crossing-v0 or G1-Pick-Place-v0 (and SEED for the crowd).
It stops after the task's time limit and prints how the episode went.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
from dora import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))
from area.crowd import Crowd  # noqa: E402
from fakes import PERSON_RADIUS, FakeG1  # noqa: E402
from g1 import joints, messages  # noqa: E402
from g1.config import SCAN_DT  # noqa: E402
from g1.planner import ROBOT_RADIUS  # noqa: E402

task = os.environ["TASK"]
crowd = None
if task == "G1-Crowd-Crossing-v0":
    from tasks.crossing import layout

    crowd = Crowd(
        layout.NUM_PEOPLE, layout.HALF_SIZE, int(os.environ.get("SEED", "0")), keep_clear=(*layout.START[:2], 2.0)
    )
    robot = FakeG1(start=layout.START, people=lambda: crowd.position)
    objects = []
else:
    from tasks.pick_and_place import layout

    robot = FakeG1(
        start=layout.START,
        tables=[layout.table_box(layout.TABLE_A), layout.table_box(layout.TABLE_B, heading=layout.HEADING_AT_B)],
        bottle=(*layout.BOTTLE_START, layout.TABLE_SIZE[2] + layout.BOTTLE_HEIGHT / 2),
        table_height=layout.TABLE_SIZE[2],
    )
    objects = ["bottle"]

node = Node()
next_scan, closest = 0.0, np.inf
while robot.time() < layout.TIME_LIMIT:
    while True:  # every command that has arrived
        event = node.next(timeout=1e-4)
        if event is None or event["type"] not in ("INPUT", "STOP"):
            break
        if event["type"] == "STOP":
            sys.exit(0)
        value = event["value"].to_numpy()
        if event["id"] == "cmd_vel":
            robot.move(*value)
        elif event["id"] == "upper_body":
            robot.set_upper_body_targets(value)
        elif event["id"] == "hands":
            robot.set_hand_targets("left", value[:7])
            robot.set_hand_targets("right", value[7:])
    t = robot.time()
    node.send_output("base_pose", pa.array(messages.pack_pose(t, *robot.pose())))
    state = messages.pack_robot_state(
        t,
        robot.pelvis_height(),
        robot.speed(),
        robot.joint_positions(joints.UPPER_BODY),
        robot.joint_positions(joints.HANDS),
    )
    node.send_output("robot_state", pa.array(state))
    for name in objects:
        node.send_output("object_pose", pa.array(messages.pack_object_pose(t, *robot.find(name))), {"name": name})
    if t >= next_scan - 1e-9:
        next_scan += SCAN_DT
        node.send_output("scan", pa.array(messages.pack_scan(t, robot.scan())))
    robot.step()
    if crowd is not None:
        crowd.step(robot.dt, np.array(robot.pose()[:2]))
        closest = min(
            closest, np.linalg.norm(crowd.position - robot.pose()[:2], axis=1).min() - PERSON_RADIUS - ROBOT_RADIUS
        )
    x, y, _ = robot.pose()
    if crowd is not None and np.hypot(layout.GOAL[0] - x, layout.GOAL[1] - y) < 0.3 and robot.speed() < 0.05:
        print(f"RESULT arrived at t={robot.time():.1f}s, closest free space {100 * closest:.0f} cm", flush=True)
        break
    if crowd is None and robot.bottle is not None and not robot.held and robot.time() > 1.0:
        off = np.hypot(robot.bottle[0] - layout.PLACE_MARK[0], robot.bottle[1] - layout.PLACE_MARK[1])
        if off < 0.03 and robot.bottle[2] > layout.TABLE_SIZE[2]:
            print(f"RESULT bottle on the mark at t={robot.time():.1f}s ({100 * off:.1f} cm off)", flush=True)
            break
