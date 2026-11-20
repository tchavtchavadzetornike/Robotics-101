"""dora node: the pick-and-place mission (tasks/pick_and_place/mission.py).

The mission is written against the robot's interfaces; g1.dora.DoraG1 gives it
the robot behind the sim node's messages, so the same code runs here unchanged.

Inputs:  base_pose, robot_state, scan, object_pose (bottle)   from sim
Outputs: cmd_vel, upper_body, hands                            to sim
"""

import logging

from dora import Node
from g1.dora import DoraG1

from tasks.pick_and_place.mission import PickAndPlace

logging.basicConfig(level=logging.INFO, format="%(message)s")  # the mission reports each step
robot = DoraG1(Node(), objects=["bottle"])
mission = PickAndPlace(robot)
while not mission.done:
    mission.update()
    robot.step()
print("mission finished" if mission.failed is None else f"mission stopped: {mission.failed}", flush=True)
robot.move(0.0, 0.0, 0.0)
robot.step()
