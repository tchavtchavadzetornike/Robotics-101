"""The crowd-crossing task's logic, on the robot's interfaces only.

Get from START to GOAL as fast as is safe: the walking skill (g1.navigator:
lidar -> people -> plan -> safety) picks every command. The arms and hands
keep the pose they start with.

    mission = Crossing(robot)          # robot: anything with Locomotion, State, Lidar
    while not mission.done:
        mission.update(); robot.step()
"""

from g1.config import SCAN_DT
from g1.interfaces import Lidar, Locomotion, State
from g1.navigator import Navigator

from .layout import GOAL

STILL = 0.05  # m/s: arrived means standing still at the goal


class Crossing:
    def __init__(self, robot: Locomotion | State | Lidar, goal=GOAL):
        self.robot = robot
        self.navigator = Navigator(goal)
        self.next_scan = robot.time()
        self.done = False
        self.failed = None  # this task has no reason to give up early

    def update(self) -> None:
        """One control step: a new walking command for the robot."""
        now = self.robot.time()
        scan = None
        if now >= self.next_scan - 1e-9:  # the lidar turns 10 times per second
            self.next_scan += SCAN_DT
            scan = self.robot.scan()
        cmd = self.navigator.update(self.robot.pose(), scan, dt=self.robot.dt)
        self.robot.move(*cmd)
        self.done = self.navigator.arrived and not any(cmd[:2]) and self.robot.speed() < STILL


def succeeded(robot, mission) -> bool:
    """At the goal and standing still (collisions and falls end the episode in the environment)."""
    return mission.done
