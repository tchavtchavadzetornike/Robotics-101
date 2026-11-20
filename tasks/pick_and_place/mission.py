"""The pick-and-place task's logic, on the robot's interfaces only.

Walk to table A, pick up the bottle with the right hand, walk to table B,
put the bottle on the green mark. Every step uses a shared skill:
walking (g1.navigator: plan + safety, docking at the tables) and the arm and
hand (g1.manipulation: IK + smooth moves). Where to stand at a table comes
from the hand's geometry (g1.kinematics), not from hand-picked numbers.

    mission = PickAndPlace(robot)      # robot: g1.interfaces.G1
    while not mission.done:
        mission.update(); robot.step()
"""

import logging
import math

import numpy as np
from g1.config import SCAN_DT
from g1.interfaces import G1
from g1.kinematics import ArmIK, grasp_point
from g1.manipulation import Manipulator
from g1.navigator import Navigator

from .layout import (
    BOTTLE_HEIGHT,
    BOTTLE_START,
    HEADING_AT_A,
    HEADING_AT_B,
    PLACE_MARK,
    TABLE_A,
    TABLE_B,
    TABLE_SIZE,
    table_box,
)
from .step import Step

logger = logging.getLogger(__name__)

SIDE = "right"  # the hand that grasps

APPROACH = 0.8  # m: walk to this far before the docking spot, then dock slowly
PRE_GRASP = 0.12  # m: the open hand waits this far behind the grasp, then moves in
LIFT = 0.10  # m: lift the bottle this high off the table
WALK_ABOVE = 0.05  # m: while walking the hands wait this far above grasp height (the most
# the G1's arm reaches with the hand level when also PRE_GRASP back; checked by IK)
PLACE_ABOVE = 0.01  # m: release the bottle this far above the table (it settles)


class PickAndPlace:
    def __init__(self, robot: G1):
        self.robot = robot
        static_map = [table_box(TABLE_A), table_box(TABLE_B, heading=HEADING_AT_B)]
        self.navigator = Navigator(static_map=static_map)
        self.hands = Manipulator(robot)
        ik = ArmIK(SIDE)
        palm, self.palm_rotation = ik.palm_pose(np.zeros(7))
        # where the object's center is, relative to the pelvis, when the arm is
        # in its natural pose (elbow bent, forearm forward): stand so the
        # bottle is there, then the arm only makes small moves
        self.reach = palm + self.palm_rotation @ grasp_point(SIDE)
        self.grasp_offset = grasp_point(SIDE)
        # while walking, both hands wait above table height in the pre-grasp
        # spot, so they can't catch a table's edge or knock the bottle (NVIDIA's
        # default G1 pose holds the forearms forward at table height)
        self.walk_pose = {}
        for side in ("left", "right"):
            side_palm, rotation = ArmIK(side).palm_pose(np.zeros(7))
            self.walk_pose[side] = (side_palm + np.array([-PRE_GRASP, 0.0, WALK_ABOVE]), rotation)
        self.next_scan = robot.time()
        self.steps = self._plan()
        self.step_index = 0
        self.started = False
        self.walking = None  # None (standing), "approach" or "dock"
        self.dock_spot = None
        self.done = False
        self.failed = None  # why it stopped early, if it did

    # ---- the plan: a list of steps ----

    def _plan(self):
        return [
            Step("raise the hands for walking", lambda: self.hands.reach_all(self.walk_pose), self._hands_still),
            Step(
                "walk to table A", lambda: self._walk_to(self._docking_spot(BOTTLE_START, HEADING_AT_A)), self._walked
            ),
            Step("open the hand", lambda: self.hands.grip(SIDE, closed=False), self._hands_still),
            Step(
                "move the hand next to the bottle",
                lambda: self._hand_to(self._bottle(), back=PRE_GRASP),
                self._hands_still,
            ),
            Step("move the hand around the bottle", lambda: self._hand_to(self._bottle()), self._hands_still),
            Step("close the hand", lambda: self.hands.grip(SIDE, closed=True), self._hands_still),
            Step("lift the bottle", lambda: self._hand_to(self._bottle(), up=LIFT), self._hands_still),
            Step("walk to table B", lambda: self._walk_to(self._docking_spot(PLACE_MARK, HEADING_AT_B)), self._walked),
            Step(
                "move the bottle over the mark", lambda: self._hand_to(self._place_spot(), up=LIFT), self._hands_still
            ),
            Step("put the bottle down", lambda: self._hand_to(self._place_spot()), self._hands_still),
            Step("open the hand", lambda: self.hands.grip(SIDE, closed=False), self._hands_still),
            Step("pull the hand back", lambda: self._hand_to(self._place_spot(), back=PRE_GRASP), self._hands_still),
        ]

    def update(self) -> None:
        """One control step."""
        if self.done:
            return
        step = self.steps[self.step_index]
        if not self.started:
            logger.info("t=%5.1fs  step %d/%d: %s", self.robot.time(), self.step_index + 1, len(self.steps), step.name)
            if step.start() is False:
                self.failed = f"{step.name}: can't reach"
                self.done = True
                return
            self.started = True
        self._drive()
        self.hands.update()
        if step.finished():
            self.step_index += 1
            self.started = False
            self.done = self.step_index == len(self.steps)

    # ---- walking ----

    def _docking_spot(self, object_xy, heading):
        """Where the pelvis stands so the object is at self.reach (pelvis frame)."""
        c, s = math.cos(heading), math.sin(heading)
        rx, ry = self.reach[0], self.reach[1]
        return (object_xy[0] - (c * rx - s * ry), object_xy[1] - (s * rx + c * ry), heading)

    def _walk_to(self, spot):
        x, y, h = spot
        self.dock_spot = spot
        self.navigator.set_goal((x - APPROACH * math.cos(h), y - APPROACH * math.sin(h)))
        self.walking = "approach"

    def _drive(self):
        """The walking command for this step (standing still unless walking)."""
        if self.walking is None:
            self.robot.move(0.0, 0.0, 0.0)
            return
        pose = self.robot.pose()
        scan = None
        if self.robot.time() >= self.next_scan - 1e-9:
            self.next_scan += SCAN_DT
            scan = self.robot.scan()
        if self.walking == "approach":
            cmd = self.navigator.update(pose, scan, self.robot.dt)
            if self.navigator.arrived:
                self.walking = "dock"
        else:
            if scan is not None:  # keep the people (if any) up to date for the safety layer
                self.navigator.update(pose, scan, self.robot.dt)
            cmd, docked = self.navigator.dock(pose, self.dock_spot, self.robot.dt)
            if docked:
                self.walking = None
                cmd = (0.0, 0.0, 0.0)
        self.robot.move(*cmd)

    def _walked(self):
        return self.walking is None and self.robot.speed() < 0.05

    # ---- the hand ----

    def _bottle(self):
        """The bottle's center, in the world (from Perception)."""
        position, _ = self.robot.find("bottle")
        return position

    def _place_spot(self):
        """Where the bottle's center goes on table B."""
        return np.array([PLACE_MARK[0], PLACE_MARK[1], TABLE_SIZE[2] + BOTTLE_HEIGHT / 2 + PLACE_ABOVE])

    def _hand_to(self, center_world, back=0.0, up=0.0):
        """Move the palm so the grasp point is at center_world (+ up, - back along
        the robot's forward direction), hand in its natural orientation."""
        x, y, heading = self.robot.pose()
        c, s = math.cos(heading), math.sin(heading)
        dx, dy = center_world[0] - x, center_world[1] - y
        center = np.array([c * dx + s * dy - back, -s * dx + c * dy, center_world[2] - self.robot.pelvis_height() + up])
        palm = center - self.palm_rotation @ self.grasp_offset
        return self.hands.reach(SIDE, palm, self.palm_rotation)

    def _hands_still(self):
        return not self.hands.busy()


ON_MARK = 0.03  # m: the bottle's center within 3 cm of the mark's center counts as on it


def succeeded(robot, mission) -> bool:
    """The bottle stands on the mark and the hand has let go."""
    position, _ = robot.find("bottle")
    on_mark = np.hypot(position[0] - PLACE_MARK[0], position[1] - PLACE_MARK[1]) < ON_MARK
    on_table = position[2] > TABLE_SIZE[2]
    return mission.done and mission.failed is None and bool(on_mark and on_table)
