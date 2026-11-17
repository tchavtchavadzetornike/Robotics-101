"""The task: walk to table A, pick up the bottle, walk to table B, put it on the mark.

The mission is a list of steps done one after another (a state machine).
A step either walks to a pose or gives the arm one command, then waits until
that is finished. Each step computes its target when it starts, from what the
robot sees at that moment (where it stands, where the bottle is). So small
errors from earlier steps don't add up.

Frames: "world" is the room (x, y on the floor). "body" is the robot's pelvis
frame (x ahead, y left, z up). Hand targets go to the arm in the body frame.
"""

import numpy as np

from g1 import frames, messages, navigation
from g1.simulator import yaw_of

# Where the bottle should be, seen from the robot, when it stands to grasp it:
# 30 cm ahead and 12 cm to the right, where the right hand reaches best
# (found by trying many targets with the IK, see README).
GRASP_SPOT = np.array([0.30, -0.12])
APPROACH = 0.08  # m: the open hand comes from the right side of the bottle
LIFT = 0.10  # m
# Hand positions in the body frame (m):
READY = np.array(
    [0.05, -0.22, 0.15]
)  # raised in front of the chest, behind the table edge
CARRY = np.array(
    [0.20, -0.18, 0.18]
)  # holding the bottle: its bottom 8 cm above table height
BACK_OFF = 0.5  # m: step back from a table before walking on, clear of its corners
BOTTLE_HALF_HEIGHT = 0.075  # m, from robot/scene.xml
PLACE_GAP = 0.005  # m: let go of the bottle 5 mm above the table
STOP_WAIT = 0.5  # s: after stopping, wait at least this long for a fresh "standing"
WALK_TIMEOUT = 60.0  # s
ARM_TIMEOUT = 20.0  # s

WALK = "walk"


class Step:
    def __init__(self, label, kind, target=None):
        self.label = label  # what a person would call it
        self.kind = kind  # WALK, or an arm command kind from messages.py
        self.target = target  # method that computes the goal when the step starts


class Mission:
    def __init__(self, pick_heading, place_point, place_heading, table_height):
        """pick_heading: which way the robot faces at table A (rad).
        place_point: (x, y) of the mark on table B. place_heading: facing at table B.
        """
        self.pick_heading = pick_heading
        self.place_point = np.array(place_point, dtype=float)
        self.place_heading = place_heading
        self.table_height = table_height

        self.steps = [
            Step("walk to table A", WALK, self.pick_stand_pose),
            Step("raise the hand", messages.MOVE_JOINTS, self.ready),
            Step(
                "move the open hand next to the bottle",
                messages.MOVE_LINE,
                self.beside_bottle,
            ),
            Step("move the hand around the bottle", messages.MOVE_LINE, self.at_bottle),
            Step("close the hand", messages.CLOSE),
            Step("lift the bottle", messages.MOVE_LINE, self.lifted),
            Step("hold it close", messages.MOVE_JOINTS, self.carry),
            Step("step back", WALK, self.backed_off),
            Step("walk to table B", WALK, self.place_stand_pose),
            Step("move the bottle over the mark", messages.MOVE_LINE, self.over_mark),
            Step("put it down", messages.MOVE_LINE, self.on_mark),
            Step("open the hand", messages.OPEN),
            Step("pull the hand away", messages.MOVE_LINE, self.hand_away),
            Step("raise the hand", messages.MOVE_LINE, self.ready),
            Step("hand in the carry pose", messages.MOVE_JOINTS, self.carry),
            Step("step back", WALK, self.backed_off),
            Step("arm down", messages.REST),
        ]
        self.index = -1  # no step started yet
        self.step_start = 0.0
        self.finished = False
        self.failed = None  # the reason, if the mission had to stop

        self.navigator = navigation.Navigator()
        self.walk_goal = None
        self.stopped_at = None  # when this walk step reached its goal
        self.command_id = 0
        self.hand = READY.copy()  # last hand target sent to the arm (body frame)

        # latest observations
        self.time = 0.0
        self.position = np.zeros(3)  # pelvis, world
        self.heading = 0.0
        self.bottle = np.zeros(3)  # bottle center, world

    # ---- frames ----

    def to_body(self, point):
        """World point -> body frame (see g1/frames.py)."""
        return frames.to_body(point, self.position, self.heading)

    @staticmethod
    def stand_pose_for(point, heading):
        """Floor pose (x, y, heading) from which `point` is at GRASP_SPOT."""
        c, s = np.cos(heading), np.sin(heading)
        dx = (
            c * GRASP_SPOT[0] - s * GRASP_SPOT[1]
        )  # GRASP_SPOT turned into the world frame
        dy = s * GRASP_SPOT[0] + c * GRASP_SPOT[1]
        return np.array([point[0] - dx, point[1] - dy, heading])

    # ---- step targets (each is called when its step starts) ----

    def pick_stand_pose(self):
        return self.stand_pose_for(self.bottle, self.pick_heading)

    def place_stand_pose(self):
        return self.stand_pose_for(self.place_point, self.place_heading)

    def backed_off(self):
        c, s = np.cos(self.heading), np.sin(self.heading)
        return np.array(
            [
                self.position[0] - BACK_OFF * c,
                self.position[1] - BACK_OFF * s,
                self.heading,
            ]
        )

    def ready(self):
        return READY.copy()

    def carry(self):
        return CARRY.copy()

    def beside_bottle(self):
        return self.to_body(self.bottle) + [0.0, -APPROACH, 0.0]

    def at_bottle(self):
        return self.to_body(self.bottle)

    def lifted(self):
        return self.hand + [0.0, 0.0, LIFT]

    def over_mark(self):
        # Move the hand so the bottle (wherever it sits in the hand) is over the mark.
        bottle_in_hand = self.to_body(self.bottle) - self.hand
        mark = self.to_body([*self.place_point, 0.0])
        return np.array(
            [mark[0] - bottle_in_hand[0], mark[1] - bottle_in_hand[1], self.hand[2]]
        )

    def on_mark(self):
        # Lower the hand until the bottle's bottom is PLACE_GAP above the table.
        bottom = self.bottle[2] - BOTTLE_HALF_HEIGHT
        return self.hand - [0.0, 0.0, bottom - self.table_height - PLACE_GAP]

    def hand_away(self):
        return self.hand + [0.0, -APPROACH, 0.0]

    # ---- the loop ----

    @property
    def label(self):
        return self.steps[self.index].label if 0 <= self.index < len(self.steps) else ""

    def update(
        self, time, base_position, base_quat, bottle_position, standing, arm_status
    ):
        """Call every 20 ms with the latest observations.

        standing: the legs stand still and steady (from locomotion).
        arm_status: (id, ok) of the last finished arm command, or None.
        Returns (cmd_vel, arm_command). arm_command is None, or a new
        command to send (once).
        """
        dt = time - self.time
        self.time = time
        self.position = np.asarray(base_position, dtype=float)
        self.heading = yaw_of(base_quat)
        self.bottle = np.asarray(bottle_position, dtype=float)
        stop = (0.0, 0.0, 0.0)
        if self.finished or self.failed:
            return stop, None
        if self.index < 0:
            return self._next_step()
        step = self.steps[self.index]

        if step.kind == WALK:
            if time - self.step_start > WALK_TIMEOUT:
                self.failed = f"'{step.label}' took longer than {WALK_TIMEOUT:.0f} s"
                return stop, None
            pose = (self.position[0], self.position[1], self.heading)
            if self.stopped_at is None:
                cmd_vel, arrived = self.navigator.update(pose, self.walk_goal, dt)
                if not arrived:
                    return cmd_vel, None
                self.stopped_at = time
            # Stopped: wait until the legs stand still. The robot sways a little
            # while it stops, so we don't steer anymore. Waiting STOP_WAIT first
            # means an old "standing" message from before the walk can't fool us.
            if not standing or time - self.stopped_at < STOP_WAIT:
                return stop, None
            if not navigation.close_to(pose, self.walk_goal):
                self.stopped_at = None  # slid too far while stopping: walk again
                return stop, None
            return self._next_step()

        # an arm step: wait for the arm to report this command as done
        if time - self.step_start > ARM_TIMEOUT:
            self.failed = f"'{step.label}' took longer than {ARM_TIMEOUT:.0f} s"
            return stop, None
        if arm_status is not None and arm_status[0] == self.command_id:
            if not arm_status[1]:
                self.failed = (
                    f"'{step.label}': the hand can't reach {np.round(self.hand, 3)}"
                )
                return stop, None
            return self._next_step()
        return stop, None

    def _next_step(self):
        self.index += 1
        self.step_start = self.time
        if self.index >= len(self.steps):
            self.finished = True
            return (0.0, 0.0, 0.0), None
        step = self.steps[self.index]
        if step.kind == WALK:
            self.walk_goal = step.target()
            self.stopped_at = None
            self.navigator.stop()
            return (0.0, 0.0, 0.0), None
        if step.target is not None:
            self.hand = step.target()
        self.command_id += 1
        command = messages.pack_arm_command(self.command_id, step.kind, self.hand, 0.0)
        return (0.0, 0.0, 0.0), command
