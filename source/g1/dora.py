"""The G1 behind dora messages: DoraG1 implements the robot's interfaces
(g1.interfaces) on top of the sim node's messages (g1/nodes/sim.py).

So a task's logic, written against the interfaces, runs unchanged inside a
dora node: step() sends the commands and waits for the robot's next state.
The same adapter would work with a real G1 that publishes these messages.

    robot = DoraG1(Node(), objects=["bottle"])
    mission = PickAndPlace(robot)
    while not mission.done:
        mission.update(); robot.step()
"""

import time

import numpy as np
import pyarrow as pa

from . import joints, messages
from .config import CONTROL_DT
from .interfaces import G1

START_TIMEOUT = 30.0  # s of real time to wait for the sim's first state before giving up


class DoraG1(G1):
    def __init__(self, node, objects=()):
        self.node = node
        self.walk = np.zeros(3)
        self.upper = None
        self.hands = np.zeros(len(joints.HANDS))
        self.objects = {}
        self._time = 0.0
        self._pose = (0.0, 0.0, 0.0)
        self._state = None
        self._scan = None
        self._index = {name: i for i, name in enumerate([*joints.UPPER_BODY, *joints.HANDS])}
        deadline = time.monotonic() + START_TIMEOUT
        while self._state is None or any(name not in self.objects for name in objects):  # the first full state
            if time.monotonic() > deadline:
                raise TimeoutError(f"the sim node sent no full state within {START_TIMEOUT:.0f} s")
            self._receive(timeout=1.0)
        _, _, _, upper, hands = self._state
        self.upper, self.hands = upper.copy(), hands.copy()

    dt = CONTROL_DT

    # Locomotion, UpperBody, Hands: remembered, sent on step()
    def move(self, vx, vy, vyaw):
        self.walk = np.array([vx, vy, vyaw], float)

    def set_stand_height(self, height):
        raise NotImplementedError("the sim node keeps the standing height")

    def set_upper_body_targets(self, angles):
        self.upper = np.asarray(angles, float)

    def set_hand_targets(self, side, angles):
        start = 0 if side == "left" else len(joints.LEFT_HAND)
        self.hands[start : start + len(joints.HAND[side])] = angles

    # State, Lidar, Perception: the newest messages
    def time(self):
        return self._time

    def joint_positions(self, names):
        _, _, _, upper, hands = self._state
        return np.concatenate([upper, hands])[[self._index[n] for n in names]]

    def pose(self):
        return self._pose

    def speed(self):
        return float(self._state[2])

    def pelvis_height(self):
        return float(self._state[1])

    def scan(self):
        return self._scan if self._scan is not None else np.full(360, np.inf)

    def find(self, name):
        return self.objects[name]

    def step(self):
        """Send this step's commands, then wait for the robot's next state."""
        self.node.send_output("cmd_vel", pa.array(messages.pack_cmd_vel(self.walk)))
        self.node.send_output("upper_body", pa.array(self.upper))
        self.node.send_output("hands", pa.array(self.hands))
        before = self._time
        while self._time <= before:
            self._receive()

    def _receive(self, timeout=None):
        """Handle the next event. With a timeout, returns when nothing came in time.
        The graph stopping (STOP, or every sender gone) ends the program."""
        event = self.node.next() if timeout is None else self.node.next(timeout=timeout)
        if event is None and timeout is not None:
            return
        if event is None or event["type"] == "STOP":
            raise SystemExit(0)
        if event["type"] != "INPUT":
            return
        value = event["value"].to_numpy()
        if event["id"] == "base_pose":
            _, (x, y), heading = messages.unpack_pose(value)
            self._pose = (float(x), float(y), float(heading))
        elif event["id"] == "robot_state":
            self._state = messages.unpack_robot_state(value)
            self._time = float(self._state[0])
        elif event["id"] == "scan":
            _, self._scan = messages.unpack_scan(value)
        elif event["id"] == "object_pose":
            _, position, heading = messages.unpack_object_pose(value)
            self.objects[event["metadata"]["name"]] = (position, float(heading))
