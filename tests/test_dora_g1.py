"""DoraG1 against a scripted dora node: startup, timeout, joint lookup."""

import g1.dora
import numpy as np
import pytest
from g1 import joints, messages
from g1.dora import DoraG1


class Value:
    def __init__(self, array):
        self.array = array

    def to_numpy(self):
        return self.array


class ScriptedNode:
    """Delivers the given events, then nothing."""

    def __init__(self, events=()):
        self.events = list(events)

    def next(self, timeout=None):
        return self.events.pop(0) if self.events else None

    def send_output(self, name, value, metadata=None):
        pass


def input_event(name, array):
    return {"type": "INPUT", "id": name, "value": Value(array)}


def test_a_silent_sim_ends_in_a_timeout(monkeypatch):
    monkeypatch.setattr(g1.dora, "START_TIMEOUT", 0.05)
    with pytest.raises(TimeoutError):
        DoraG1(ScriptedNode())


def test_joint_positions_come_back_in_the_order_asked():
    upper = np.arange(len(joints.UPPER_BODY)) * 0.01
    hands = np.arange(len(joints.HANDS)) * 0.1
    state = messages.pack_robot_state(0.0, 0.75, 0.0, upper, hands)
    robot = DoraG1(ScriptedNode([input_event("robot_state", state)]))
    assert np.allclose(robot.joint_positions(joints.HAND["right"]), hands[len(joints.LEFT_HAND) :])
    assert np.allclose(robot.joint_positions(joints.UPPER_BODY[::-1]), upper[::-1])
