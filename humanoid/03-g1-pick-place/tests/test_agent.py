"""The agent's loop and safety check (g1/agent.py), with a scripted fake model: no API, no cost.

  uv run --with pytest pytest tests/test_agent.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1 import agent  # noqa: E402

OBJECTS = [
    {"id": 1, "kind": "bottle", "color": "blue", "position": [1.42, 0.0, 0.875]},
    {"id": 2, "kind": "can", "color": "red", "position": [1.6, 0.0, 0.86]},
]


def tool(name, **inputs):
    return NS(type="tool_use", id=f"call_{name}_{len(inputs)}", name=name, input=inputs)


def text(t):
    return NS(type="text", text=t)


class FakeModel:
    """Answers each request with the next scripted response; remembers what it was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.sent = []
        self.beta = NS(messages=NS(create=self.create))
        self.messages = NS(create=self.create)

    def create(self, **request):
        self.sent.append({**request, "messages": list(request["messages"])})  # as sent: the agent keeps appending
        blocks, stop = self.responses.pop(0)
        return NS(content=blocks, stop_reason=stop)


class FakeRobot:
    def __init__(self, objects=OBJECTS, held=None, fail=()):
        self._objects, self._held, self.fail, self.ran = list(objects), held, set(fail), []

    def objects(self):
        return self._objects

    def held(self):
        return self._held

    def run(self, command):
        self.ran.append(command)
        if command["skill"] in self.fail:
            return False, f"{command['skill']} failed: the hand closed on nothing"
        if command["skill"] == "pick":
            self._held = command["object"]
        if command["skill"] == "place":
            self._held = None
        return True, f"{command['skill']} done"


def test_pick_and_place_in_order():
    model = FakeModel(
        ([tool("get_objects")], "tool_use"),
        ([tool("pick", object_id=1)], "tool_use"),
        ([tool("place", target="mark", next_to_id=None)], "tool_use"),
        ([text("The blue bottle is on the mark.")], "end_turn"),
    )
    robot = FakeRobot()
    reply = agent.converse(agent.Agent(model), "put the blue bottle on the mark", robot)
    assert reply == "The blue bottle is on the mark."
    assert robot.ran == [{"skill": "pick", "object": 1}, {"skill": "place", "target": "mark"}]
    first_result = model.sent[1]["messages"][-1]["content"][0]
    assert "#1 blue bottle on table_a" in first_result["content"] and "can be picked" in first_result["content"]


def test_unknown_id_is_rejected_before_the_robot_moves():
    model = FakeModel(
        ([tool("pick", object_id=7)], "tool_use"),
        ([text("I don't see that object.")], "end_turn"),
    )
    robot = FakeRobot()
    agent.converse(agent.Agent(model), "pick object 7", robot)
    assert robot.ran == []
    result = model.sent[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "no object #7" in result["content"]


def test_a_failure_reaches_the_model():
    model = FakeModel(
        ([tool("pick", object_id=1)], "tool_use"),
        ([text("The grasp missed.")], "end_turn"),
    )
    agent.converse(agent.Agent(model), "pick the bottle", FakeRobot(fail={"pick"}))
    result = model.sent[1]["messages"][-1]["content"][0]
    assert result["is_error"] and result["content"].startswith("FAILED:")


def test_place_with_empty_hand_and_next_to_itself():
    a = agent.check_call(agent.Call("x", "place", {"target": "mark", "next_to_id": None}), OBJECTS, None)
    assert "empty" in a
    b = agent.check_call(agent.Call("x", "place", {"target": "next_to", "next_to_id": 1}), OBJECTS, 1)
    assert "in the hand" in b


def test_several_calls_in_one_turn_run_one_at_a_time_with_all_results_together():
    model = FakeModel(
        ([tool("get_objects"), tool("pick", object_id=2)], "tool_use"),
        ([text("Holding the red can.")], "end_turn"),
    )
    robot = FakeRobot()
    agent.converse(agent.Agent(model), "pick the red can", robot)
    results = model.sent[1]["messages"][-1]["content"]
    assert len(results) == 2 and robot.ran == [{"skill": "pick", "object": 2}]


def test_the_conversation_only_grows_and_the_request_is_stable():
    model = FakeModel(([tool("get_objects")], "tool_use"), ([text("ok")], "end_turn"))
    a = agent.Agent(model)
    agent.converse(a, "what do you see", FakeRobot())
    first, second = model.sent
    assert second["system"] == first["system"] and second["tools"] == first["tools"]
    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"


def test_refusal_ends_the_turn():
    model = FakeModel(([], "refusal"))
    reply = agent.converse(agent.Agent(model), "something", FakeRobot())
    assert reply == "I can't do that."


def test_without_fallbacks_no_beta_parameters_are_sent():
    model = FakeModel(([text("ok")], "end_turn"))
    agent.converse(agent.Agent(model, model="claude-opus-4-8", fallbacks=False), "hello", FakeRobot())
    sent = model.sent[0]
    assert "betas" not in sent and "fallbacks" not in sent
    assert sent["model"] == "claude-opus-4-8" and sent["thinking"] == {"type": "adaptive"}


LYING = [{**OBJECTS[0], "standing": False}, OBJECTS[1]]


def test_lying_objects_are_shown_as_fallen_and_cannot_be_picked():
    text = agent.describe_objects(LYING, None)
    assert "#1 blue bottle" in text and "LYING ON ITS SIDE" in text and "only grasps standing objects" in text
    problem = agent.check_call(agent.Call("x", "pick", {"object_id": 1}), LYING, None)
    assert "lies on its side" in problem
    assert agent.check_call(agent.Call("x", "pick", {"object_id": 2}), LYING, None) is None
