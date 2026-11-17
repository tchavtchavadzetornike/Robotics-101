"""The brain: Claude turns a sentence into skill calls, reads each result, decides the next.

The model never moves a joint. It chooses from TOOLS, the skills from phase 4
plus get_objects (the memory). Every call passes check_call first: an id the
memory doesn't have, picking with a full hand, placing with an empty one, are
rejected with a reason the model reads and can act on (grounding: the memory
owns the truth, the model refers to it by id).

One conversation, three methods, no threads and no dora (nodes/agent.py adds those):

    agent = Agent(client)
    agent.hear("put the red can next to the blue bottle")
    turn = agent.think()                  # one API call: tool calls, or a final reply
    ... run turn.calls one at a time ...  # robot skills take seconds to a minute
    agent.give([(call_id, text, is_error), ...])
    turn = agent.think()                  # ... until turn.done: turn.text is the reply

The conversation only grows (append-only): thinking blocks and cached prefixes stay valid.

Model: AGENT_MODEL, default claude-opus-5-5. Through a proxy that only offers older
models (Exxeta's LiteLLM: claude-opus-4-8), set AGENT_MODEL and ANTHROPIC_BASE_URL.
Thinking is switched on explicitly (adaptive): on Opus 4.8 leaving it out means no
thinking. Effort medium (AGENT_EFFORT): the decisions are short, a robot skill takes
far longer than the thinking. Refusal fallbacks (AGENT_FALLBACKS, default on): if a
safety classifier declines a request, the API retries it on a fallback model in the
same call. A beta: switch it off (AGENT_FALLBACKS=0) through proxies without betas.
"""

import json
import os
from dataclasses import dataclass, field

from g1 import reach

MODEL = os.environ.get("AGENT_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("AGENT_EFFORT", "medium")
FALLBACKS = os.environ.get("AGENT_FALLBACKS", "1") != "0"
MAX_TOKENS = 8000
ROBOT_TOOLS = ("look_around", "go_to", "pick", "place")

SYSTEM = """You control a Unitree G1 humanoid robot in a room with two tables. You act only through the tools.

The room:
- table_a: 0.6 m x 0.8 m, in front of the robot's start position
- table_b: 0.8 m x 0.6 m, to the right, with a green mark on it ("mark")
- objects stand on the tables: bottles, cans and boxes in red, yellow, green and blue

What the robot can do:
- look_around: walks to two viewpoints and looks at both tables; the memory fills up
- get_objects: what the memory holds now: id, color, kind, position, standing or lying on its side, whether it can be picked and why not, what is in the hand
- pick(object_id): walks to the best table side, grasps, lifts; one hand, one object at a time
- place(target): puts the held object on table_a, table_b, the mark, or next to another object (target next_to + next_to_id)
- go_to(place): start, table_a, table_b, mark

Rules:
- Refer to objects only by the ids get_objects returns. Call get_objects before choosing an id; call look_around first if the memory is empty or may be out of date.
- If a request matches more than one object (two red cans) or none, ask which one in one short question and stop; don't guess.
- One robot action at a time: wait for its result before the next.
- Every result says what happened. If a skill fails, read the reason: try another way when there is one (another object, another place, look_around again), otherwise say what failed and stop. Never say something worked unless its result says so.
- "but #N moved", "#N fell over" or "#N is gone" in a result means something was disturbed: mention it.
- A FAILED result stays failed. Don't explain it away when get_objects later shows the object nearby: it may be lying on its side. Report what the result says.
- An object lying on its side has fallen over and can't be picked: say so.
- Your replies are spoken aloud: one or two short sentences, no lists, no markdown. Say what you did or what you need."""

TOOLS = [
    {
        "name": "look_around",
        "description": "Walk to two viewpoints, look at both tables, update the memory. Takes about 40 s.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "get_objects",
        "description": "The memory: every object seen, with id, color, kind, position (m), whether it can be picked and why not, and what is in the hand. Instant.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "pick",
        "description": "Walk to the object, grasp it, lift it. The hand must be empty. Takes 20-40 s. Fails with a reason if it can't reach or the grasp misses.",
        "input_schema": {
            "type": "object",
            "properties": {"object_id": {"type": "integer", "description": "an id from get_objects"}},
            "required": ["object_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "place",
        "description": "Carry the held object and put it down: on table_a, on table_b, on the mark, or next to another object (target next_to with next_to_id). Takes 30-60 s. Checks afterwards that it stands there.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": ["table_a", "table_b", "mark", "next_to"]},
                "next_to_id": {"type": ["integer", "null"], "description": "with target next_to: the id to put it next to; else null"},
            },
            "required": ["target", "next_to_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "go_to",
        "description": "Walk to a place and stand there.",
        "input_schema": {
            "type": "object",
            "properties": {"place": {"type": "string", "enum": ["start", "table_a", "table_b", "mark"]}},
            "required": ["place"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


@dataclass
class Call:
    id: str
    name: str
    input: dict


@dataclass
class Turn:
    calls: list = field(default_factory=list)
    text: str = ""
    done: bool = False
    stop_reason: str = ""


def describe_objects(objects, held):
    """The memory as text for the model: one line per object, with what pick would say."""
    if not objects:
        return "The memory is empty: nothing seen yet. Call look_around."
    lines = []
    for o in objects:
        if o["id"] == held:
            lines.append(f"#{o['id']} {o['color']} {o['kind'] or 'object'}: in the hand")
            continue
        posture = "standing" if o.get("standing", True) else "LYING ON ITS SIDE (fell over)"
        others = [x["position"] for x in objects if x["id"] not in (o["id"], held)]
        options, reason = reach.grasp_options(o["position"], others)
        table = reach.table_of(o["position"]) or "no table"
        x, y = o["position"][0], o["position"][1]
        if not o.get("standing", True):
            can = "can't be picked: it lies on its side, the hand only grasps standing objects"
        else:
            can = "can be picked" if options else f"can't be picked: {reason}"
        lines.append(f"#{o['id']} {o['color']} {o['kind'] or 'object'} on {table} at ({x:.2f}, {y:.2f}), {posture}: {can}")
    lines.append(f"In the hand: #{held}" if held is not None else "The hand is empty.")
    return "\n".join(lines)


def check_call(call, objects, held):
    """None if the call may go to the robot, else the reason it may not (the model reads it)."""
    ids = {o["id"] for o in objects}
    if call.name == "pick":
        object_id = call.input.get("object_id")
        if held is not None:
            return f"The hand already holds #{held}. Place it first."
        if object_id not in ids:
            return f"There is no object #{object_id} in the memory. Call get_objects (or look_around) and use an id from it."
        if next(o for o in objects if o["id"] == object_id).get("standing", True) is False:
            return f"#{object_id} lies on its side: the hand only grasps standing objects."
    if call.name == "place":
        if held is None:
            return "The hand is empty. Pick something first."
        if call.input.get("target") == "next_to":
            anchor = call.input.get("next_to_id")
            if anchor not in ids:
                return f"There is no object #{anchor} in the memory to put it next to. Use an id from get_objects."
            if anchor == held:
                return "That is the object in the hand. Choose another object to put it next to."
    return None


def skill_command(call):
    """A checked tool call -> the JSON command nodes/skills.py takes."""
    if call.name == "look_around":
        return {"skill": "look_around"}
    if call.name == "go_to":
        return {"skill": "go_to", "place": call.input["place"]}
    if call.name == "pick":
        return {"skill": "pick", "object": call.input["object_id"]}
    if call.name == "place":
        target = call.input["target"]
        if target == "next_to":
            return {"skill": "place", "target": {"next_to": call.input["next_to_id"]}}
        return {"skill": "place", "target": target}
    raise ValueError(f"{call.name} is not a robot skill")


class Agent:
    def __init__(self, client, model=MODEL, effort=EFFORT, fallbacks=FALLBACKS):
        self.client = client
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.messages = []
        self.log = []  # (role, text) for the dashboard and the notes

    def hear(self, text):
        self.messages.append({"role": "user", "content": text})
        self.log.append(("user", text))

    def think(self):
        """One request to the model. Returns its tool calls, or its final reply (done)."""
        request = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=TOOLS,
            messages=self.messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        if self.fallbacks:
            response = self.client.beta.messages.create(
                **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            )
        else:
            response = self.client.messages.create(**request)
        self.messages.append({"role": "assistant", "content": response.content})
        turn = Turn(stop_reason=response.stop_reason)
        turn.text = " ".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "refusal":
            turn.done, turn.text = True, turn.text or "I can't do that."
        elif response.stop_reason == "max_tokens":
            turn.done, turn.text = True, "I lost track of that; please say it again more simply."
        else:
            turn.calls = [Call(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
            turn.done = not turn.calls
        for call in turn.calls:
            self.log.append(("tool", f"{call.name}({json.dumps(call.input)})"))
        if turn.done:
            self.log.append(("robot", turn.text))
        return turn

    def give(self, results):
        """results: [(call id, text, is_error)], all calls of the last turn, in one message."""
        content = [{"type": "tool_result", "tool_use_id": i, "content": text, "is_error": bool(error)} for i, text, error in results]
        self.messages.append({"role": "user", "content": content})
        for _, text, error in results:
            self.log.append(("result", ("ERROR: " if error else "") + text))


class Conductor:
    """Goes through one turn's tool calls in order.

    get_objects and calls rejected by check_call are answered on the spot; a
    robot skill is handed out (next() returns it) and the caller reports its
    result with finished(). When done, results go to Agent.give().
    """

    def __init__(self, turn):
        self.calls = list(turn.calls)
        self.results = []
        self.waiting = None

    @property
    def done(self):
        return not self.calls and self.waiting is None

    def next(self, objects, held):
        """The next robot call to run, or None when every call has a result."""
        while self.calls and self.waiting is None:
            call = self.calls.pop(0)
            if call.name == "get_objects":
                self.results.append((call.id, describe_objects(objects, held), False))
                continue
            problem = check_call(call, objects, held)
            if problem:
                self.results.append((call.id, problem, True))
                continue
            if call.name not in ROBOT_TOOLS:
                self.results.append((call.id, f"Unknown tool {call.name}.", True))
                continue
            self.waiting = call
            return call
        return None

    def finished(self, ok, message):
        call, self.waiting = self.waiting, None
        self.results.append((call.id, message if ok else f"FAILED: {message}", not ok))


def converse(agent, text, robot, max_turns=20, on_call=None):
    """One request from start to reply, for a robot with objects(), held() and run(command) -> (ok, message).

    Returns the reply text. on_call(call) is called before each robot skill (for logs).
    """
    agent.hear(text)
    for _ in range(max_turns):
        turn = agent.think()
        if turn.done:
            return turn.text
        conductor = Conductor(turn)
        while not conductor.done:
            call = conductor.next(robot.objects(), robot.held())
            if call is None:
                break
            if on_call:
                on_call(call)
            ok, message = robot.run(skill_command(call))
            conductor.finished(ok, message)
        agent.give(conductor.results)
    return "I stopped: that took too many steps."
