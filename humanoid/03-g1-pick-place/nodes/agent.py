"""The brain as a dora node: sentences in, skill commands out, a spoken reply back (g1/agent.py).

The model call takes seconds and the robot must keep running meanwhile, so
Agent.think() runs in a worker thread; a 100 ms tick checks whether it is done.
Robot skills run one at a time: a command goes to skills, the next waits for
its result.

"stop" (or "halt") never reaches the model: it goes straight to skills, which
stops walking and the arm at once, and the current request is dropped.

Inputs:  text      a sentence from a person (chat node, or AGENT_SCRIPT)
         result    JSON text from skills when a skill ends
         objects   the memory (world)
         held      [id, ...] picked up / [-1, ...] put down (skills)
         tick      100 ms: is the model's answer here yet?
Outputs: command   JSON text, one skill, to skills
         reply     what the robot says, to chat and dashboard
         log       one line per event (heard, tool call, result, reply), for the dashboard

AGENT_SCRIPT (agent.yml) or ROBOT_SAY (shell, wins): sentences separated by "|", sent one after
another when the robot is idle
(automatic runs without typing). Needs ANTHROPIC_API_KEY in the environment.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
from anthropic import Anthropic
from dora import Node

from g1 import agent

brain = agent.Agent(Anthropic())
worker = ThreadPoolExecutor(max_workers=1)
thinking = None  # Future of agent.think()
conductor = None  # the calls of the current turn
objects, held = [], None
said = os.environ.get("ROBOT_SAY") or os.environ.get("AGENT_SCRIPT", "")  # ROBOT_SAY from the shell wins
script = [s.strip() for s in said.split("|") if s.strip()]
node = Node()


def send(output, text):
    node.send_output(output, pa.array([text]))


def log(line):
    node.log_info(line)
    send("log", line)


def idle():
    return thinking is None and conductor is None


def start_thinking():
    global thinking
    thinking = worker.submit(brain.think)


def advance():
    """Hand out the next robot call of the turn, or give all results to the model."""
    global conductor
    call = conductor.next(objects, held)
    if call is not None:
        log(f"-> {call.name}({json.dumps(call.input)})")
        send("command", json.dumps(agent.skill_command(call)))
        return
    if conductor.done:
        brain.give(conductor.results)
        conductor = None
        start_thinking()


def hear(text):
    global conductor, thinking
    if text.lower().strip(" .!") in ("stop", "halt"):
        send("command", json.dumps({"skill": "stop"}))
        conductor, thinking = None, None
        send("reply", "Stopped.")
        log("heard stop: robot stopped")
        return
    if not idle():
        send("reply", "I'm still busy with the last request. Say stop to stop it.")
        return
    log(f"heard: {text}")
    brain.hear(text)
    start_thinking()


for event in node:
    if event["type"] == "STOP":
        break
    if event["type"] != "INPUT":
        continue
    name = event["id"]

    if name == "objects":
        objects = json.loads(event["value"][0].as_py())
    elif name == "held":
        value = event["value"].to_numpy()
        held = int(value[0]) if value[0] >= 0 else None
    elif name == "text":
        hear(event["value"][0].as_py())
    elif name == "result" and conductor is not None and conductor.waiting is not None:
        result = json.loads(event["value"][0].as_py())
        log(f"<- {result['skill']}: {'OK' if result['ok'] else 'FAILED'}: {result['message']}")
        conductor.finished(result["ok"], result["message"])
        advance()
    elif name == "tick":
        if thinking is not None and thinking.done():
            future, thinking = thinking, None
            try:
                turn = future.result()
            except Exception as error:  # the API call failed: say so, keep running
                send("reply", f"I can't think right now: {type(error).__name__}.")
                log(f"model error: {error}")
                continue
            if turn.done:
                send("reply", turn.text)
                log(f"says: {turn.text}")
            else:
                conductor = agent.Conductor(turn)
                advance()
        elif idle() and script:
            hear(script.pop(0))
