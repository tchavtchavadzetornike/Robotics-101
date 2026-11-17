"""Sends a script of skills, one after another, and stops at the first failure.

SCRIPT (in skills.yml) or ROBOT_SCRIPT (from the shell, wins): commands separated by ";", e.g.
  look_around; pick blue bottle; place mark; pick red can; place next_to blue bottle; go_to start
Phase 5 replaces this node with the language model, which decides the next command from each result.

Inputs:  tick     start after the dataflow is up
         result   JSON text from skills
Outputs: command  JSON text, to skills
"""

import json
import os

import pyarrow as pa
from dora import Node


def parse(line):
    words = line.split()
    name, rest = words[0], " ".join(words[1:])
    if name == "pick":
        return {"skill": "pick", "object": int(rest)} if rest.isdigit() else {"skill": "pick", "label": rest}
    if name == "place":
        if rest.startswith("next_to "):
            anchor = rest[len("next_to ") :]
            return {"skill": "place", "target": {"next_to": int(anchor) if anchor.isdigit() else anchor}}
        return {"skill": "place", "target": rest}
    if name == "go_to":
        return {"skill": "go_to", "place": rest}
    return {"skill": name}


text = os.environ.get("ROBOT_SCRIPT") or os.environ.get("SCRIPT", "look_around")  # ROBOT_SCRIPT from the shell wins
script = [parse(s.strip()) for s in text.split(";") if s.strip()]
step = 0
started = False
node = Node()

for event in node:
    if event["type"] == "STOP":
        break
    if event["type"] != "INPUT":
        continue
    if event["id"] == "tick" and not started:
        started = True
    elif event["id"] == "result":
        result = json.loads(event["value"][0].as_py())
        node.log_info(f"{step}/{len(script)} {result['skill']}: {'OK' if result['ok'] else 'FAILED'}: {result['message']}")
        if not result["ok"]:
            node.log_warn("the script stops here")
            step = len(script)
            continue
    else:
        continue
    if step < len(script):
        node.send_output("command", pa.array([json.dumps(script[step])]))
        step += 1
    elif event["id"] == "result":
        node.log_info("script done")
