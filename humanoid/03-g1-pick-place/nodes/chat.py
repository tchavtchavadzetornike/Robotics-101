"""Talk to the robot from a terminal: type a sentence, read its reply.

A dynamic dora node: start the dataflow first (./run.sh g1-agent), then in a
second terminal, from humanoid/03-g1-pick-place:

    uv run --no-sync python nodes/chat.py

Type "stop" to stop the robot at once. Ctrl-D or Ctrl-C to leave (the robot keeps running).
"""

import sys
import threading

import pyarrow as pa
from dora import Node

node = Node("chat")


def read_keyboard():
    for line in sys.stdin:
        line = line.strip()
        if line:
            node.send_output("text", pa.array([line]))


threading.Thread(target=read_keyboard, daemon=True).start()
print("Say something to the robot (\"stop\" stops it):", flush=True)
for event in node:
    if event["type"] == "STOP":
        break
    if event["type"] == "INPUT" and event["id"] == "reply":
        print(f"robot: {event['value'][0].as_py()}", flush=True)
