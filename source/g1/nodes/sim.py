"""dora node: the G1 and its world, in Isaac Lab. The same node for every task.

On a real G1 this node is the robot itself: its walking controller takes
velocity commands, its arm SDK and hands take joint targets, and it reports
its state, its lidar and (with a camera) where objects are.

Settings (env in dataflow.yml):
  TASK     the Isaac Lab environment, e.g. G1-Crowd-Crossing-v0
  SEED     which random variant (e.g. which crowd), default 0
  OBJECTS  scene objects to report, comma separated (e.g. "bottle"), default none
Isaac Lab arguments (--headless, --livestream 2 --viz kit, ...) come from args.

Inputs:  cmd_vel      walking command (forward, left, turn)
         upper_body   17 waist + arm joint targets (optional)
         hands        14 finger joint targets (optional)
Outputs: base_pose    time, x, y, heading                          (every step, 50 Hz)
         robot_state  time, pelvis height, speed, joint angles     (every step, 50 Hz)
         scan         one turn of the lidar                        (10 Hz)
         object_pose  one message per object in OBJECTS            (every step)

It steps the simulation as fast as it can with the newest commands. When the
episode ends it reports how, and keeps the picture up until the graph stops.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")  # g1.isaac.DEVICE: one robot simulates faster on the CPU
args = parser.parse_args()
app = AppLauncher(args).app

# ---- from here on Isaac Sim is running ----
import gymnasium as gym  # noqa: E402
import pyarrow as pa  # noqa: E402
from dora import Node  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tasks  # noqa: E402  registers the tasks' environments
from g1 import joints, messages  # noqa: E402
from g1.config import SCAN_DT  # noqa: E402
from g1.isaac import IsaacG1  # noqa: E402

POLL = 1e-4  # s: how long to wait for a message before the next step


def take_messages(node, robot):
    """Apply every command that has arrived. Returns True when the graph stops."""
    while True:
        event = node.next(timeout=POLL)
        if event is None or event["type"] not in ("INPUT", "STOP"):
            return False  # nothing new (timeout), or all senders are gone
        if event["type"] == "STOP":
            return True
        value = event["value"].to_numpy()
        if event["id"] == "cmd_vel":
            robot.move(*messages.unpack_cmd_vel(value))
        elif event["id"] == "upper_body":
            robot.set_upper_body_targets(value)
        elif event["id"] == "hands":
            robot.set_hand_targets("left", value[: len(joints.LEFT_HAND)])
            robot.set_hand_targets("right", value[len(joints.LEFT_HAND) :])


def publish(node, robot, objects):
    t = robot.time()
    node.send_output("base_pose", pa.array(messages.pack_pose(t, *robot.pose())))
    state = messages.pack_robot_state(
        t,
        robot.pelvis_height(),
        robot.speed(),
        robot.joint_positions(joints.UPPER_BODY),
        robot.joint_positions(joints.HANDS),
    )
    node.send_output("robot_state", pa.array(state))
    for name in objects:
        position, heading = robot.find(name)
        node.send_output("object_pose", pa.array(messages.pack_object_pose(t, position, heading)), {"name": name})


def main():
    task = os.environ["TASK"]
    seed = int(os.environ.get("SEED", "0"))
    objects = [name for name in os.environ.get("OBJECTS", "").split(",") if name]
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    tasks.configure(task, cfg, seed)
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    robot = IsaacG1(env, seed)
    node = Node()

    stop = False
    next_scan = 0.0
    ended = None
    while app.is_running() and not stop and ended is None:
        stop = take_messages(node, robot)
        publish(node, robot, objects)
        if robot.time() >= next_scan - 1e-9:
            next_scan += SCAN_DT
            node.send_output("scan", pa.array(messages.pack_scan(robot.time(), robot.scan())))
        robot.step()
        _, _, terminated, truncated, _ = robot.last_step
        if bool(terminated[0]) or bool(truncated[0]):
            active = [
                name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[0])
            ]
            ended = ", ".join(active) or "ended"

    if ended:
        node.log_info(f"{task}: episode ended at t={robot.time():.1f}s ({ended})")
    while app.is_running() and not stop:  # keep the picture up until the graph is stopped
        stop = take_messages(node, robot)
        env.sim.render()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
