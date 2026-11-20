"""Run a task in Isaac Lab, without dora, and score it. Works for every task in tasks/.

    $ISAACLAB/isaaclab.sh -p scripts/run_task.py crossing --headless                 # 5 episodes
    $ISAACLAB/isaaclab.sh -p scripts/run_task.py crossing --headless --episodes 20
    $ISAACLAB/isaaclab.sh -p scripts/run_task.py pick_and_place --livestream 2 --viz kit   # watch at <link>/viewer/

Each episode: the task's mission drives the G1 (through g1.isaac.IsaacG1, the
robot's interfaces) until it finishes or the environment ends the episode
(for example: touched a person, fell, dropped the bottle, out of time).
PASS = every episode succeeded.
"""

import argparse
import importlib
import logging
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a G1 task and score it.")
parser.add_argument("task", help="a folder in tasks/, e.g. crossing or pick_and_place")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=0, help="first variant (e.g. crowd); each episode gets the next")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")  # g1.isaac.DEVICE: one robot simulates faster on the CPU
args = parser.parse_args()
app = AppLauncher(args).app

# ---- from here on Isaac Sim is running ----
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from g1.isaac import IsaacG1  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

task = importlib.import_module(f"tasks.{args.task}")
logging.basicConfig(level=logging.INFO, format="%(message)s")  # the mission reports each step


def episode(env, robot):
    """One episode. Returns (outcome, time)."""
    mission = task.Mission(robot)
    while app.is_running():
        mission.update()
        robot.step()
        _, _, terminated, truncated, _ = robot.last_step
        if bool(terminated[0]) or bool(truncated[0]):
            terms = env.termination_manager
            ended = [name for name in terms.active_terms if bool(terms.get_term(name)[0])]
            return ", ".join(ended) or "ended", robot.time()
        if mission.done:
            outcome = "success" if task.succeeded(robot, mission) else f"failed: {mission.failed or 'not done right'}"
            return outcome, robot.time()
    return "stopped", robot.time()


def main() -> bool:
    cfg = parse_env_cfg(task.ENV_ID, device=args.device, num_envs=1)
    task.configure(cfg, args.seed)
    env = gym.make(task.ENV_ID, cfg=cfg).unwrapped
    env.reset()
    robot = IsaacG1(env, args.seed)
    results = []
    for k in range(args.episodes):
        if k > 0:
            robot.reset()
        outcome, t = episode(env, robot)
        results.append((outcome, t))
        print(f"episode {k + 1}: {outcome} at t={t:.1f}s", flush=True)
        if not app.is_running():
            break
    wins = [t for outcome, t in results if outcome == "success"]
    print("-" * 60)
    print(f"{args.task}: success {len(wins)}/{len(results)}")
    if wins:
        print(f"time: mean {np.mean(wins):.1f} s, fastest {min(wins):.1f} s, slowest {max(wins):.1f} s")
    ok = len(wins) == len(results) == args.episodes
    print("PASS" if ok else "FAIL", flush=True)
    if not args.headless:  # keep the picture up until Ctrl-C
        try:
            while app.is_running():
                env.sim.render()
        except KeyboardInterrupt:
            pass
    env.close()
    return ok


if __name__ == "__main__":
    ok = main()
    app.close()
    sys.exit(0 if ok else 1)
