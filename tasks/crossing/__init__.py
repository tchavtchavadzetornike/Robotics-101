"""Crowd crossing: get across a square full of walking people as fast as is safe.

What every task folder provides (used by scripts/run_task.py):
  ENV_ID      its Isaac Lab environment
  Mission     its logic class: Mission(robot), update(), done, failed
  succeeded   did the episode succeed? succeeded(robot, mission) -> bool
  configure   configure(cfg, seed): which random variant (here: which crowd) the cfg builds
"""

import gymnasium as gym

from .mission import Crossing as Mission  # noqa: F401
from .mission import succeeded  # noqa: F401

ENV_ID = "G1-Crowd-Crossing-v0"


def configure(cfg, seed):
    cfg.seed = seed
    cfg.events.move_crowd.params["seed"] = seed


gym.register(
    id=ENV_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:CrossingEnvCfg"},
)
