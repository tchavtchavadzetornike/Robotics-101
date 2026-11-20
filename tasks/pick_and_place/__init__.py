"""Pick and place: walk to table A, pick up the bottle, walk to table B, put it on the mark.

What every task folder provides (used by scripts/run_task.py): ENV_ID,
Mission (its logic class), succeeded(robot, mission) and configure(cfg, seed).
"""

import gymnasium as gym

from .mission import PickAndPlace as Mission  # noqa: F401
from .mission import succeeded  # noqa: F401

ENV_ID = "G1-Pick-Place-v0"


def configure(cfg, seed):
    cfg.seed = seed


gym.register(
    id=ENV_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:PickPlaceEnvCfg"},
)
