from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    observation_dim: int
    action_dim: int
    action_type: str


def infer_env_spec(env_id: str) -> EnvSpec:
    env = gym.make(env_id)
    try:
        observation_dim = int(np.prod(env.observation_space.shape))
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = int(env.action_space.n)
            action_type = "discrete"
        elif isinstance(env.action_space, gym.spaces.Box):
            action_dim = int(np.prod(env.action_space.shape))
            action_type = "continuous"
        else:
            raise TypeError(f"Unsupported action space: {type(env.action_space).__name__}")
    finally:
        env.close()
    return EnvSpec(
        env_id=env_id,
        observation_dim=observation_dim,
        action_dim=action_dim,
        action_type=action_type,
    )

