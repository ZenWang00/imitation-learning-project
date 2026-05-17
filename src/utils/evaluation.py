from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np


def evaluate_policy(
    env_id: str,
    act_fn: Callable[[np.ndarray], np.ndarray | int],
    *,
    episodes: int,
    seed: int,
) -> tuple[float, float]:
    returns: list[float] = []
    env = gym.make(env_id)
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            terminated = truncated = False
            episode_return = 0.0
            while not (terminated or truncated):
                action = act_fn(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
            returns.append(episode_return)
    finally:
        env.close()
    return float(np.mean(returns)), float(np.std(returns))
