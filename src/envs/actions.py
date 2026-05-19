from __future__ import annotations

import gymnasium as gym
import numpy as np


def format_action_for_env(action: np.ndarray | int | float, action_space: gym.Space):
    if isinstance(action_space, gym.spaces.Discrete):
        return int(np.asarray(action).item())
    if isinstance(action_space, gym.spaces.Box):
        return np.asarray(action, dtype=np.float32).reshape(action_space.shape)
    raise TypeError(f"Unsupported action space: {type(action_space).__name__}")


def action_dtype_for_space(action_space: gym.Space) -> np.dtype:
    if isinstance(action_space, gym.spaces.Discrete):
        return np.dtype(np.int64)
    if isinstance(action_space, gym.spaces.Box):
        return np.dtype(np.float32)
    raise TypeError(f"Unsupported action space: {type(action_space).__name__}")
