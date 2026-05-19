import gymnasium as gym
import numpy as np

from src.envs.actions import action_dtype_for_space, format_action_for_env


def test_format_action_for_discrete_space() -> None:
    action_space = gym.spaces.Discrete(2)

    action = format_action_for_env(np.asarray(1.0, dtype=np.float32), action_space)

    assert action == 1
    assert isinstance(action, int)
    assert action_dtype_for_space(action_space) == np.dtype(np.int64)


def test_format_action_for_box_space() -> None:
    action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)

    action = format_action_for_env(np.asarray([0.5]), action_space)

    assert action.shape == (1,)
    assert action.dtype == np.float32
    assert action_dtype_for_space(action_space) == np.dtype(np.float32)
