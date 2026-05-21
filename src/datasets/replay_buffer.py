from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.datasets.transitions import TransitionBatch, TransitionDataset


@dataclass(frozen=True)
class ReplayTransition:
    observation: np.ndarray
    action: np.ndarray | int | float
    next_observation: np.ndarray
    done: float
    reward: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._storage: deque[ReplayTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray | int | float,
        next_observation: np.ndarray,
        done: float,
        reward: float,
    ) -> None:
        self._storage.append(
            ReplayTransition(
                observation=np.asarray(observation, dtype=np.float32),
                action=(
                    np.asarray(action, dtype=np.float32)
                    if isinstance(action, np.ndarray)
                    else action
                ),
                next_observation=np.asarray(next_observation, dtype=np.float32),
                done=float(done),
                reward=float(reward),
            )
        )

    def sample_batch(
        self,
        batch_size: int,
        *,
        generator: np.random.Generator,
        device: torch.device | str = "cpu",
    ) -> TransitionBatch:
        if not self._storage:
            raise ValueError("Cannot sample from an empty replay buffer")

        indices = generator.integers(low=0, high=len(self._storage), size=batch_size)
        observations = np.asarray([self._storage[index].observation for index in indices])
        actions = np.asarray([self._storage[index].action for index in indices])
        next_observations = np.asarray(
            [self._storage[index].next_observation for index in indices]
        )
        dones = np.asarray([self._storage[index].done for index in indices])
        rewards = np.asarray([self._storage[index].reward for index in indices])
        dataset = TransitionDataset(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
            episode_ids=np.zeros(batch_size, dtype=np.int64),
            timesteps=np.zeros(batch_size, dtype=np.int64),
        )
        return dataset.sample_batch(batch_size, generator=generator, device=device)
