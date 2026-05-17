from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class TransitionBatch:
    observations: Tensor
    actions: Tensor
    next_observations: Tensor
    dones: Tensor
    rewards: Tensor | None = None
    action_type: str = "discrete"


@dataclass(frozen=True)
class TransitionDataset:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            observations=self.observations,
            actions=self.actions,
            next_observations=self.next_observations,
            dones=self.dones,
            rewards=self.rewards,
            episode_ids=self.episode_ids,
            timesteps=self.timesteps,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TransitionDataset":
        data = np.load(path)
        return cls(
            observations=data["observations"],
            actions=data["actions"],
            next_observations=data["next_observations"],
            dones=data["dones"],
            rewards=data["rewards"],
            episode_ids=data["episode_ids"],
            timesteps=data["timesteps"],
        )

    def sample_batch(
        self,
        batch_size: int,
        *,
        generator: np.random.Generator,
        device: torch.device | str = "cpu",
    ) -> TransitionBatch:
        indices = generator.integers(low=0, high=len(self), size=batch_size)
        action_type = "continuous" if np.issubdtype(self.actions.dtype, np.floating) else "discrete"
        action_dtype = torch.float32 if action_type == "continuous" else torch.long
        return TransitionBatch(
            observations=torch.as_tensor(
                self.observations[indices], dtype=torch.float32, device=device
            ),
            actions=torch.as_tensor(self.actions[indices], dtype=action_dtype, device=device),
            next_observations=torch.as_tensor(
                self.next_observations[indices], dtype=torch.float32, device=device
            ),
            dones=torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
            rewards=torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            action_type=action_type,
        )
