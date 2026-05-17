from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets.transitions import TransitionBatch


class ImitationAgent(ABC):
    @classmethod
    @abstractmethod
    def build_model(
        cls,
        observation_dim: int,
        action_dim: int,
        action_type: str,
        config: dict[str, Any],
    ) -> "ImitationAgent":
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: TransitionBatch) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray | int:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "ImitationAgent":
        raise NotImplementedError
