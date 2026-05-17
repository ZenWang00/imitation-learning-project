from __future__ import annotations

from typing import Any

from src.imitation.base import ImitationAgent
from src.imitation.continuous_iqlearn_agent import ContinuousIQLearnAgent
from src.imitation.iqlearn_agent import IQLearnAgent


ALGORITHM_REGISTRY: dict[str, type[ImitationAgent]] = {
    "iqlearn": IQLearnAgent,
    "iqlearn_continuous": ContinuousIQLearnAgent,
}


def build_algorithm(
    name: str,
    *,
    observation_dim: int,
    action_dim: int,
    action_type: str,
    config: dict[str, Any],
) -> ImitationAgent:
    try:
        algorithm_cls = ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(ALGORITHM_REGISTRY))
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}") from exc
    return algorithm_cls.build_model(observation_dim, action_dim, action_type, config)
