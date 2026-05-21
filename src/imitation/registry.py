from __future__ import annotations

from typing import Any

from src.imitation.base import ImitationAgent
from src.imitation.continuous_csil_agent import ContinuousCSILAgent
from src.imitation.continuous_iqlearn_agent import ContinuousIQLearnAgent
from src.imitation.continuous_soar_csil_agent import ContinuousSOARCSILAgent
from src.imitation.csil_agent import CSILAgent
from src.imitation.iqlearn_agent import IQLearnAgent
from src.imitation.soar_csil_agent import SOARCSILAgent


ALGORITHM_REGISTRY: dict[str, type[ImitationAgent]] = {
    "iqlearn": IQLearnAgent,
    "iqlearn_continuous": ContinuousIQLearnAgent,
    "csil": CSILAgent,
    "csil_continuous": ContinuousCSILAgent,
    "soar_csil": SOARCSILAgent,
    "soar_csil_continuous": ContinuousSOARCSILAgent,
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
