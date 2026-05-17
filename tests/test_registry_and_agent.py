from pathlib import Path

import numpy as np
import torch

from src.datasets.transitions import TransitionBatch
from src.imitation.iqlearn_agent import IQLearnAgent
from src.imitation.registry import build_algorithm


def test_registry_builds_iqlearn() -> None:
    agent = build_algorithm(
        "iqlearn",
        observation_dim=4,
        action_dim=2,
        action_type="discrete",
        config={"hidden_dims": [8], "target_update_interval": 1, "max_grad_norm": 10.0},
    )

    assert isinstance(agent, IQLearnAgent)


def test_iqlearn_update_returns_metrics() -> None:
    agent = IQLearnAgent.build_model(
        4,
        2,
        "discrete",
        {"hidden_dims": [8], "target_update_interval": 1, "max_grad_norm": 10.0},
    )
    batch = TransitionBatch(
        observations=torch.zeros((4, 4)),
        actions=torch.zeros(4, dtype=torch.long),
        next_observations=torch.zeros((4, 4)),
        dones=torch.ones(4),
        rewards=torch.ones(4),
    )

    metrics = agent.update_with_replay(batch, batch)

    assert "loss" in metrics
    assert "expert_reward_mean" in metrics
    assert "replay_term" in metrics


def test_iqlearn_checkpoint_round_trip(tmp_path: Path) -> None:
    agent = IQLearnAgent.build_model(4, 2, "discrete", {"hidden_dims": [8]})
    observation = np.zeros(4, dtype=np.float32)
    path = tmp_path / "agent.pt"

    action_before = agent.act(observation)
    agent.save(path)
    restored = IQLearnAgent.load(path)
    action_after = restored.act(observation)

    assert action_before == action_after
