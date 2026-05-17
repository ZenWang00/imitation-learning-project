from pathlib import Path

import numpy as np
import torch

from src.datasets.transitions import TransitionBatch
from src.imitation.continuous_iqlearn_agent import ContinuousIQLearnAgent


def test_continuous_iqlearn_update_returns_metrics() -> None:
    agent = ContinuousIQLearnAgent.build_model(
        3,
        1,
        "continuous",
        {"hidden_dims": [8], "target_update_interval": 1},
    )
    batch = TransitionBatch(
        observations=torch.zeros((4, 3)),
        actions=torch.zeros((4, 1)),
        next_observations=torch.zeros((4, 3)),
        dones=torch.ones(4),
        rewards=torch.ones(4),
        action_type="continuous",
    )

    metrics = agent.update_with_replay(batch, batch)

    assert "loss" in metrics
    assert "actor_loss" in metrics


def test_continuous_iqlearn_checkpoint_round_trip(tmp_path: Path) -> None:
    agent = ContinuousIQLearnAgent.build_model(3, 1, "continuous", {"hidden_dims": [8]})
    observation = np.zeros(3, dtype=np.float32)
    path = tmp_path / "agent.pt"

    action_before = agent.act(observation)
    agent.save(path)
    restored = ContinuousIQLearnAgent.load(path)
    action_after = restored.act(observation)

    assert action_before.shape == (1,)
    assert np.allclose(action_before, action_after)
