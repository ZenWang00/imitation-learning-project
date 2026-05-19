from pathlib import Path

import numpy as np

from scripts.inspect_dataset import inspect_dataset, summary_to_frame


def test_inspect_dataset_reports_quality_metrics(tmp_path: Path) -> None:
    path = tmp_path / "dataset.npz"
    np.savez_compressed(
        path,
        observations=np.zeros((4, 2), dtype=np.float32),
        actions=np.asarray([0, 1, 0, 1], dtype=np.int64),
        next_observations=np.zeros((4, 2), dtype=np.float32),
        dones=np.asarray([0, 1, 0, 1], dtype=np.float32),
        rewards=np.asarray([1, 1, 2, 2], dtype=np.float32),
        episode_ids=np.asarray([0, 0, 1, 1], dtype=np.int64),
        timesteps=np.asarray([0, 1, 0, 1], dtype=np.int64),
    )

    summary = inspect_dataset(path, env_id="CartPole-v1")
    frame = summary_to_frame(summary)

    assert summary["episodes"] == 2
    assert summary["transitions"] == 4
    assert summary["return_mean"] == 3.0
    assert summary["unique_actions"] == [0, 1]
    assert summary["checks"]["cartpole_actions_are_0_or_1"]
    assert not summary["checks"]["cartpole_full_length_expert"]
    assert frame.loc[0, "check_has_transitions"]
