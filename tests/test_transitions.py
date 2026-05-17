from pathlib import Path

import numpy as np

from src.datasets.transitions import TransitionDataset


def make_dataset() -> TransitionDataset:
    return TransitionDataset(
        observations=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        actions=np.asarray([0, 1], dtype=np.int64),
        next_observations=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        dones=np.asarray([0.0, 1.0], dtype=np.float32),
        rewards=np.asarray([1.0, 1.0], dtype=np.float32),
        episode_ids=np.asarray([0, 0], dtype=np.int64),
        timesteps=np.asarray([0, 1], dtype=np.int64),
    )


def test_transition_dataset_round_trip(tmp_path: Path) -> None:
    dataset = make_dataset()
    path = tmp_path / "dataset.npz"

    dataset.save(path)
    loaded = TransitionDataset.load(path)

    assert len(loaded) == 2
    assert np.array_equal(loaded.actions, dataset.actions)
    assert np.array_equal(loaded.timesteps, dataset.timesteps)


def test_transition_dataset_samples_batch() -> None:
    dataset = make_dataset()
    batch = dataset.sample_batch(4, generator=np.random.default_rng(0))

    assert batch.observations.shape == (4, 2)
    assert batch.actions.shape == (4,)
    assert batch.rewards is not None

