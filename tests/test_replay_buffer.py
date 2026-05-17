import numpy as np

from src.datasets.replay_buffer import ReplayBuffer


def test_replay_buffer_samples_batch() -> None:
    buffer = ReplayBuffer(capacity=4)
    buffer.add(
        observation=np.asarray([0.0, 1.0], dtype=np.float32),
        action=1,
        next_observation=np.asarray([1.0, 2.0], dtype=np.float32),
        done=0.0,
        reward=1.0,
    )

    batch = buffer.sample_batch(3, generator=np.random.default_rng(0))

    assert batch.observations.shape == (3, 2)
    assert batch.actions.tolist() == [1, 1, 1]
