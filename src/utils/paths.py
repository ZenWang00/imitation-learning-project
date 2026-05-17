from __future__ import annotations

from pathlib import Path


def expert_model_path(root: str | Path, env_id: str, seed: int) -> Path:
    return Path(root) / "models" / "expert" / env_id / f"expert_seed_{seed}.zip"


def expert_checkpoint_dir(root: str | Path, env_id: str, seed: int) -> Path:
    return Path(root) / "models" / "expert" / env_id / f"seed_{seed}_checkpoints"


def expert_dataset_path(root: str | Path, env_id: str, trajectories: int, seed: int) -> Path:
    return (
        Path(root)
        / "data"
        / "expert"
        / env_id
        / f"traj_{trajectories}"
        / f"seed_{seed}.npz"
    )


def imitation_model_path(
    root: str | Path,
    env_id: str,
    algorithm: str,
    trajectories: int,
    seed: int,
) -> Path:
    return (
        Path(root)
        / "models"
        / "imitation"
        / env_id
        / algorithm
        / f"traj_{trajectories}"
        / f"seed_{seed}.pt"
    )


def imitation_log_path(
    root: str | Path,
    env_id: str,
    algorithm: str,
    trajectories: int,
    seed: int,
) -> Path:
    return (
        Path(root)
        / "results"
        / "logs"
        / env_id
        / algorithm
        / f"traj_{trajectories}"
        / f"seed_{seed}.csv"
    )
