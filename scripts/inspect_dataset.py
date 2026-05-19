from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--output")
    return parser.parse_args()


def inspect_dataset(path: str | Path, env_id: str = "CartPole-v1") -> dict[str, Any]:
    dataset_path = Path(path)
    data = np.load(dataset_path)
    required_keys = {
        "observations",
        "actions",
        "next_observations",
        "dones",
        "rewards",
        "episode_ids",
        "timesteps",
    }
    missing = sorted(required_keys.difference(data.files))
    if missing:
        raise ValueError(f"Dataset {dataset_path} missing keys: {missing}")

    rewards = data["rewards"]
    episode_ids = data["episode_ids"]
    actions = data["actions"]

    episode_returns = []
    episode_lengths = []
    for episode_id in sorted(np.unique(episode_ids)):
        mask = episode_ids == episode_id
        episode_returns.append(float(rewards[mask].sum()))
        episode_lengths.append(int(mask.sum()))

    action_min = np.min(actions, axis=0).tolist()
    action_max = np.max(actions, axis=0).tolist()
    unique_actions = (
        sorted(np.unique(actions).astype(int).tolist())
        if np.issubdtype(actions.dtype, np.integer)
        else None
    )

    checks: dict[str, bool] = {
        "has_transitions": int(len(rewards)) > 0,
        "episode_count_matches_ids": len(episode_returns) == len(np.unique(episode_ids)),
        "has_matching_array_lengths": all(
            len(data[key]) == len(rewards)
            for key in [
                "observations",
                "actions",
                "next_observations",
                "dones",
                "episode_ids",
                "timesteps",
            ]
        ),
    }

    if env_id == "CartPole-v1":
        checks.update(
            {
                "cartpole_actions_are_0_or_1": unique_actions is not None
                and set(unique_actions).issubset({0, 1}),
                "cartpole_full_length_expert": bool(episode_lengths)
                and min(episode_lengths) >= 500,
            }
        )

    return {
        "path": str(dataset_path),
        "env_id": env_id,
        "episodes": int(len(episode_returns)),
        "transitions": int(len(rewards)),
        "return_mean": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns)),
        "return_min": float(np.min(episode_returns)),
        "return_max": float(np.max(episode_returns)),
        "length_mean": float(np.mean(episode_lengths)),
        "length_min": int(np.min(episode_lengths)),
        "length_max": int(np.max(episode_lengths)),
        "action_dtype": str(actions.dtype),
        "action_shape": tuple(actions.shape[1:]),
        "action_min": action_min,
        "action_max": action_max,
        "unique_actions": unique_actions,
        "checks": checks,
    }


def summary_to_frame(summary: dict[str, Any]) -> pd.DataFrame:
    flat_summary = {
        key: value
        for key, value in summary.items()
        if key not in {"checks", "action_min", "action_max", "unique_actions"}
    }
    flat_summary["action_min"] = summary["action_min"]
    flat_summary["action_max"] = summary["action_max"]
    flat_summary["unique_actions"] = summary["unique_actions"]
    flat_summary.update({f"check_{key}": value for key, value in summary["checks"].items()})
    return pd.DataFrame([flat_summary])


def main() -> None:
    args = parse_args()
    summary = inspect_dataset(args.path, env_id=args.env_id)
    frame = summary_to_frame(summary)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(f"saved={output_path}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

