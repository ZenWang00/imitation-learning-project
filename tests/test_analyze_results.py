from pathlib import Path

import pandas as pd

from scripts.analyze_results import aggregate_runs, load_run_summary


def test_load_run_summary_extracts_best_and_final_rows(tmp_path: Path) -> None:
    path = tmp_path / "seed_0.csv"
    pd.DataFrame(
        [
            {
                "step": 1,
                "eval_return_mean": 10.0,
                "eval_return_std": 1.0,
                "loss": 0.5,
                "seed": 0,
                "dataset_size": 5,
                "algorithm": "iqlearn",
                "env_id": "CartPole-v1",
            },
            {
                "step": 500,
                "eval_return_mean": 100.0,
                "eval_return_std": 2.0,
                "loss": 0.2,
                "seed": 0,
                "dataset_size": 5,
                "algorithm": "iqlearn",
                "env_id": "CartPole-v1",
            },
        ]
    ).to_csv(path, index=False)

    summary = load_run_summary(path)

    assert summary["best_step"] == 500
    assert summary["best_eval_return_mean"] == 100.0
    assert summary["final_eval_return_mean"] == 100.0


def test_aggregate_runs_groups_by_env_algorithm_and_dataset_size() -> None:
    frame = pd.DataFrame(
        [
            {
                "env_id": "CartPole-v1",
                "algorithm": "iqlearn",
                "dataset_size": 5,
                "seed": 0,
                "best_eval_return_mean": 100.0,
                "final_eval_return_mean": 80.0,
            },
            {
                "env_id": "CartPole-v1",
                "algorithm": "iqlearn",
                "dataset_size": 5,
                "seed": 1,
                "best_eval_return_mean": 200.0,
                "final_eval_return_mean": 100.0,
            },
        ]
    )

    aggregate = aggregate_runs(frame)

    assert aggregate.loc[0, "best_return_mean"] == 150.0
    assert aggregate.loc[0, "final_return_mean"] == 90.0
    assert aggregate.loc[0, "seeds"] == 2
