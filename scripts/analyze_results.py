from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", default="results/logs")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--env-id")
    parser.add_argument("--algorithm")
    return parser.parse_args()


def discover_log_files(logs_root: Path) -> list[Path]:
    return sorted(path for path in logs_root.glob("*/*/traj_*/seed_*.csv") if path.is_file())


def load_run_summary(path: Path) -> dict[str, float | int | str]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty log file: {path}")

    best_idx = frame["eval_return_mean"].idxmax()
    best_row = frame.loc[best_idx]
    final_row = frame.iloc[-1]

    env_id = str(frame["env_id"].iloc[0])
    algorithm = str(frame["algorithm"].iloc[0])
    dataset_size = int(frame["dataset_size"].iloc[0])
    seed = int(frame["seed"].iloc[0])

    return {
        "env_id": env_id,
        "algorithm": algorithm,
        "dataset_size": dataset_size,
        "seed": seed,
        "best_step": int(best_row["step"]),
        "best_eval_return_mean": float(best_row["eval_return_mean"]),
        "best_eval_return_std": float(best_row["eval_return_std"]),
        "final_step": int(final_row["step"]),
        "final_eval_return_mean": float(final_row["eval_return_mean"]),
        "final_loss": float(final_row["loss"]),
        "log_path": str(path),
    }


def aggregate_runs(run_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = run_frame.groupby(["env_id", "algorithm", "dataset_size"], as_index=False)
    return grouped.agg(
        best_return_mean=("best_eval_return_mean", "mean"),
        best_return_std=("best_eval_return_mean", "std"),
        best_return_min=("best_eval_return_mean", "min"),
        best_return_max=("best_eval_return_mean", "max"),
        final_return_mean=("final_eval_return_mean", "mean"),
        final_return_std=("final_eval_return_mean", "std"),
        seeds=("seed", "count"),
    )


def plot_best_return(aggregate_frame: pd.DataFrame, figures_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    sns.set_theme(style="whitegrid")

    for (env_id, algorithm), frame in aggregate_frame.groupby(["env_id", "algorithm"]):
        frame = frame.sort_values("dataset_size")
        plt.figure(figsize=(7, 4.5))
        plt.errorbar(
            frame["dataset_size"],
            frame["best_return_mean"],
            yerr=frame["best_return_std"].fillna(0.0),
            marker="o",
            capsize=4,
            linewidth=2,
        )
        plt.xlabel("Expert trajectories")
        plt.ylabel("Best evaluation return")
        plt.title(f"{env_id} - {algorithm}")
        plt.xticks(frame["dataset_size"])
        plt.tight_layout()

        output_path = figures_dir / f"{env_id}_{algorithm}_best_return.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        outputs.append(output_path)

    return outputs


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in discover_log_files(logs_root):
        summary = load_run_summary(path)
        if args.env_id and summary["env_id"] != args.env_id:
            continue
        if args.algorithm and summary["algorithm"] != args.algorithm:
            continue
        summaries.append(summary)

    if not summaries:
        raise ValueError(f"No matching logs found under {logs_root}")

    run_frame = pd.DataFrame(summaries).sort_values(
        ["env_id", "algorithm", "dataset_size", "seed"]
    )
    aggregate_frame = aggregate_runs(run_frame)

    runs_path = tables_dir / "run_summary.csv"
    aggregate_path = tables_dir / "aggregate_summary.csv"
    run_frame.to_csv(runs_path, index=False)
    aggregate_frame.to_csv(aggregate_path, index=False)
    figure_paths = plot_best_return(aggregate_frame, figures_dir)

    print(f"runs={len(run_frame)}")
    print(f"saved={runs_path}")
    print(f"saved={aggregate_path}")
    for figure_path in figure_paths:
        print(f"saved={figure_path}")


if __name__ == "__main__":
    main()

