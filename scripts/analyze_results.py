"""Aggregate results from all 72 runs and produce figures + tables.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --output-root .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Config ────────────────────────────────────────────────────────────────────

ENV_ALGORITHM_MAP = {
    "CartPole-v1": ["iqlearn", "csil", "soar_csil"],
    "Pendulum-v1": ["iqlearn_continuous", "csil_continuous", "soar_csil_continuous"],
}

DISPLAY_NAMES = {
    "iqlearn": "IQ-Learn",
    "iqlearn_continuous": "IQ-Learn",
    "csil": "CSIL",
    "csil_continuous": "CSIL",
    "soar_csil": "SOAR-CSIL",
    "soar_csil_continuous": "SOAR-CSIL",
}

ALGORITHM_COLORS = {
    "IQ-Learn": "#1f77b4",
    "CSIL": "#ff7f0e",
    "SOAR-CSIL": "#2ca02c",
}

TRAJ_SIZES = [1, 5, 10, 20]
SEEDS = [0, 1, 2]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_results(root: Path) -> pd.DataFrame:
    log_root = root / "results" / "logs"
    rows = []
    for env_id, algorithms in ENV_ALGORITHM_MAP.items():
        for algorithm in algorithms:
            for traj in TRAJ_SIZES:
                for seed in SEEDS:
                    path = log_root / env_id / algorithm / f"traj_{traj}" / f"seed_{seed}.csv"
                    if not path.exists():
                        continue
                    df = pd.read_csv(path)
                    df["env_id"] = env_id
                    df["algorithm"] = algorithm
                    df["display_name"] = DISPLAY_NAMES[algorithm]
                    df["traj"] = traj
                    df["seed"] = seed
                    rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No result CSVs found under {log_root}. "
            "Have you run scripts/run_all.sh yet?"
        )
    return pd.concat(rows, ignore_index=True)


def load_baselines(root: Path) -> dict | None:
    path = root / "results" / "baselines.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def best_return_per_run(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["env_id", "algorithm", "display_name", "traj", "seed"])["eval_return_mean"]
        .max()
        .reset_index()
        .rename(columns={"eval_return_mean": "best_return"})
    )


def aggregate_over_seeds(best: pd.DataFrame) -> pd.DataFrame:
    agg = (
        best.groupby(["env_id", "display_name", "traj"])["best_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "return_mean", "std": "return_std", "count": "n_seeds"})
    )
    agg["return_std"] = agg["return_std"].fillna(0.0)
    return agg


# ── Plot 1: data-efficiency curves ────────────────────────────────────────────

def plot_data_efficiency(agg: pd.DataFrame, out_dir: Path, baselines: dict | None = None) -> None:
    envs = agg["env_id"].unique()
    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 4.5), sharey=False)
    if len(envs) == 1:
        axes = [axes]

    for ax, env_id in zip(axes, sorted(envs)):
        env_data = agg[agg["env_id"] == env_id]
        for display_name in ["IQ-Learn", "CSIL", "SOAR-CSIL"]:
            alg_data = env_data[env_data["display_name"] == display_name].sort_values("traj")
            if alg_data.empty:
                continue
            color = ALGORITHM_COLORS[display_name]
            ax.plot(
                alg_data["traj"],
                alg_data["return_mean"],
                marker="o",
                label=display_name,
                color=color,
                linewidth=2,
            )
        if baselines and env_id in baselines:
            ax.axhline(
                baselines[env_id]["expert_mean"],
                color="black", linestyle="--", linewidth=1.5, label="Expert"
            )
            ax.axhline(
                baselines[env_id]["random_mean"],
                color="gray", linestyle=":", linewidth=1.5, label="Random"
            )
        ax.set_xlabel("Number of expert trajectories", fontsize=12)
        ax.set_ylabel("Mean episode return", fontsize=12)
        ax.set_title(env_id, fontsize=13, fontweight="bold")
        ax.set_xticks(TRAJ_SIZES)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Data Efficiency: Final Performance vs Expert Dataset Size", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "data_efficiency.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 2: learning curves ────────────────────────────────────────────────────

def _draw_learning_curves(axes, env_df, traj_sizes, expert_return=None):
    for ax, traj in zip(axes, traj_sizes):
        traj_df = env_df[env_df["traj"] == traj]
        for display_name in ["IQ-Learn", "CSIL", "SOAR-CSIL"]:
            alg_df = traj_df[traj_df["display_name"] == display_name]
            if alg_df.empty:
                continue
            color = ALGORITHM_COLORS[display_name]
            curve = (
                alg_df.groupby("step")["eval_return_mean"]
                .agg(["mean", "std"])
                .reset_index()
            )
            curve["std"] = curve["std"].fillna(0.0)
            ax.plot(curve["step"], curve["mean"], label=display_name, color=color, linewidth=1.8)
        if expert_return is not None:
            ax.axhline(expert_return, color="black", linestyle="--", linewidth=1.2, label="Expert")
        ax.set_title(f"{traj} traj.", fontsize=11)
        ax.set_xlabel("Training steps", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Eval return (mean)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)


def plot_learning_curves(df: pd.DataFrame, out_dir: Path, baselines: dict | None = None) -> None:
    envs = sorted(df["env_id"].unique())
    traj_sizes = sorted(df["traj"].unique())

    for env_id in envs:
        env_df = df[df["env_id"] == env_id]
        env_slug = env_id.replace("-", "_").replace(".", "").lower()
        expert_return = baselines[env_id]["expert_mean"] if baselines and env_id in baselines else None

        # 1x4 layout
        fig, axes = plt.subplots(1, 4, figsize=(5.5 * 4, 4), sharey=True)
        _draw_learning_curves(axes, env_df, traj_sizes, expert_return)
        fig.suptitle(f"Learning Curves — {env_id}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out_path = out_dir / f"learning_curves_{env_slug}_1x4.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

        # 2x2 layout
        fig, axes_grid = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
        _draw_learning_curves(axes_grid.flatten(), env_df, traj_sizes, expert_return)
        fig.suptitle(f"Learning Curves — {env_id}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out_path = out_dir / f"learning_curves_{env_slug}_2x2.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ── Plot 3: bar chart ─────────────────────────────────────────────────────────

def plot_bar_chart(agg: pd.DataFrame, out_dir: Path, baselines: dict | None = None) -> None:
    envs = sorted(agg["env_id"].unique())
    alg_names = ["IQ-Learn", "CSIL", "SOAR-CSIL"]
    n_algs = len(alg_names)
    x = np.arange(len(TRAJ_SIZES))
    width = 0.25

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 4.5), sharey=False)
    if len(envs) == 1:
        axes = [axes]

    for ax, env_id in zip(axes, envs):
        env_data = agg[agg["env_id"] == env_id]
        for i, display_name in enumerate(alg_names):
            alg_data = env_data[env_data["display_name"] == display_name].sort_values("traj")
            means = alg_data["return_mean"].values
            stds = alg_data["return_std"].values
            ax.bar(
                x + (i - 1) * width,
                means,
                width,
                label=display_name,
                color=ALGORITHM_COLORS[display_name],
                yerr=stds,
                capsize=4,
                error_kw={"linewidth": 1.2},
            )
        if baselines and env_id in baselines:
            ax.axhline(
                baselines[env_id]["expert_mean"],
                color="black", linestyle="--", linewidth=1.5, label="Expert"
            )
            ax.axhline(
                baselines[env_id]["random_mean"],
                color="gray", linestyle=":", linewidth=1.5, label="Random"
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t} traj." for t in TRAJ_SIZES], fontsize=10)
        ax.set_xlabel("Expert dataset size", fontsize=12)
        ax.set_ylabel("Mean episode return", fontsize=12)
        ax.set_title(env_id, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Final Performance by Algorithm and Dataset Size", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "bar_chart.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 4: normalized performance ───────────────────────────────────────────

def plot_normalized_performance(agg: pd.DataFrame, out_dir: Path, baselines: dict) -> None:
    rows = []
    for _, row in agg.iterrows():
        env_id = row["env_id"]
        b = baselines.get(env_id)
        if b is None:
            continue
        lo, hi = b["random_mean"], b["expert_mean"]
        span = hi - lo
        if abs(span) < 1e-6:
            continue
        norm_mean = (row["return_mean"] - lo) / span
        norm_std = row["return_std"] / abs(span)
        rows.append({**row.to_dict(), "norm_mean": norm_mean, "norm_std": norm_std})

    if not rows:
        print("No baseline data for normalized performance plot — skipping.")
        return

    norm_df = pd.DataFrame(rows)
    envs = sorted(norm_df["env_id"].unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 4.5), sharey=True)
    if len(envs) == 1:
        axes = [axes]

    for ax, env_id in zip(axes, envs):
        env_data = norm_df[norm_df["env_id"] == env_id]
        for display_name in ["IQ-Learn", "CSIL", "SOAR-CSIL"]:
            alg_data = env_data[env_data["display_name"] == display_name].sort_values("traj")
            if alg_data.empty:
                continue
            color = ALGORITHM_COLORS[display_name]
            ax.plot(
                alg_data["traj"],
                alg_data["norm_mean"],
                marker="o",
                label=display_name,
                color=color,
                linewidth=2,
            )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Expert")
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.2, label="Random")
        ax.set_xlabel("Number of expert trajectories", fontsize=12)
        ax.set_ylabel("Normalized return", fontsize=12)
        ax.set_title(env_id, fontsize=13, fontweight="bold")
        ax.set_xticks(TRAJ_SIZES)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Normalized Performance (0 = Random, 1 = Expert)", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "normalized_performance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 5: steps to threshold ───────────────────────────────────────────────

def _steps_to_threshold_df(df: pd.DataFrame, baselines: dict, threshold: float) -> pd.DataFrame:
    """For each (env, display_name, traj, seed), find the first step where
    normalized return >= threshold. DNF runs get the run's max step."""
    records = []
    for (env_id, display_name, traj, seed), group in df.groupby(
        ["env_id", "display_name", "traj", "seed"]
    ):
        b = baselines.get(env_id)
        if b is None:
            continue
        lo, hi = b["random_mean"], b["expert_mean"]
        span = hi - lo
        if abs(span) < 1e-6:
            continue
        target = lo + threshold * span
        sorted_group = group.sort_values("step")
        max_step = int(sorted_group["step"].max())
        reached = sorted_group[sorted_group["eval_return_mean"] >= target]
        if reached.empty:
            step = max_step
            dnf = True
        else:
            step = int(reached.iloc[0]["step"])
            dnf = False
        records.append({
            "env_id": env_id,
            "display_name": display_name,
            "traj": traj,
            "seed": seed,
            "step": step,
            "dnf": dnf,
            "max_step": max_step,
        })
    return pd.DataFrame(records)


def plot_steps_to_threshold(
    df: pd.DataFrame,
    out_dir: Path,
    baselines: dict,
    thresholds: list[float] = (0.5, 0.75),
) -> None:
    envs = sorted(df["env_id"].unique())
    alg_names = ["IQ-Learn", "CSIL", "SOAR-CSIL"]
    n_algs = len(alg_names)
    x = np.arange(len(TRAJ_SIZES))
    width = 0.25

    fig, axes_grid = plt.subplots(
        len(thresholds), len(envs),
        figsize=(6 * len(envs), 4.5 * len(thresholds)),
        sharey=False,
    )
    # Ensure 2-D indexing
    if len(thresholds) == 1:
        axes_grid = axes_grid[np.newaxis, :]
    if len(envs) == 1:
        axes_grid = axes_grid[:, np.newaxis]

    for row_i, threshold in enumerate(thresholds):
        stt = _steps_to_threshold_df(df, baselines, threshold)
        agg = (
            stt.groupby(["env_id", "display_name", "traj"])
            .agg(step_mean=("step", "mean"), step_std=("step", "std"), dnf_count=("dnf", "sum"), n=("dnf", "count"))
            .reset_index()
        )
        agg["step_std"] = agg["step_std"].fillna(0.0)

        for col_i, env_id in enumerate(envs):
            ax = axes_grid[row_i, col_i]
            env_data = agg[agg["env_id"] == env_id]

            for i, display_name in enumerate(alg_names):
                alg_data = env_data[env_data["display_name"] == display_name].sort_values("traj")
                means = alg_data["step_mean"].values
                stds = alg_data["step_std"].values
                dnf_counts = alg_data["dnf_count"].values
                ns = alg_data["n"].values

                bars = ax.bar(
                    x + (i - 1) * width,
                    means,
                    width,
                    label=display_name,
                    color=ALGORITHM_COLORS[display_name],
                    alpha=0.85,
                )
                # Annotate bars where any seed DNF'd
                for bar, dnf, n in zip(bars, dnf_counts, ns):
                    if dnf > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(stds) * 0.05,
                            f"DNF {int(dnf)}/{int(n)}",
                            ha="center", va="bottom", fontsize=6.5, color="dimgray",
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([f"{t} traj." for t in TRAJ_SIZES], fontsize=9)
            ax.set_xlabel("Expert dataset size", fontsize=11)
            ax.set_ylabel("Steps to reach threshold", fontsize=11)
            ax.set_title(
                f"{env_id}  —  {int(threshold * 100)}% of expert",
                fontsize=12, fontweight="bold",
            )
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Sample Efficiency: Steps to Reach X% of Expert Performance",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = out_dir / "steps_to_threshold.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Table: LaTeX + CSV ────────────────────────────────────────────────────────

def build_summary_table(agg: pd.DataFrame) -> pd.DataFrame:
    agg["value"] = agg.apply(
        lambda r: f"{r['return_mean']:.1f} ± {r['return_std']:.1f}", axis=1
    )
    pivot = agg.pivot_table(
        index=["env_id", "traj"],
        columns="display_name",
        values="value",
        aggfunc="first",
    )
    pivot = pivot.reindex(columns=["IQ-Learn", "CSIL", "SOAR-CSIL"])
    pivot.index.names = ["Environment", "Trajectories"]
    return pivot


def save_tables(agg: pd.DataFrame, out_dir: Path) -> None:
    table = build_summary_table(agg)

    csv_path = out_dir / "summary_table.csv"
    table.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    tex_path = out_dir / "summary_table.tex"
    latex = table.to_latex(
        caption="Mean ± std episode return across 3 seeds. Best value per row in bold.",
        label="tab:results",
        escape=False,
    )
    tex_path.write_text(latex)
    print(f"Saved: {tex_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    fig_dir = root / "results" / "figures"
    table_dir = root / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_all_results(root)
    print(f"  Loaded {len(df)} rows from {df[['env_id','algorithm','traj','seed']].drop_duplicates().shape[0]} runs")

    baselines = load_baselines(root)
    if baselines:
        print("  Loaded baselines from results/baselines.json")
    else:
        print("  No baselines found — run scripts/compute_baselines.py for expert/random reference lines")

    best = best_return_per_run(df)
    agg = aggregate_over_seeds(best)

    print("\nGenerating figures...")
    plot_data_efficiency(agg, fig_dir, baselines)
    plot_learning_curves(df, fig_dir, baselines)
    plot_bar_chart(agg, fig_dir, baselines)
    if baselines:
        plot_normalized_performance(agg, fig_dir, baselines)
        plot_steps_to_threshold(df, fig_dir, baselines)

    print("\nGenerating tables...")
    save_tables(agg, table_dir)

    print("\nDone. Summary:")
    print(build_summary_table(agg).to_string())


if __name__ == "__main__":
    main()
