"""Evaluate expert and random baselines and save to results/baselines.json.

Usage:
    python scripts/compute_baselines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.evaluation import evaluate_policy
from src.utils.paths import expert_model_path

CONFIGS = [
    {"env_id": "CartPole-v1", "algorithm": "ppo", "model_cls": PPO},
    {"env_id": "Pendulum-v1", "algorithm": "sac", "model_cls": SAC},
]
SEEDS = [0, 1, 2]
EPISODES = 10


def evaluate_expert(env_id: str, model_cls, seed: int) -> tuple[float, float]:
    path = expert_model_path(str(PROJECT_ROOT), env_id, seed)
    model = model_cls.load(path)
    return evaluate_policy(
        env_id,
        lambda obs: model.predict(obs, deterministic=True)[0],
        episodes=EPISODES,
        seed=seed,
    )


def evaluate_random(env_id: str, seed: int) -> tuple[float, float]:
    import gymnasium as gym
    env = gym.make(env_id)
    return evaluate_policy(
        env_id,
        lambda obs: env.action_space.sample(),
        episodes=EPISODES,
        seed=seed,
    )


def main() -> None:
    out_path = PROJECT_ROOT / "results" / "baselines.json"
    baselines: dict = {}

    for cfg in CONFIGS:
        env_id = cfg["env_id"]
        model_cls = cfg["model_cls"]
        print(f"\n=== {env_id} ===")

        expert_returns = []
        for seed in SEEDS:
            mean, std = evaluate_expert(env_id, model_cls, seed)
            expert_returns.append(mean)
            print(f"  expert seed={seed}: {mean:.2f} ± {std:.2f}")

        random_returns = []
        for seed in SEEDS:
            mean, std = evaluate_random(env_id, seed)
            random_returns.append(mean)
            print(f"  random seed={seed}: {mean:.2f} ± {std:.2f}")

        baselines[env_id] = {
            "expert_mean": float(np.mean(expert_returns)),
            "expert_std": float(np.std(expert_returns)),
            "random_mean": float(np.mean(random_returns)),
            "random_std": float(np.std(random_returns)),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baselines, indent=2))
    print(f"\nSaved: {out_path}")
    print(json.dumps(baselines, indent=2))


if __name__ == "__main__":
    main()
