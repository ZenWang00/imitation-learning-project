from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.replay_buffer import ReplayBuffer
from src.datasets.transitions import TransitionDataset
from src.envs.specs import infer_env_spec
from src.imitation.registry import build_algorithm
from src.utils.config import load_yaml
from src.utils.evaluation import evaluate_policy
from src.utils.paths import expert_dataset_path, imitation_log_path, imitation_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/cartpole/base.yaml")
    parser.add_argument("--algorithm-config", default="configs/algorithms/iqlearn.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectories", type=int, required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--dataset-root")
    parser.add_argument("--device", default=None, help="Override device in config (e.g. cuda, cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = load_yaml(args.env_config)
    algorithm_config = load_yaml(args.algorithm_config)
    if args.device is not None:
        algorithm_config["device"] = args.device
    dataset_root = args.dataset_root or args.output_root
    dataset = TransitionDataset.load(
        expert_dataset_path(dataset_root, env_config["env_id"], args.trajectories, args.seed)
    )
    env_spec = infer_env_spec(env_config["env_id"])

    agent = build_algorithm(
        algorithm_config["name"],
        observation_dim=env_spec.observation_dim,
        action_dim=env_spec.action_dim,
        action_type=env_spec.action_type,
        config=algorithm_config,
    )
    rng = np.random.default_rng(args.seed)
    device = algorithm_config.get("device", "cpu")
    updates = args.updates or int(algorithm_config["total_updates"])
    batch_size = int(algorithm_config["batch_size"])
    replay_buffer = ReplayBuffer(int(algorithm_config["replay_buffer_size"]))
    rollout_env = gym.make(env_config["env_id"])
    observation, _ = rollout_env.reset(seed=args.seed)
    for _ in range(int(algorithm_config["warmup_steps"])):
        action = rollout_env.action_space.sample()
        next_observation, reward, terminated, truncated, _ = rollout_env.step(action)
        done = terminated or truncated
        replay_buffer.add(observation, action, next_observation, done, reward)
        observation = next_observation
        if done:
            observation, _ = rollout_env.reset()
    log_path = imitation_log_path(
        args.output_root,
        env_config["env_id"],
        algorithm_config["name"],
        args.trajectories,
        args.seed,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(log_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "eval_return_mean",
                "eval_return_std",
                "loss",
                "expert_reward_mean",
                "expert_term",
                "replay_term",
                "regularizer",
                "q_mean",
                "q_abs_max",
                "seed",
                "dataset_size",
                "algorithm",
                "env_id",
            ],
        )
        writer.writeheader()
        last_metrics: dict[str, float] = {"loss": 0.0}
        best_return = float("-inf")
        evals_without_improvement = 0
        model_path = imitation_model_path(
            args.output_root,
            env_config["env_id"],
            algorithm_config["name"],
            args.trajectories,
            args.seed,
        )
        for step in range(1, updates + 1):
            for _ in range(int(algorithm_config["rollout_steps_per_iter"])):
                action = agent.act(observation, deterministic=False)
                next_observation, reward, terminated, truncated, _ = rollout_env.step(action)
                done = terminated or truncated
                replay_buffer.add(observation, action, next_observation, done, reward)
                observation = next_observation
                if done:
                    observation, _ = rollout_env.reset()

            expert_batch = dataset.sample_batch(batch_size, generator=rng, device=device)
            replay_batch = replay_buffer.sample_batch(batch_size, generator=rng, device=device)
            last_metrics = agent.update_with_replay(expert_batch, replay_batch)
            if (
                step == 1
                or step % int(algorithm_config["eval_interval"]) == 0
                or step == updates
            ):
                mean_return, std_return = evaluate_policy(
                    env_config["env_id"],
                    lambda obs: agent.act(obs, deterministic=True),
                    episodes=int(env_config["evaluation"]["episodes"]),
                    seed=args.seed,
                )
                writer.writerow(
                    {
                        "step": step,
                        "eval_return_mean": mean_return,
                        "eval_return_std": std_return,
                        "loss": last_metrics["loss"],
                        "expert_reward_mean": last_metrics["expert_reward_mean"],
                        "expert_term": last_metrics["expert_term"],
                        "replay_term": last_metrics["replay_term"],
                        "regularizer": last_metrics["regularizer"],
                        "q_mean": last_metrics["q_mean"],
                        "q_abs_max": last_metrics["q_abs_max"],
                        "seed": args.seed,
                        "dataset_size": args.trajectories,
                        "algorithm": algorithm_config["name"],
                        "env_id": env_config["env_id"],
                    }
                )
                if mean_return > best_return:
                    best_return = mean_return
                    evals_without_improvement = 0
                    agent.save(model_path)
                else:
                    evals_without_improvement += 1
                    if evals_without_improvement >= int(
                        algorithm_config["early_stopping_patience_evals"]
                    ):
                        break

    rollout_env.close()
    print(
        f"saved={model_path} log={log_path} "
        f"best_return={best_return:.2f} final_loss={last_metrics['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
