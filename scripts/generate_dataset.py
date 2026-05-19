from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.transitions import TransitionDataset
from src.envs.actions import action_dtype_for_space, format_action_for_env
from src.utils.config import load_yaml
from src.utils.paths import expert_dataset_path, expert_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cartpole/base.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectories", type=int, required=True)
    parser.add_argument("--output-root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    algorithm = config["expert"]["algorithm"].lower()
    if algorithm == "ppo":
        model = PPO.load(expert_model_path(args.output_root, config["env_id"], args.seed))
    elif algorithm == "sac":
        model = SAC.load(expert_model_path(args.output_root, config["env_id"], args.seed))
    else:
        raise ValueError(f"Unsupported expert algorithm: {algorithm}")
    observations: list[np.ndarray] = []
    actions: list[np.ndarray | int] = []
    next_observations: list[np.ndarray] = []
    dones: list[float] = []
    rewards: list[float] = []
    episode_ids: list[int] = []
    timesteps: list[int] = []

    env = gym.make(config["env_id"])
    try:
        for episode_id in range(args.trajectories):
            observation, _ = env.reset(seed=args.seed + episode_id)
            terminated = truncated = False
            timestep = 0
            while not (terminated or truncated):
                predicted_action = model.predict(observation, deterministic=True)[0]
                action = format_action_for_env(predicted_action, env.action_space)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(action)
                next_observations.append(np.asarray(next_observation, dtype=np.float32))
                dones.append(float(terminated or truncated))
                rewards.append(float(reward))
                episode_ids.append(episode_id)
                timesteps.append(timestep)
                observation = next_observation
                timestep += 1
    finally:
        env.close()

    dataset = TransitionDataset(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=action_dtype_for_space(env.action_space)),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        timesteps=np.asarray(timesteps, dtype=np.int64),
    )
    output_path = expert_dataset_path(
        args.output_root, config["env_id"], args.trajectories, args.seed
    )
    dataset.save(output_path)
    print(f"saved={output_path} transitions={len(dataset)}")


if __name__ == "__main__":
    main()
