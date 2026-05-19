from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.actions import format_action_for_env
from src.utils.config import load_yaml
from src.utils.evaluation import evaluate_policy
from src.utils.paths import expert_checkpoint_dir, expert_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cartpole/base.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--resume-from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    timesteps = args.timesteps or int(config["expert"]["total_timesteps"])
    env = gym.make(config["env_id"])
    algorithm = config["expert"]["algorithm"].lower()
    if algorithm == "ppo":
        model_cls = PPO
    elif algorithm == "sac":
        model_cls = SAC
    else:
        raise ValueError(f"Unsupported expert algorithm: {algorithm}")

    if args.resume_from:
        model = model_cls.load(args.resume_from, env=env)
    else:
        model = model_cls("MlpPolicy", env, seed=args.seed, verbose=0)

    checkpoint_dir = expert_checkpoint_dir(args.output_root, config["env_id"], args.seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=int(config["expert"]["checkpoint_freq"]),
        save_path=str(checkpoint_dir),
        name_prefix="expert",
    )
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=not bool(args.resume_from),
    )
    output_path = expert_model_path(args.output_root, config["env_id"], args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    env.close()

    mean_return, std_return = evaluate_policy(
        config["env_id"],
        lambda obs: format_action_for_env(
            model.predict(obs, deterministic=True)[0], env.action_space
        ),
        episodes=int(config["expert"]["eval_episodes"]),
        seed=args.seed,
    )
    print(
        f"saved={Path(output_path)} checkpoints={checkpoint_dir} "
        f"mean_return={mean_return:.2f} std_return={std_return:.2f}"
    )


if __name__ == "__main__":
    main()
