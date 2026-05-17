from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.imitation.continuous_iqlearn_agent import ContinuousIQLearnAgent
from src.imitation.iqlearn_agent import IQLearnAgent
from src.utils.config import load_yaml
from src.utils.evaluation import evaluate_policy
from src.utils.paths import imitation_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/cartpole/base.yaml")
    parser.add_argument("--algorithm", default="iqlearn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectories", type=int, required=True)
    parser.add_argument("--output-root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = load_yaml(args.env_config)
    path = imitation_model_path(
        args.output_root,
        env_config["env_id"],
        args.algorithm,
        args.trajectories,
        args.seed,
    )
    if args.algorithm == "iqlearn":
        agent = IQLearnAgent.load(path)
    elif args.algorithm == "iqlearn_continuous":
        agent = ContinuousIQLearnAgent.load(path)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
    mean_return, std_return = evaluate_policy(
        env_config["env_id"],
        lambda obs: agent.act(obs, deterministic=True),
        episodes=int(env_config["evaluation"]["episodes"]),
        seed=args.seed,
    )
    print(f"mean_return={mean_return:.2f} std_return={std_return:.2f}")


if __name__ == "__main__":
    main()
