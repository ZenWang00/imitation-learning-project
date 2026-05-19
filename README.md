# Applied Project 3: Imitation Learning

This repository is the shared workspace for the imitation learning project.

## Project Goal

Compare three imitation learning variants on at least two classic-control environments:

1. `IQ-Learn`
2. One method from `CSIL`, `HyPE`, `f-IRL`, or `ML-IRL`
3. The SOAR-enhanced version of the chosen second method

Chosen project setup:

- Environments: `CartPole-v1` and `Pendulum-v1`
- Algorithms: `IQ-Learn`, `CSIL`, and `SOAR-CSIL`

## Required Workflow

1. Train an expert policy for each environment with an RL algorithm.
2. Roll out the trained expert policy to generate expert trajectories.
3. Build expert datasets with multiple sizes, for example `1`, `5`, `10`, and `20` trajectories.
4. Train each imitation learning algorithm on each dataset size.
5. Repeat the experiments over multiple random seeds.
6. Evaluate, aggregate, plot, and analyze the results.

## Suggested Experiment Matrix

| Dimension | Recommended Values |
| --- | --- |
| Environments | `CartPole-v1`, `Pendulum-v1` |
| Algorithms | `iqlearn`, `csil`, `soar_csil` |
| Expert dataset sizes | `1`, `5`, `10`, `20` trajectories |
| Seeds | `0`, `1`, `2` |

This gives `2 x 3 x 4 x 3 = 72` final training runs.

## Repository Layout

```text
.
├── configs/
│   ├── cartpole/
│   └── pendulum/
├── data/
│   ├── expert/
│   └── processed/
├── models/
│   ├── expert/
│   └── imitation/
├── notebooks/
├── results/
│   ├── figures/
│   ├── logs/
│   └── tables/
├── scripts/
├── src/
│   ├── datasets/
│   ├── envs/
│   ├── imitation/
│   ├── rl/
│   └── utils/
├── environment.yml
├── requirements.txt
└── README.md
```

## Team Workflow

Recommended split:

| Role | Responsibility |
| --- | --- |
| Member A | Expert policy training and dataset generation |
| Member B | IQ-Learn implementation and experiments |
| Member C | Baseline imitation algorithm implementation and experiments |
| Member D | SOAR-enhanced variant and result aggregation |
| Shared | Final report, plots, and presentation |

Recommended collaboration rules:

- Use feature branches instead of committing directly to `main`.
- Keep hyperparameters in config files rather than editing scripts by hand.
- Agree on seeds, evaluation frequency, result file names, and dataset sizes before running final experiments.
- Run one small debug experiment before launching the full matrix.
- Store large model checkpoints and datasets outside normal Git history if they become large.

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate imitation-learning
```

If conda is unavailable, install from `requirements.txt` in a Python 3.10 virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## First Milestones

1. Use `CartPole-v1` and `Pendulum-v1` as the fixed project environments.
2. Implement or import expert training scripts.
3. Train one expert policy per environment.
4. Generate expert datasets at all target sizes.
5. Standardize config files and result formats.
6. Run one end-to-end smoke test before scaling to the full experiment matrix.

## Environment Defaults

| Environment | Action Space | Recommended Expert RL Algorithm | Config |
| --- | --- | --- | --- |
| `CartPole-v1` | Discrete | `PPO` | `configs/cartpole/base.yaml` |
| `Pendulum-v1` | Continuous | `SAC` | `configs/pendulum/base.yaml` |

The shared experiment matrix lives in `configs/experiment_matrix.yaml`.

## Current Implementation Status

Implemented first:

- Discrete-action IQ-Learn core utilities in `src/imitation/iqlearn.py`
- Continuous-action IQ-Learn agent in `src/imitation/continuous_iqlearn_agent.py`
- Transition dataset utilities in `src/datasets/transitions.py`
- Pluggable imitation-agent interface and registry in `src/imitation/`
- End-to-end CartPole scripts for expert training, dataset generation, imitation training, and evaluation
- Unit tests for IQ-Learn helpers, dataset IO, registry construction, updates, and checkpoint reloads

The pipeline now supports the discrete `CartPole-v1` branch and has a continuous-action IQ-Learn implementation for `Pendulum-v1`.

## CartPole-v1 Commands

Train an expert:

```bash
conda run -n imitation-learning python scripts/train_expert.py --seed 0
```

Expert checkpoints are written periodically during training. Resume a long run from a checkpoint with:

```bash
conda run -n imitation-learning python scripts/train_expert.py --config configs/pendulum/base.yaml --seed 0 --resume-from models/expert/Pendulum-v1/seed_0_checkpoints/expert_20000_steps.zip
```

Generate one expert dataset:

```bash
conda run -n imitation-learning python scripts/generate_dataset.py --seed 0 --trajectories 1
```

Train IQ-Learn:

```bash
conda run -n imitation-learning python scripts/train_imitation.py --seed 0 --trajectories 1
```

Evaluate IQ-Learn:

```bash
conda run -n imitation-learning python scripts/evaluate.py --seed 0 --trajectories 1
```

For `Pendulum-v1`, use:

```bash
conda run -n imitation-learning python scripts/train_expert.py --config configs/pendulum/base.yaml --seed 0
conda run -n imitation-learning python scripts/generate_dataset.py --config configs/pendulum/base.yaml --seed 0 --trajectories 1
conda run -n imitation-learning python scripts/train_imitation.py --env-config configs/pendulum/base.yaml --algorithm-config configs/algorithms/iqlearn_continuous.yaml --seed 0 --trajectories 1
conda run -n imitation-learning python scripts/evaluate.py --env-config configs/pendulum/base.yaml --algorithm iqlearn_continuous --seed 0 --trajectories 1
```

## Suggested Result Format

Use a consistent naming convention such as:

```text
models/expert/{env}/expert_seed_{seed}.zip
models/expert/{env}/seed_{seed}_checkpoints/expert_{step}_steps.zip
data/expert/{env}/traj_{k}/seed_{seed}.npz
results/logs/{env}/{algorithm}/traj_{k}/seed_{seed}.csv
models/imitation/{env}/{algorithm}/traj_{k}/seed_{seed}.pt
```

Expert policies and expert datasets are shared across imitation algorithms. They are keyed by environment, dataset size, and seed, but not by imitation algorithm. This is intentional: `IQ-Learn`, `CSIL`, and `SOAR-CSIL` should train on the same expert demonstrations for a fair comparison.

Imitation outputs include the algorithm name in the path, so different algorithms do not overwrite each other:

```text
models/imitation/CartPole-v1/iqlearn/traj_5/seed_0.pt
models/imitation/CartPole-v1/csil/traj_5/seed_0.pt
results/logs/CartPole-v1/iqlearn/traj_5/seed_0.csv
results/logs/CartPole-v1/csil/traj_5/seed_0.csv
```

Rerunning the same environment, algorithm, dataset size, and seed overwrites that specific imitation checkpoint and log. Use a separate `--output-root` when comparing hyperparameter variants that should be kept side by side.

## Shared Artifacts

Large generated artifacts are not committed to Git. Use Google Drive for shared expert policies and expert datasets.

Current Google Drive layout:

```text
gdrive:imitation-learning-project-artifacts/
└── CartPole-v1/
    ├── cartpole_expert_artifacts.tar.gz
    ├── cartpole_expert_artifacts.sha256
    └── cartpole_expert_artifacts_README.txt
```

Install and configure `rclone`:

```bash
brew install rclone
rclone config
```

Create a Google Drive remote named `gdrive`. After configuration, verify access:

```bash
rclone listremotes
rclone lsl gdrive:imitation-learning-project-artifacts/CartPole-v1
```

Download the shared CartPole expert artifacts from the repository root:

```bash
rclone copy gdrive:imitation-learning-project-artifacts/CartPole-v1/cartpole_expert_artifacts.tar.gz .
rclone copy gdrive:imitation-learning-project-artifacts/CartPole-v1/cartpole_expert_artifacts.sha256 .
shasum -a 256 -c cartpole_expert_artifacts.sha256
tar -xzf cartpole_expert_artifacts.tar.gz
```

After extraction, the repository should contain:

```text
models/expert/CartPole-v1/expert_seed_{0,1,2}.zip
data/expert/CartPole-v1/traj_{1,5,10,20}/seed_{0,1,2}.npz
```

Upload a refreshed CartPole artifact bundle:

```bash
tar --exclude='models/expert/CartPole-v1/seed_*_checkpoints' \
  -czf artifacts/cartpole_expert_artifacts.tar.gz \
  models/expert/CartPole-v1 \
  data/expert/CartPole-v1

shasum -a 256 artifacts/cartpole_expert_artifacts.tar.gz > artifacts/cartpole_expert_artifacts.sha256

rclone copy artifacts gdrive:imitation-learning-project-artifacts/CartPole-v1 \
  --include 'cartpole_expert_artifacts*'
```

Each result row should at least record:

- environment
- algorithm
- dataset size
- seed
- training step
- evaluation return
- wall-clock time

## Practical Notes

- CPU is enough for development and smoke tests on these classic-control tasks.
- The full experiment matrix is better distributed across teammates or run on at least one GPU machine.
- Start with `1` seed during debugging, then use the agreed final seeds for report-quality results.
