#!/bin/bash
# Full experiment matrix: 2 envs x 3 algorithms x 4 dataset sizes x 3 seeds = 72 runs
# Usage: bash scripts/run_all.sh [cuda|cpu]
# Example: bash scripts/run_all.sh cuda

set -e

DEVICE=${1:-cuda}
ENV_PREFIX=/mnt/vita/scratch/vita-students/users/yunyi/conda/envs/imitation-learning
PYTHON="conda run -p $ENV_PREFIX python"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG_FILE="$ROOT/results/run_all_$(date +%Y%m%d_%H%M%S).log"

# PyTorch needs a writable cache dir in containerised environments where
# getpwuid() fails (uid not in /etc/passwd).
export TORCHINDUCTOR_CACHE_DIR=/mnt/vita/scratch/vita-students/users/yunyi/.cache/torchinductor
export TRITON_CACHE_DIR=/mnt/vita/scratch/vita-students/users/yunyi/.cache/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

mkdir -p "$ROOT/results"
cd "$ROOT"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

run() {
    log "RUN: $*"
    $PYTHON "$@" 2>&1 | tee -a "$LOG_FILE"
}

log "===== Starting full experiment matrix (device=$DEVICE) ====="

# ──────────────────────────────────────────────
# Step 1: Train expert policies
# ──────────────────────────────────────────────
log "--- Step 1: Expert training ---"

if [ ! -f "models/expert/CartPole-v1/expert_seed_0.zip" ]; then
    run scripts/train_expert.py --config configs/cartpole/base.yaml --seed 0
else
    log "SKIP: CartPole expert seed 0 already exists"
fi

for SEED in 1 2; do
    if [ ! -f "models/expert/CartPole-v1/expert_seed_${SEED}.zip" ]; then
        run scripts/train_expert.py --config configs/cartpole/base.yaml --seed $SEED
    else
        log "SKIP: CartPole expert seed $SEED already exists"
    fi
done

for SEED in 0 1 2; do
    if [ ! -f "models/expert/Pendulum-v1/expert_seed_${SEED}.zip" ]; then
        run scripts/train_expert.py --config configs/pendulum/base.yaml --seed $SEED
    else
        log "SKIP: Pendulum expert seed $SEED already exists"
    fi
done

# ──────────────────────────────────────────────
# Step 2: Generate expert datasets
# ──────────────────────────────────────────────
log "--- Step 2: Dataset generation ---"

for SEED in 0 1 2; do
    for TRAJ in 1 5 10 20; do
        if [ ! -f "data/expert/CartPole-v1/traj_${TRAJ}/seed_${SEED}.npz" ]; then
            run scripts/generate_dataset.py --config configs/cartpole/base.yaml \
                --seed $SEED --trajectories $TRAJ
        else
            log "SKIP: CartPole dataset traj=$TRAJ seed=$SEED already exists"
        fi

        if [ ! -f "data/expert/Pendulum-v1/traj_${TRAJ}/seed_${SEED}.npz" ]; then
            run scripts/generate_dataset.py --config configs/pendulum/base.yaml \
                --seed $SEED --trajectories $TRAJ
        else
            log "SKIP: Pendulum dataset traj=$TRAJ seed=$SEED already exists"
        fi
    done
done

# ──────────────────────────────────────────────
# Step 3: Imitation learning — CartPole-v1
# ──────────────────────────────────────────────
log "--- Step 3: Imitation learning (CartPole-v1) ---"

CARTPOLE_ALGORITHMS=(
    "iqlearn:configs/algorithms/iqlearn.yaml"
    "csil:configs/algorithms/csil.yaml"
    "soar_csil:configs/algorithms/soar_csil.yaml"
)

for ENTRY in "${CARTPOLE_ALGORITHMS[@]}"; do
    ALG="${ENTRY%%:*}"
    ALG_CFG="${ENTRY##*:}"
    for SEED in 0 1 2; do
        for TRAJ in 1 5 10 20; do
            OUT="models/imitation/CartPole-v1/${ALG}/traj_${TRAJ}/seed_${SEED}.pt"
            if [ ! -f "$OUT" ]; then
                log "Training CartPole | alg=$ALG | traj=$TRAJ | seed=$SEED"
                run scripts/train_imitation.py \
                    --env-config configs/cartpole/base.yaml \
                    --algorithm-config "$ALG_CFG" \
                    --seed $SEED --trajectories $TRAJ --device $DEVICE
            else
                log "SKIP: $OUT already exists"
            fi
        done
    done
done

# ──────────────────────────────────────────────
# Step 4: Imitation learning — Pendulum-v1
# ──────────────────────────────────────────────
log "--- Step 4: Imitation learning (Pendulum-v1) ---"

PENDULUM_ALGORITHMS=(
    "iqlearn_continuous:configs/algorithms/iqlearn_continuous.yaml"
    "csil_continuous:configs/algorithms/csil_continuous.yaml"
    "soar_csil_continuous:configs/algorithms/soar_csil_continuous.yaml"
)

for ENTRY in "${PENDULUM_ALGORITHMS[@]}"; do
    ALG="${ENTRY%%:*}"
    ALG_CFG="${ENTRY##*:}"
    for SEED in 0 1 2; do
        for TRAJ in 1 5 10 20; do
            OUT="models/imitation/Pendulum-v1/${ALG}/traj_${TRAJ}/seed_${SEED}.pt"
            if [ ! -f "$OUT" ]; then
                log "Training Pendulum | alg=$ALG | traj=$TRAJ | seed=$SEED"
                run scripts/train_imitation.py \
                    --env-config configs/pendulum/base.yaml \
                    --algorithm-config "$ALG_CFG" \
                    --seed $SEED --trajectories $TRAJ --device $DEVICE
            else
                log "SKIP: $OUT already exists"
            fi
        done
    done
done

log "===== All experiments completed. Log: $LOG_FILE ====="
