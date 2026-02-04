#!/usr/bin/env bash
# Build retrieval_database embeddings cache (offline) using Chronos-Bolt base model.
#
# This is a **pre-step** for TS-Memory teacher building on datasets that don't yet have
# `../retrieval_database/<dataset>_*_<seq_len>.pkl` caches (e.g., electricity/traffic/exchange_rate).
#
# Supports feature sharding to parallelize large datasets:
#   sbatch --export=ALL,DATASET_NAME=traffic,SEQ_LEN=512,FEATURE_SHARD_TOTAL=8,FEATURE_SHARD_IDX=0 script/build_retrieval_database_chronosbolt.sbatch
#
# Run 8 shards (IDX=0..7) concurrently to utilize up to 8 GPUs.

#SBATCH --job-name=tsmemory-retrdb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH -D .
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_DIR}"

mkdir -p logs

CONDA_INIT_PATH="${CONDA_INIT_PATH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tsmemory}"
if [ -f "${CONDA_INIT_PATH}" ]; then
  # shellcheck disable=SC1090
  source "${CONDA_INIT_PATH}"
  if command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV_NAME}" || true
  fi
else
  echo "[warn] CONDA_INIT_PATH not found: ${CONDA_INIT_PATH}; assuming environment already active."
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "GPUs: ${SLURM_JOB_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

# ---------------- dataset config (single source of truth) ----------------
source "${REPO_DIR}/script/dataset_env.sh"

FEATURE_SHARD_TOTAL="${FEATURE_SHARD_TOTAL:-1}"
if [ -z "${FEATURE_SHARD_IDX:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  FEATURE_SHARD_IDX="${SLURM_ARRAY_TASK_ID}"
fi
FEATURE_SHARD_IDX="${FEATURE_SHARD_IDX:-0}"

BATCH_SIZE="${BATCH_SIZE:-512}"
DTYPE="${DTYPE:-float16}"   # float16 recommended to reduce disk for large datasets
OVERWRITE="${OVERWRITE:-0}" # set to 1 to overwrite existing embeddings

if [ ! -f "${ROOT_PATH}/${DATA_PATH}" ]; then
  echo "[error] dataset CSV not found: ${ROOT_PATH}/${DATA_PATH}" >&2
  exit 2
fi
if [ ! -f "${BASE_MODEL_PATH}/config.json" ]; then
  echo "[error] base model config not found: ${BASE_MODEL_PATH}/config.json" >&2
  exit 2
fi
# Base weights:
# - Some checkpoints include an optional autogluon_model.pth as a weight override.
# - If missing, scripts/build_retrieval_database_chronosbolt.py falls back to the weights loaded by from_pretrained.
REQUIRE_AUTOGLOON_WEIGHTS="${REQUIRE_AUTOGLOON_WEIGHTS:-1}"  # 1: require autogluon_model.pth; 0: allow missing
if [ ! -f "${BASE_MODEL_PATH}/autogluon_model.pth" ]; then
  if [ "${REQUIRE_AUTOGLOON_WEIGHTS}" = "1" ]; then
    echo "[error] base model weights not found: ${BASE_MODEL_PATH}/autogluon_model.pth" >&2
    echo "        Set REQUIRE_AUTOGLOON_WEIGHTS=0 to allow using checkpoints without an autogluon_model.pth override." >&2
    exit 2
  fi
  echo "[warn] autogluon_model.pth not found at ${BASE_MODEL_PATH}/autogluon_model.pth; using HF weights only."
fi

extra_args=()
if [ "${OVERWRITE}" = "1" ]; then
  extra_args+=(--overwrite)
fi

python -u scripts/build_retrieval_database_chronosbolt.py \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --seq_len "${SEQ_LEN}" \
  --base_model_path "${BASE_MODEL_PATH}" \
  --retrieval_database_dir "${RETRIEVAL_DATABASE_DIR}" \
  --dtype "${DTYPE}" \
  --batch_size "${BATCH_SIZE}" \
  --feature_shard_total "${FEATURE_SHARD_TOTAL}" \
  --feature_shard_idx "${FEATURE_SHARD_IDX}" \
  "${extra_args[@]}"
