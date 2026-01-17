#!/usr/bin/env bash
# Chronos base (no retrieval, no memory) evaluation on Slurm GPU.

#SBATCH --job-name=chronos-base
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH -D .
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# NOTE: Slurm copies the sbatch script to /var/spool/... before execution, so
# ${BASH_SOURCE[0]} points to the spool path. Use working dir (set by #SBATCH -D)
# / SLURM_SUBMIT_DIR to find the repo checkout instead.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_DIR}"

mkdir -p logs results/forecast_evaluation

CONDA_INIT_PATH="${CONDA_INIT_PATH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tsrag}"
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

BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_LEN="${MAX_LEN:--1}"
MODEL_ID="${MODEL_ID:-${EXPERIMENT_TAG}_chronos_base}"
SAVE_FILE_NAME="${SAVE_FILE_NAME:-chronos_base_eval_${EXPERIMENT_TAG}.txt}"
# ------------------------------------------------------------------------

if [ ! -f "${ROOT_PATH}/${DATA_PATH}" ]; then
  echo "[error] dataset CSV not found: ${ROOT_PATH}/${DATA_PATH}" >&2
  exit 2
fi
if [ ! -f "${BASE_MODEL_PATH}/config.json" ]; then
  echo "[error] base model config not found: ${BASE_MODEL_PATH}/config.json" >&2
  exit 2
fi
# Base weights:
# - Our default TS-RAG base checkpoint uses autogluon_model.pth as a weight override.
# - Official HF checkpoints (e.g., checkpoints/amazon/*) may not have autogluon_model.pth; in that case, zeroshot.py
#   will fall back to the HF weights from from_pretrained.
REQUIRE_AUTOGLOON_WEIGHTS="${REQUIRE_AUTOGLOON_WEIGHTS:-1}"  # 1: require autogluon_model.pth; 0: allow missing
if [ ! -f "${BASE_MODEL_PATH}/autogluon_model.pth" ]; then
  if [ "${REQUIRE_AUTOGLOON_WEIGHTS}" = "1" ]; then
    echo "[error] base model weights not found: ${BASE_MODEL_PATH}/autogluon_model.pth" >&2
    echo "        Set REQUIRE_AUTOGLOON_WEIGHTS=0 to allow using HF-only checkpoints (e.g., checkpoints/amazon/*)." >&2
    exit 2
  fi
  echo "[warn] autogluon_model.pth not found at ${BASE_MODEL_PATH}/autogluon_model.pth; using HF weights only."
fi

python -u zeroshot.py \
  --run_mode base \
  --model_id "${MODEL_ID}" \
  --model ChronosBolt \
  --pretrained_model_path "${BASE_MODEL_PATH}" \
  --gpu_loc 0 \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --data "${DATA_CLASS}" \
  --features "${FEATURES}" \
  --target "${TARGET}" \
  --seq_len "${SEQ_LEN}" \
  --pred_len "${PRED_LEN}" \
  --label_len 0 \
  --batch_size "${BATCH_SIZE}" \
  --max_len "${MAX_LEN}" \
  --freq 0 \
  --percent 100 \
  --save_file_name "${SAVE_FILE_NAME}"
