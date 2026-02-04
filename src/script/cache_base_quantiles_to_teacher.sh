#!/usr/bin/env bash
#
# Cache base (ChronosBolt) quantile predictions (q_base) for a teacher split directory.
# This avoids per-batch base forward during Memory training for long pred_len (e.g., 720).
#
# Required env vars:
#   TEACHER_SPLIT_DIR, BASE_MODEL_PATH
#
# Optional:
#   PRED_LEN=720, BATCH_SIZE=256, DTYPE=fp16, FORCE=0, RESUME=1
#
#SBATCH --job-name=tsmemory-cache
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
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

: "${TEACHER_SPLIT_DIR:?TEACHER_SPLIT_DIR is required}"
: "${BASE_MODEL_PATH:?BASE_MODEL_PATH is required}"

PRED_LEN="${PRED_LEN:-720}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DTYPE="${DTYPE:-fp16}"
FORCE="${FORCE:-0}"
RESUME="${RESUME:-1}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

python -u scripts/cache_base_quantiles_to_teacher.py \
  --teacher_split_dir "${TEACHER_SPLIT_DIR}" \
  --base_model_path "${BASE_MODEL_PATH}" \
  --pred_len "${PRED_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --force "${FORCE}" \
  --resume "${RESUME}"
