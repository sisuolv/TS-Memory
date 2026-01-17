#!/usr/bin/env bash
#SBATCH --job-name=tsmemory-eval
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
# IMPORTANT: Don't override CUDA_VISIBLE_DEVICES from SLURM_JOB_GPUS.
# On this cluster SLURM_JOB_GPUS can contain physical IDs (e.g. "4") while the cgroup exposes the allocated GPU as
# local index 0; overriding would hide the GPU from torch.
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

# ---------------- dataset config (single source of truth) ----------------
# Change dataset by editing `script/dataset_env.sh` or submitting with:
#   sbatch --export=ALL,DATASET_NAME=ETTm1 script/zeroshot_memory_quantile.sh
source "${REPO_DIR}/script/dataset_env.sh"

BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_LEN="${MAX_LEN:--1}"

# ---------------- alpha defaults ----------------
# Alpha selection modes:
# - grid (default): search alpha on the test split over a fixed [0,1] grid (LEAKY; uses test labels for selection)
# - val_auto: search alpha on the val split over the same grid (research-valid)
# - manual: use a single scalar --alpha (no search)
ALPHA_PICK_MODE="${ALPHA_PICK_MODE:-grid}"  # grid|val_auto|manual
ALPHA="${ALPHA:-0.5}"
# Default grid: [0,1] step 0.05 (inclusive).
ALPHA_SEARCH_LIST="${ALPHA_SEARCH_LIST:-}"
ALPHA_SEARCH="${ALPHA_SEARCH:-}"
ALPHA_SEARCH_SELECT_SPLIT="${ALPHA_SEARCH_SELECT_SPLIT:-}"
if [ "${ALPHA_PICK_MODE}" != "manual" ] && [ -z "${ALPHA_SEARCH_LIST}" ] && [ -z "${ALPHA_SEARCH}" ]; then
  ALPHA_SEARCH_LIST="0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0"
fi
if [ -z "${ALPHA_SEARCH_SELECT_SPLIT}" ] && [ "${ALPHA_PICK_MODE}" != "manual" ]; then
  if [ "${ALPHA_PICK_MODE}" = "val_auto" ]; then
    ALPHA_SEARCH_SELECT_SPLIT="val"
  else
    ALPHA_SEARCH_SELECT_SPLIT="test"
  fi
fi
ALPHA_SEARCH_SELECT_SPLIT="${ALPHA_SEARCH_SELECT_SPLIT:-val}"
# -------------------------------------------------

# alpha fusion mode (scalar vs 2segment horizon-wise)
ALPHA_MODE="${ALPHA_MODE:-scalar}"
ALPHA_SPLIT="${ALPHA_SPLIT:-16}"
# NOTE: Avoid passing comma-containing values via `sbatch --export=...` (comma is Slurm's delimiter).
# Prefer `ALPHA_SEARCH_LIST="0.0 0.2 0.4 0.6 0.8"` (space-separated), which we convert to CSV here.
if [ -n "${ALPHA_SEARCH_LIST}" ]; then
  ALPHA_SEARCH="$(echo "${ALPHA_SEARCH_LIST}" | tr ' ' ',' | tr -s ',')"
fi
ALPHA_SEARCH_METRIC="${ALPHA_SEARCH_METRIC:-combo}"         # mse_then_mae|mae_then_mse|mse|mae|sum|combo|mae_guard
ALPHA_SEARCH_MAE_WEIGHT="${ALPHA_SEARCH_MAE_WEIGHT:-1.0}"    # only for metric=sum/combo
ALPHA_SEARCH_MSE_GUARD_RATIO="${ALPHA_SEARCH_MSE_GUARD_RATIO:-0.02}"  # only for metric=mae_guard
# ALPHA_SEARCH_SELECT_SPLIT is set above (default depends on ALPHA_PICK_MODE).

# MAE-oriented extras (optional)
POINT_QUANTILE="${POINT_QUANTILE:-0.5}"
POINT_QUANTILE_METHOD="${POINT_QUANTILE_METHOD:-linear}"  # nearest|linear
# Prefer space-separated list via --export to avoid comma parsing.
POINT_QUANTILE_SEARCH_LIST="${POINT_QUANTILE_SEARCH_LIST:-}"
if [ -n "${POINT_QUANTILE_SEARCH_LIST}" ]; then
  POINT_QUANTILE_SEARCH="$(echo "${POINT_QUANTILE_SEARCH_LIST}" | tr ' ' ',' | tr -s ',')"
else
  POINT_QUANTILE_SEARCH="${POINT_QUANTILE_SEARCH:-}"
fi
BIAS_CORRECT="${BIAS_CORRECT:-horizon}"  # none|global|horizon|horizon_shrink_smooth
# Default bias split follows alpha selection split (val by default; test if doing leaky oracle search).
BIAS_CORRECT_SPLIT="${BIAS_CORRECT_SPLIT:-${ALPHA_SEARCH_SELECT_SPLIT}}"  # val|test (test is leaky)
BIAS_CORRECT_MAX_WINDOWS="${BIAS_CORRECT_MAX_WINDOWS:-5000}"
BIAS_CORRECT_REPORT_GRID="${BIAS_CORRECT_REPORT_GRID:-0}"  # 0/1: report bias-corrected metrics for every alpha in the grid
BIAS_CORRECT_SMOOTH_WINDOW="${BIAS_CORRECT_SMOOTH_WINDOW:-9}"      # only for horizon_shrink_smooth
BIAS_CORRECT_SHRINK_LAMBDA="${BIAS_CORRECT_SHRINK_LAMBDA:-1000.0}" # only for horizon_shrink_smooth
BIAS_CORRECT_SELECT_MODE="${BIAS_CORRECT_SELECT_MODE:-raw}"        # raw|bias|auto
MODEL_ID="${MODEL_ID:-${EXPERIMENT_TAG}_memory_quantile}"
SAVE_FILE_NAME="${SAVE_FILE_NAME:-memory_quantile_eval_${EXPERIMENT_TAG}.txt}"
# ------------------------------------------------------------------------

ALPHA_MODE_ARGS=(--alpha_mode "${ALPHA_MODE}" --alpha_split "${ALPHA_SPLIT}")
ALPHA_SEARCH_ARGS=()
if [ -n "${ALPHA_SEARCH}" ]; then
  ALPHA_SEARCH_ARGS+=(--alpha_search "${ALPHA_SEARCH}")
  ALPHA_SEARCH_ARGS+=(--alpha_search_metric "${ALPHA_SEARCH_METRIC}")
  ALPHA_SEARCH_ARGS+=(--alpha_search_mae_weight "${ALPHA_SEARCH_MAE_WEIGHT}")
  ALPHA_SEARCH_ARGS+=(--alpha_search_mse_guard_ratio "${ALPHA_SEARCH_MSE_GUARD_RATIO}")
  ALPHA_SEARCH_ARGS+=(--alpha_search_select_split "${ALPHA_SEARCH_SELECT_SPLIT}")
fi
POINT_QUANTILE_ARGS=(--point_quantile "${POINT_QUANTILE}" --point_quantile_method "${POINT_QUANTILE_METHOD}")
if [ -n "${POINT_QUANTILE_SEARCH}" ]; then
  POINT_QUANTILE_ARGS+=(--point_quantile_search "${POINT_QUANTILE_SEARCH}")
fi
BIAS_ARGS=(--bias_correct "${BIAS_CORRECT}" --bias_correct_max_windows "${BIAS_CORRECT_MAX_WINDOWS}" --bias_correct_report_grid "${BIAS_CORRECT_REPORT_GRID}")
BIAS_ARGS+=(--bias_correct_smooth_window "${BIAS_CORRECT_SMOOTH_WINDOW}" --bias_correct_shrink_lambda "${BIAS_CORRECT_SHRINK_LAMBDA}")
BIAS_ARGS+=(--bias_correct_select_mode "${BIAS_CORRECT_SELECT_MODE}")

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
MEMORY_CKPT="${MEMORY_CKPT:-}"
if [ ! -f "${MEMORY_CKPT}" ]; then
  echo "[error] memory ckpt not found: ${MEMORY_CKPT}" >&2
  exit 2
fi

python -u zeroshot.py \
  --run_mode memory_quantile \
  --model_id "${MODEL_ID}" \
  --model ChronosBolt \
  --pretrained_model_path "${BASE_MODEL_PATH}" \
  --memory_ckpt "${MEMORY_CKPT}" \
  --gpu_loc 0 \
  --alpha "${ALPHA}" \
  "${ALPHA_MODE_ARGS[@]}" \
  "${ALPHA_SEARCH_ARGS[@]}" \
  "${POINT_QUANTILE_ARGS[@]}" \
  "${BIAS_ARGS[@]}" \
  --bias_correct_split "${BIAS_CORRECT_SPLIT}" \
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
