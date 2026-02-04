#!/usr/bin/env bash
# Evaluate ChronosBolt + Memory on a single dataset, searching alpha on a fixed grid and reporting
# per-alpha test metrics (including alpha=0.0) across multiple pred_len values.
#
# Submit one dataset per job, e.g.:
#   sbatch --job-name tsmemory-etth1 --export=ALL,DATASET_NAME=ETTh1,MAX_LEN=-1 script/eval_alpha0_best_one_dataset.sh
#   sbatch --job-name tsmemory-etth2 --export=ALL,DATASET_NAME=ETTh2,MAX_LEN=-1 script/eval_alpha0_best_one_dataset.sh
#
# Optional overrides:
#   PRED_LENS="64 96 192 336 720" SEQ_LEN=512 BATCH_SIZE=256 NUM_WORKERS=8
#   OUT_SUBDIR=alpha_grid_runs FORCE=1
#   CKPT_ROOT=/path/to/checkpoints/memory_ts_quantile CKPT_METHOD=C_maeR_v2_m1_time
#
# Notes:
# - alpha selection inside zeroshot.py can select on val (research-valid) or test (leaky). Default below is grid-on-test.
#
#SBATCH --job-name=tsmemory-alpha0best
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
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

DATASET_NAME="${DATASET_NAME:?DATASET_NAME is required (e.g., ETTh1 or ETTh2)}"
SEQ_LEN="${SEQ_LEN:-512}"
PRED_LENS="${PRED_LENS:-${PRED_LENS_LIST:-64 96 192 336 720}}"
MAX_LEN="${MAX_LEN:--1}"  # -1 means full split
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FORCE="${FORCE:-0}"

# Resolve common dataset/base-model paths via the shared env script.
export DATASET_NAME
export SEQ_LEN
export PRED_LEN="${PRED_LEN:-64}"
source "${REPO_DIR}/script/dataset_env.sh"

# Resolve ckpt root (default: checkpoints/memory_ts_quantile under this repo checkout).
CKPT_ROOT="${CKPT_ROOT:-${REPO_DIR}/checkpoints/memory_ts_quantile}"
if [ -z "${CKPT_ROOT}" ] || [ ! -d "${CKPT_ROOT}" ]; then
  echo "[error] CKPT_ROOT not found. Set CKPT_ROOT to the checkpoints/memory_ts_quantile dir." >&2
  exit 2
fi

# Default checkpoint method profile per dataset (matches the MAE-R_v2 final profile).
CKPT_METHOD="${CKPT_METHOD:-}"
if [ -z "${CKPT_METHOD}" ]; then
  case "${DATASET_NAME}" in
    ETTh1) CKPT_METHOD="C_maeR_v2_m1_time" ;;
    ETTh2) CKPT_METHOD="C_maeR_v2_m2_stats" ;;
    *) CKPT_METHOD="C_maeR_v2_m123_all" ;;
  esac
fi

# Alpha selection:
# - grid: select best alpha on test (leaky), alpha candidates are [0,1] step 0.05
# - val_auto: select best alpha on val (research-valid), same grid
ALPHA_PICK_MODE="${ALPHA_PICK_MODE:-grid}" # grid|val_auto
ALPHA_SEARCH_LIST="${ALPHA_SEARCH_LIST:-0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0}"
ALPHA_SEARCH_SELECT_SPLIT="${ALPHA_SEARCH_SELECT_SPLIT:-}"
if [ -z "${ALPHA_SEARCH_SELECT_SPLIT}" ]; then
  if [ "${ALPHA_PICK_MODE}" = "val_auto" ]; then
    ALPHA_SEARCH_SELECT_SPLIT="val"
  else
    ALPHA_SEARCH_SELECT_SPLIT="test"
  fi
fi
ALPHA_SEARCH="$(echo "${ALPHA_SEARCH_LIST}" | tr ' ' ',' | tr -s ',')"

timestamp="$(date +%Y%m%d-%H%M%S)"
OUT_SUBDIR="${OUT_SUBDIR:-alpha_grid_${DATASET_NAME}_${timestamp}}"
mkdir -p "results/forecast_evaluation/${OUT_SUBDIR}"

echo "[run] DATASET_NAME=${DATASET_NAME} SEQ_LEN=${SEQ_LEN} PRED_LENS=${PRED_LENS} MAX_LEN=${MAX_LEN}"
echo "[run] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "[run] CKPT_ROOT=${CKPT_ROOT} CKPT_METHOD=${CKPT_METHOD}"
echo "[run] OUT_SUBDIR=results/forecast_evaluation/${OUT_SUBDIR}"
echo "[run] ALPHA_PICK_MODE=${ALPHA_PICK_MODE} ALPHA_SEARCH_SELECT_SPLIT=${ALPHA_SEARCH_SELECT_SPLIT} ALPHA_SEARCH_LIST=${ALPHA_SEARCH_LIST}"

for pl in ${PRED_LENS}; do
  ckpt="${CKPT_ROOT}/${DATASET_NAME}_sl${SEQ_LEN}_pl${pl}_${CKPT_METHOD}/best.pth"
  if [ ! -f "${ckpt}" ]; then
    echo "[error] memory ckpt not found: ${ckpt}" >&2
    exit 2
  fi

  save_name="${OUT_SUBDIR}/${DATASET_NAME}_sl${SEQ_LEN}_pl${pl}_alpha_grid_${ALPHA_PICK_MODE}.txt"
  if [ "${FORCE}" != "1" ] && [ -f "results/forecast_evaluation/${save_name}" ]; then
    echo "[skip] exists: results/forecast_evaluation/${save_name}"
    continue
  fi
  rm -f "results/forecast_evaluation/${save_name}" || true

  model_id="${DATASET_NAME}_sl${SEQ_LEN}_pl${pl}_alpha_grid_${ALPHA_PICK_MODE}"
  echo "[eval] dataset=${DATASET_NAME} pred_len=${pl} alpha_search_split=${ALPHA_SEARCH_SELECT_SPLIT} alpha_grid=[0,1]@0.05 ckpt=$(basename "$(dirname "${ckpt}")")"

  python -u zeroshot.py \
    --run_mode memory_quantile \
    --model_id "${model_id}" \
    --model ChronosBolt \
    --pretrained_model_path "${BASE_MODEL_PATH}" \
    --memory_ckpt "${ckpt}" \
    --gpu_loc 0 \
    --alpha 0.0 \
    --alpha_search "${ALPHA_SEARCH}" \
    --alpha_search_select_split "${ALPHA_SEARCH_SELECT_SPLIT}" \
    --alpha_search_metric mse_then_mae \
    --bias_correct none \
    --point_quantile 0.5 \
    --point_quantile_method nearest \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_PATH}" \
    --data "${DATA_CLASS}" \
    --features "${FEATURES}" \
    --target "${TARGET}" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${pl}" \
    --label_len 0 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_len "${MAX_LEN}" \
    --freq 0 \
    --percent 100 \
    --save_file_name "${save_name}"
done

echo "[done] results written under: results/forecast_evaluation/${OUT_SUBDIR}/"
