#!/usr/bin/env bash
#SBATCH --job-name=tsmemory-train
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
# IMPORTANT: Don't override CUDA_VISIBLE_DEVICES from SLURM_JOB_GPUS.
# On this cluster SLURM_JOB_GPUS can contain physical IDs (e.g. "4") while the cgroup exposes the allocated GPU as
# local index 0; overriding would hide the GPU from torch.
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

# ---------------- dataset config (single source of truth) ----------------
# Change dataset by editing `script/dataset_env.sh` or submitting with:
#   sbatch --export=ALL,DATASET_NAME=ETTm1 script/train_memory_ts_quantile.sh
source "${REPO_DIR}/script/dataset_env.sh"

TEACHER_DIR="${TEACHER_DIR:-${TEACHER_SAVE_DIR}/train}"
VAL_TEACHER_DIR="${VAL_TEACHER_DIR:-${TEACHER_SAVE_DIR}/val}"
SAVE_DIR="${SAVE_DIR:-${MEMORY_SAVE_DIR:-}}"

if [ ! -f "${TEACHER_DIR}/manifest.json" ]; then
  echo "[error] missing train teacher manifest: ${TEACHER_DIR}/manifest.json" >&2
  exit 2
fi
if [ ! -f "${VAL_TEACHER_DIR}/manifest.json" ]; then
  echo "[error] missing val teacher manifest: ${VAL_TEACHER_DIR}/manifest.json" >&2
  exit 2
fi
if [ ! -f "${BASE_MODEL_PATH}/config.json" ]; then
  echo "[error] base model config not found: ${BASE_MODEL_PATH}/config.json" >&2
  exit 2
fi

BETA=${BETA:-0.5}
BETA_SCHEDULE=${BETA_SCHEDULE:-fixed}  # fixed|conf
BETA_MIN=${BETA_MIN:-0.1}
BETA_MAX=${BETA_MAX:-0.9}
CONF_TYPE=${CONF_TYPE:-w_max}          # w_max|entropy|effective_k
CONF_POWER=${CONF_POWER:-1.0}
DISTILL_TARGET=${DISTILL_TARGET:-absolute}  # absolute|delta|abs_delta|abs_tail_delta_med
# Only for DISTILL_TARGET in {abs_delta, abs_tail_delta_med}
DISTILL_ABS_WEIGHT=${DISTILL_ABS_WEIGHT:-1.0}
DISTILL_DELTA_WEIGHT=${DISTILL_DELTA_WEIGHT:-1.0}
ALIGN_LOSS=${ALIGN_LOSS:-huber}   # l2|huber
CROSSING_LAMBDA=${CROSSING_LAMBDA:-0.0}
MEMORY_TYPE="${MEMORY_TYPE:-context}"  # context|context_multivar|base_hidden_delta
BASE_HIDDEN_SOURCE=${BASE_HIDDEN_SOURCE:-encoder}  # encoder|decoder (only for base_hidden_delta)
MAX_TOKENS=${MAX_TOKENS:-256}                # only for base_hidden_delta
NUM_CHANNELS=${NUM_CHANNELS:-}               # only for context_multivar; empty = infer from teacher shards

# Optional: multi-horizon training on teacher built with max pred_len.
# Provide SPACE-separated lists via `sbatch --export` (Slurm uses commas as delimiters):
#   --export=ALL,TRAIN_PRED_LENS_LIST="96 192 336 720",VAL_PRED_LENS_LIST="96 192 336 720"
TRAIN_PRED_LENS_LIST="${TRAIN_PRED_LENS_LIST:-}"
TRAIN_PRED_LENS_PROBS="${TRAIN_PRED_LENS_PROBS:-}"   # comma- or space-separated, same length as TRAIN_PRED_LENS_LIST
VAL_PRED_LENS_LIST="${VAL_PRED_LENS_LIST:-}"

# MAE/median-friendly knobs (default off; enable via sbatch --export or by editing here)
TASK_CENTRAL_QUANTILE_WEIGHT=${TASK_CENTRAL_QUANTILE_WEIGHT:-0.0}
MEDIAN_TASK_LAMBDA=${MEDIAN_TASK_LAMBDA:-0.0}
MEDIAN_TASK_LOSS=${MEDIAN_TASK_LOSS:-huber}  # l1|huber|l2
BASE_ANCHOR_LAMBDA=${BASE_ANCHOR_LAMBDA:-0.0}
BASE_ANCHOR_LOSS=${BASE_ANCHOR_LOSS:-huber}  # l1|huber|l2
BASE_ANCHOR_GATE=${BASE_ANCHOR_GATE:-none}   # none|inv_conf
DISTILL_MED_GATE=${DISTILL_MED_GATE:-none}   # none|advantage (only for abs_tail_delta_med)
DISTILL_MED_ADV_MARGIN=${DISTILL_MED_ADV_MARGIN:-0.0}
SELECT_METRIC=${SELECT_METRIC:-mse_then_mae}  # mse_then_mae|mae_then_mse|mse|mae|sum
SELECT_MAE_WEIGHT=${SELECT_MAE_WEIGHT:-1.0}   # only for SELECT_METRIC=sum

LR=${LR:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
BATCH_SIZE=${BATCH_SIZE:-256}
EPOCHS=${EPOCHS:-20}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-0}
EARLY_STOP_MIN_EPOCHS=${EARLY_STOP_MIN_EPOCHS:-0}
SEED=${SEED:-42}

D_MODEL=${D_MODEL:-256}
N_HEADS=${N_HEADS:-8}
N_LAYERS=${N_LAYERS:-4}
N_DECODER_LAYERS=${N_DECODER_LAYERS:-2}
PATCH_LEN=${PATCH_LEN:-16}
DROPOUT=${DROPOUT:-0.1}
# ---------------------------------------------------

EXTRA_ARGS=(
  --beta_schedule "${BETA_SCHEDULE}"
  --beta_min "${BETA_MIN}"
  --beta_max "${BETA_MAX}"
  --conf_type "${CONF_TYPE}"
  --conf_power "${CONF_POWER}"
  --distill_target "${DISTILL_TARGET}"
  --distill_abs_weight "${DISTILL_ABS_WEIGHT}"
  --distill_delta_weight "${DISTILL_DELTA_WEIGHT}"
)
if [ -n "${TRAIN_PRED_LENS_LIST}" ]; then
  train_pred_csv="$(echo "${TRAIN_PRED_LENS_LIST}" | tr ',' ' ' | xargs | tr ' ' ',' | tr -s ',')"
  EXTRA_ARGS+=(--train_pred_lens "${train_pred_csv}")
fi
if [ -n "${TRAIN_PRED_LENS_PROBS}" ]; then
  train_prob_csv="$(echo "${TRAIN_PRED_LENS_PROBS}" | tr ',' ' ' | xargs | tr ' ' ',' | tr -s ',')"
  EXTRA_ARGS+=(--train_pred_lens_probs "${train_prob_csv}")
fi
if [ -n "${VAL_PRED_LENS_LIST}" ]; then
  val_pred_csv="$(echo "${VAL_PRED_LENS_LIST}" | tr ',' ' ' | xargs | tr ' ' ',' | tr -s ',')"
  EXTRA_ARGS+=(--val_pred_lens "${val_pred_csv}")
fi
if [ -n "${NUM_CHANNELS}" ]; then
  EXTRA_ARGS+=(--num_channels "${NUM_CHANNELS}")
fi

python -u train_memory_ts_quantile.py \
  --teacher_dir "${TEACHER_DIR}" \
  --val_teacher_dir "${VAL_TEACHER_DIR}" \
  --base_model_path "${BASE_MODEL_PATH}" \
  --context_len "${SEQ_LEN}" \
  --pred_len "${PRED_LEN}" \
  --memory_type "${MEMORY_TYPE}" \
  --base_hidden_source "${BASE_HIDDEN_SOURCE}" \
  --max_tokens "${MAX_TOKENS}" \
  --beta "${BETA}" \
  --align_loss "${ALIGN_LOSS}" \
  --crossing_lambda "${CROSSING_LAMBDA}" \
  --task_central_quantile_weight "${TASK_CENTRAL_QUANTILE_WEIGHT}" \
  --median_task_lambda "${MEDIAN_TASK_LAMBDA}" \
  --median_task_loss "${MEDIAN_TASK_LOSS}" \
  --base_anchor_lambda "${BASE_ANCHOR_LAMBDA}" \
  --base_anchor_loss "${BASE_ANCHOR_LOSS}" \
  --base_anchor_gate "${BASE_ANCHOR_GATE}" \
  --distill_med_gate "${DISTILL_MED_GATE}" \
  --distill_med_adv_margin "${DISTILL_MED_ADV_MARGIN}" \
  --select_metric "${SELECT_METRIC}" \
  --select_mae_weight "${SELECT_MAE_WEIGHT}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --early_stop_min_epochs "${EARLY_STOP_MIN_EPOCHS}" \
  --seed "${SEED}" \
  --d_model "${D_MODEL}" \
  --n_heads "${N_HEADS}" \
  --n_layers "${N_LAYERS}" \
  --n_decoder_layers "${N_DECODER_LAYERS}" \
  --patch_len "${PATCH_LEN}" \
  --dropout "${DROPOUT}" \
  --save_dir "${SAVE_DIR}" \
  "${EXTRA_ARGS[@]}"

# ---------------- optional cleanup ----------------
# Teacher shards can be very large (tens of GB per dataset/horizon). Since Memory training is the
# only consumer of teacher_ds, you can reclaim space by setting CLEAN_TEACHER_DS=1 at submit time:
#   sbatch --export=ALL,CLEAN_TEACHER_DS=1 script/train_memory_ts_quantile.sh
#
# Safety: only deletes directories under ${REPO_DIR}/teacher_ds and only if a checkpoint exists.
CLEAN_TEACHER_DS="${CLEAN_TEACHER_DS:-0}"
if [ "${CLEAN_TEACHER_DS}" = "1" ]; then
  if [ ! -f "${SAVE_DIR}/best.pth" ]; then
    echo "[cleanup] best.pth not found in ${SAVE_DIR}; refusing to delete teacher_ds." >&2
    exit 3
  fi

  # Preserve small manifests next to the trained checkpoint for traceability.
  for sp in train val test; do
    if [ -f "${TEACHER_SAVE_DIR}/${sp}/manifest.json" ]; then
      cp -f "${TEACHER_SAVE_DIR}/${sp}/manifest.json" "${SAVE_DIR}/teacher_manifest_${sp}.json"
    fi
  done

  teacher_real="$(realpath -m "${TEACHER_SAVE_DIR}")"
  root_real="$(realpath -m "${REPO_DIR}/teacher_ds")"
  if [[ "${teacher_real}" != "${root_real}/"* ]]; then
    echo "[cleanup] refusing to delete non-teacher_ds path: ${teacher_real}" >&2
    exit 3
  fi
  echo "[cleanup] deleting teacher_ds dir: ${teacher_real}"
  rm -rf "${teacher_real}"
fi
