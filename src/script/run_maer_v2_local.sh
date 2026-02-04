#!/usr/bin/env bash
set -euo pipefail

# Local (non-Slurm) runner for the MAE-R_v2 "final profile" pipeline.
#
# This runs the same 4 stages as the Slurm submit script, sequentially:
#   1) build retrieval_database (ChronosBolt embeddings)
#   2) build teacher_ds (train+val)
#   3) cache q_base into teacher_ds (train+val) [optional but recommended]
#   4) train memory per pred_len
#   5) eval memory_quantile with alpha sweep (raw + bc)
#
# Usage examples:
#   DATASET_NAME=ETTh1 PRED_LENS="96 192" bash script/run_maer_v2_local.sh
#   DATASET_NAME=traffic TEACHER_QUERY_STRIDE=32 PRED_LENS="96" bash script/run_maer_v2_local.sh
#
# Logs are written under: logs/local/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p logs/local results/forecast_evaluation

timestamp="$(date +%Y%m%d-%H%M%S)"

# Core knobs (override via env)
DATASET_NAME="${DATASET_NAME:-ETTh1}"
SEQ_LEN="${SEQ_LEN:-512}"
PRED_LENS="${PRED_LENS:-96}"
TEACHER_PRED_LEN="${TEACHER_PRED_LEN:-720}"
CACHE_Q_BASE="${CACHE_Q_BASE:-1}"

# Resolve DATA_ROOT (raw CSV root).
DATA_ROOT="${DATA_ROOT:-}"
if [ -z "${DATA_ROOT}" ]; then
  for candidate in \
    "${REPO_DIR}/../all_datasets" \
    "${REPO_DIR}/../../all_datasets" \
    ; do
    if [ -d "${candidate}" ]; then
      DATA_ROOT="${candidate}"
      break
    fi
  done
fi
if [ -z "${DATA_ROOT}" ]; then
  echo "[error] DATA_ROOT not set and default all_datasets not found." >&2
  exit 2
fi

# Resolve base (ChronosBolt) checkpoint dir.
if [ -z "${BASE_MODEL_PATH:-}" ]; then
  for candidate in \
    "${REPO_DIR}/checkpoints/base" \
    "${REPO_DIR}/../checkpoints/base" \
    "${REPO_DIR}/../../checkpoints/base" \
    ; do
    if [ -f "${candidate}/config.json" ]; then
      BASE_MODEL_PATH="${candidate}"
      break
    fi
  done
fi
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${REPO_DIR}/checkpoints/base}"

TEACHER_TAG="${TEACHER_TAG:-${DATASET_NAME}_sl${SEQ_LEN}_pl${TEACHER_PRED_LEN}_T_local}"
EVAL_SUBDIR="${EVAL_SUBDIR:-local_${timestamp}}"

export DATASET_NAME
export SEQ_LEN
export DATA_ROOT
export BASE_MODEL_PATH
export TEACHER_TAG

mkdir -p "results/forecast_evaluation/${EVAL_SUBDIR}"

# Stage 0: retrieval_db
echo "[local] build retrieval_db: ${DATASET_NAME} sl${SEQ_LEN}" | tee "logs/local/${timestamp}_retrdb_${DATASET_NAME}.log"
bash script/build_retrieval_database_chronosbolt.sh \
  2>&1 | tee -a "logs/local/${timestamp}_retrdb_${DATASET_NAME}.log"

# Stage 1: teacher train/val
for split in train val; do
  echo "[local] build teacher: ${DATASET_NAME} split=${split} pl=${TEACHER_PRED_LEN}" | tee "logs/local/${timestamp}_teacher_${DATASET_NAME}_${split}.log"
  SPLIT="${split}" PRED_LEN="${TEACHER_PRED_LEN}" bash script/build_teacher_ts_quantiles.sh \
    2>&1 | tee -a "logs/local/${timestamp}_teacher_${DATASET_NAME}_${split}.log"
done

# Stage 2: cache q_base (optional)
if [ "${CACHE_Q_BASE}" = "1" ]; then
  for split in train val; do
    teacher_split_dir="teacher_ds/${TEACHER_TAG}/$(printf "%s" "${split}")"
    echo "[local] cache q_base: ${DATASET_NAME} split=${split} dir=${teacher_split_dir}" | tee "logs/local/${timestamp}_qbase_${DATASET_NAME}_${split}.log"
    TEACHER_SPLIT_DIR="${teacher_split_dir}" BASE_MODEL_PATH="${BASE_MODEL_PATH}" PRED_LEN="${TEACHER_PRED_LEN}" bash script/cache_base_quantiles_to_teacher.sh \
      2>&1 | tee -a "logs/local/${timestamp}_qbase_${DATASET_NAME}_${split}.log"
  done
fi

# Stage 3/4: train + eval per pred_len
for pred_len in ${PRED_LENS}; do
  scheme_tag="maeR_v2_local"
  train_tag="${DATASET_NAME}_sl${SEQ_LEN}_pl${pred_len}_C_${scheme_tag}"
  eval_tag="${DATASET_NAME}_sl${SEQ_LEN}_pl${pred_len}_E_${scheme_tag}_alphaSweepTest"

  echo "[local] train memory: ${DATASET_NAME} pl=${pred_len}" | tee "logs/local/${timestamp}_train_${DATASET_NAME}_pl${pred_len}.log"
  EXPERIMENT_TAG="${train_tag}" MEMORY_SOURCE_TAG="${train_tag}" PRED_LEN="${pred_len}" bash script/train_memory_ts_quantile.sh \
    2>&1 | tee -a "logs/local/${timestamp}_train_${DATASET_NAME}_pl${pred_len}.log"

  echo "[local] eval memory: ${DATASET_NAME} pl=${pred_len}" | tee "logs/local/${timestamp}_eval_${DATASET_NAME}_pl${pred_len}.log"
  save_name="${EVAL_SUBDIR}/memory_quantile_eval_${eval_tag}.txt"
  EXPERIMENT_TAG="${eval_tag}" MEMORY_SOURCE_TAG="${train_tag}" SAVE_FILE_NAME="${save_name}" PRED_LEN="${pred_len}" bash script/zeroshot_memory_quantile.sh \
    2>&1 | tee -a "logs/local/${timestamp}_eval_${DATASET_NAME}_pl${pred_len}.log"
done

echo "[local] done: ${DATASET_NAME} pred_lens=${PRED_LENS} (logs/local/${timestamp}_*)" | tee -a "logs/local/${timestamp}_run_done.log"
