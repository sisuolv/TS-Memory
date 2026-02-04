#!/usr/bin/env bash
# Shared dataset config for TS-Memory Slurm pipeline.
# Usage:
#   - Edit DATASET_NAME below, OR submit with `sbatch --export=ALL,DATASET_NAME=ETTm1 ...`
#   - All 3 sbatch scripts source this file so you only change config in one place.

set -euo pipefail

# Repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Data root: directory containing dataset subfolders, e.g.
#   <DATA_ROOT>/ETT-small/ETTh1.csv
#   <DATA_ROOT>/weather/weather.csv
#   <DATA_ROOT>/traffic/traffic.csv
#
# Default: try to auto-detect a nearby all_datasets directory.
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
  echo "[dataset_env] DATA_ROOT is not set and no default all_datasets dir found." >&2
  echo "             Set DATA_ROOT to a folder like <...>/all_datasets (containing ETT-small/, weather/, ...)." >&2
  exit 2
fi

# ---------------- dataset switch (edit here) ----------------
DATASET_NAME="${DATASET_NAME:-ETTh1}"    # ETTh1|ETTh2|ETTm1|ETTm2|weather|traffic|electricity|exchange_rate|custom

# Default forecasting setup
SEQ_LEN="${SEQ_LEN:-512}"
PRED_LEN="${PRED_LEN:-64}"
LABEL_LEN="${LABEL_LEN:-0}"
LOOKBACK_LENGTH="${LOOKBACK_LENGTH:-${SEQ_LEN}}"

# Experiment tag used to avoid overwriting outputs when sweeping horizons.
# Default keeps backward-compatible paths (tag == dataset name).
EXPERIMENT_TAG="${EXPERIMENT_TAG:-${DATASET_NAME}}"
# Optional: decouple teacher dataset tag from memory/eval tag.
# This lets you reuse a single teacher build across multiple memory variants.
TEACHER_TAG="${TEACHER_TAG:-${EXPERIMENT_TAG}}"

# Datasets are standardized with StandardScaler; keep scaling consistent across teacher/train/eval.
FEATURES="${FEATURES:-M}"   # M yields per-feature univariate samples.
# NOTE: Some custom datasets (e.g., electricity/traffic from all_datasets) don't have an explicit "OT" column.
# We set a dataset-aware default TARGET later after we resolve the CSV path.
TARGET="${TARGET:-}"

# Retrieval-distillation (teacher) defaults
K="${K:-10}"
TAU="${TAU:-0.07}"
DROP_SELF_EPS="${DROP_SELF_EPS:-1e-8}"
DROP_SELF_MODE="${DROP_SELF_MODE:-dist}"                 # dist|idx
DIST_TRANSFORM="${DIST_TRANSFORM:-none}"                 # none|sqrt
DISTANCE_METRIC="${DISTANCE_METRIC:-l2}"                 # l2|cosine
KB_SPLIT="${KB_SPLIT:-train}"                      # hard constraint (leakage prevention)
QUERY_EMBEDDING_SOURCE="${QUERY_EMBEDDING_SOURCE:-cache}"  # cache|model ; cache is fully offline (recommended)
# Teacher sample mode
SAMPLE_MODE="${SAMPLE_MODE:-univariate}"                 # univariate|multivariate
RETRIEVAL_FEATURE_ID="${RETRIEVAL_FEATURE_ID:-}"         # only for multivariate teacher; empty = auto
# Teacher type
TEACHER_MODE="${TEACHER_MODE:-weighted_quantile}"  # weighted_quantile|rag_output
RAG_MODEL_CKPT="${RAG_MODEL_CKPT:-${REPO_DIR}/checkpoints/chronos-bolt/best.pth}"
RAG_AUGMENT_MODE="${RAG_AUGMENT_MODE:-moe2}"
# Teacher alignment / rerank (optional; default keeps old behavior)
TEACHER_ALIGN="${TEACHER_ALIGN:-none}"             # none|retrieved_to_query|shift_last|shift_mean_last_m
SHIFT_LAST_M="${SHIFT_LAST_M:-16}"                 # only for shift_mean_last_m and raw rerank
RERANK_MODE="${RERANK_MODE:-none}"                 # none|raw_l1_shift|raw_l2_shift
RERANK_K0="${RERANK_K0:-0}"                        # 0 disables rerank
RERANK_WEIGHT_SOURCE="${RERANK_WEIGHT_SOURCE:-embedding}"  # embedding|raw
RERANK_TAU="${RERANK_TAU:-}"                       # empty = default to TAU inside python
# Optional subsampling of query windows to keep teacher size manageable on very wide datasets.
# Per-dataset defaults are set later; override via sbatch --export if needed.
TEACHER_QUERY_STRIDE="${TEACHER_QUERY_STRIDE:-}"

# Paths
# Base model checkpoint dir (ChronosBolt). If not set, try common workspace locations.
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

# Retrieval database output/cache dir (built from raw CSV by script/build_retrieval_database_chronosbolt.sh).
RETRIEVAL_DATABASE_DIR="${RETRIEVAL_DATABASE_DIR:-${REPO_DIR}/retrieval_database}"
TEACHER_SAVE_DIR="${TEACHER_SAVE_DIR:-${REPO_DIR}/teacher_ds/${TEACHER_TAG}}"
# Allow evaluating shorter horizons using a single long-horizon Memory checkpoint:
#   - Set EXPERIMENT_TAG per eval horizon (for result file naming)
#   - Set MEMORY_SOURCE_TAG to the training tag that owns the checkpoint
MEMORY_SOURCE_TAG="${MEMORY_SOURCE_TAG:-${EXPERIMENT_TAG}}"
MEMORY_SAVE_DIR="${MEMORY_SAVE_DIR:-${REPO_DIR}/checkpoints/memory_ts_quantile/${MEMORY_SOURCE_TAG}}"
MEMORY_CKPT="${MEMORY_CKPT:-${MEMORY_SAVE_DIR}/best.pth}"

# Memory training knobs (defaults keep old behavior)
BETA_SCHEDULE="${BETA_SCHEDULE:-fixed}"  # fixed|conf
BETA_MIN="${BETA_MIN:-0.1}"
BETA_MAX="${BETA_MAX:-0.9}"
CONF_TYPE="${CONF_TYPE:-w_max}"         # w_max|entropy|effective_k
CONF_POWER="${CONF_POWER:-1.0}"
DISTILL_TARGET="${DISTILL_TARGET:-absolute}"  # absolute|delta|abs_delta|abs_tail_delta_med
# Only used when DISTILL_TARGET in {abs_delta, abs_tail_delta_med} (unified objective).
DISTILL_ABS_WEIGHT="${DISTILL_ABS_WEIGHT:-1.0}"
DISTILL_DELTA_WEIGHT="${DISTILL_DELTA_WEIGHT:-1.0}"
# Optional MAE-v2 knobs (training-side)
BASE_ANCHOR_LAMBDA="${BASE_ANCHOR_LAMBDA:-0.0}"
BASE_ANCHOR_LOSS="${BASE_ANCHOR_LOSS:-huber}"      # l1|huber|l2
BASE_ANCHOR_GATE="${BASE_ANCHOR_GATE:-none}"       # none|inv_conf
DISTILL_MED_GATE="${DISTILL_MED_GATE:-none}"       # none|advantage (only for distill_target=abs_tail_delta_med)
DISTILL_MED_ADV_MARGIN="${DISTILL_MED_ADV_MARGIN:-0.0}"
NUM_CHANNELS="${NUM_CHANNELS:-}"        # only for context_multivar; empty = infer from teacher shards

# Dataset-specific root + dataloader class (non-retrieve).
case "${DATASET_NAME}" in
  ETTh1|ETTh2)
    ROOT_PATH="${ROOT_PATH:-${DATA_ROOT}/ETT-small}"
    DATA_PATH="${DATA_PATH:-${DATASET_NAME}.csv}"
    DATA_CLASS="${DATA_CLASS:-ett_h}"
    TARGET="${TARGET:-OT}"
    TEACHER_QUERY_STRIDE="${TEACHER_QUERY_STRIDE:-1}"
    ;;
  ETTm1|ETTm2)
    ROOT_PATH="${ROOT_PATH:-${DATA_ROOT}/ETT-small}"
    DATA_PATH="${DATA_PATH:-${DATASET_NAME}.csv}"
    DATA_CLASS="${DATA_CLASS:-ett_m}"
    TARGET="${TARGET:-OT}"
    TEACHER_QUERY_STRIDE="${TEACHER_QUERY_STRIDE:-1}"
    ;;
  weather|traffic|electricity|exchange_rate)
    ROOT_PATH="${ROOT_PATH:-${DATA_ROOT}/${DATASET_NAME}}"
    DATA_PATH="${DATA_PATH:-${DATASET_NAME}.csv}"
    DATA_CLASS="${DATA_CLASS:-custom}"

    # Dataset_Custom_S requires `target` to exist even when FEATURES=M; if OT is missing,
    # pick the last column name as a stable default.
    if [ -z "${TARGET}" ]; then
      header="$(head -n 1 "${ROOT_PATH}/${DATA_PATH}" | tr -d '\r')"
      if printf "%s\n" "${header}" | tr ',' '\n' | tr -d '\r' | grep -Fxq "OT"; then
        TARGET="OT"
      else
        TARGET="$(printf "%s\n" "${header}" | awk -F',' '{print $NF}' | tr -d '\r')"
        echo "[dataset_env] OT column not found; default TARGET=${TARGET}"
      fi
    fi

    # Default teacher subsampling for very wide datasets to avoid TB-scale teacher shards.
    # Override with `sbatch --export=ALL,TEACHER_QUERY_STRIDE=1 ...` to build full teachers.
    case "${DATASET_NAME}" in
      traffic)
        TEACHER_QUERY_STRIDE="${TEACHER_QUERY_STRIDE:-32}"
        ;;
      electricity)
        TEACHER_QUERY_STRIDE="${TEACHER_QUERY_STRIDE:-16}"
        ;;
      exchange_rate)
        TEACHER_QUERY_STRIDE="${TEACHER_QUERY_STRIDE:-1}"
        ;;
      weather)
        TEACHER_QUERY_STRIDE="${TEACHER_QUERY_STRIDE:-1}"
        ;;
    esac
    ;;
  custom)
    # Generic "custom" dataset (you must provide ROOT_PATH and DATA_PATH via env or by editing this file).
    if [ -z "${ROOT_PATH:-}" ] || [ -z "${DATA_PATH:-}" ]; then
      echo "[dataset_env] DATASET_NAME=custom requires ROOT_PATH and DATA_PATH to be set." >&2
      exit 2
    fi
    DATA_CLASS="${DATA_CLASS:-custom}"
    ;;
  *)
    echo "[dataset_env] Unsupported DATASET_NAME=${DATASET_NAME}" >&2
    exit 2
    ;;
esac

# Final fallback for safety.
TEACHER_QUERY_STRIDE="${TEACHER_QUERY_STRIDE:-1}"

# Keep a small echo for logs (helps debugging sbatch exports).
echo "[dataset_env] DATA_ROOT=${DATA_ROOT}"
echo "[dataset_env] DATASET_NAME=${DATASET_NAME} ROOT_PATH=${ROOT_PATH} DATA_PATH=${DATA_PATH} DATA_CLASS=${DATA_CLASS}"
echo "[dataset_env] EXPERIMENT_TAG=${EXPERIMENT_TAG} SEQ_LEN=${SEQ_LEN} PRED_LEN=${PRED_LEN} FEATURES=${FEATURES} K=${K} TAU=${TAU} DROP_SELF_MODE=${DROP_SELF_MODE} DIST_TRANSFORM=${DIST_TRANSFORM} DISTANCE_METRIC=${DISTANCE_METRIC} SAMPLE_MODE=${SAMPLE_MODE} QUERY_EMBEDDING_SOURCE=${QUERY_EMBEDDING_SOURCE}"
echo "[dataset_env] TARGET=${TARGET}"
echo "[dataset_env] TEACHER_QUERY_STRIDE=${TEACHER_QUERY_STRIDE}"
echo "[dataset_env] TEACHER_MODE=${TEACHER_MODE}"
echo "[dataset_env] TEACHER_ALIGN=${TEACHER_ALIGN} SHIFT_LAST_M=${SHIFT_LAST_M} RERANK_MODE=${RERANK_MODE} RERANK_K0=${RERANK_K0} RERANK_WEIGHT_SOURCE=${RERANK_WEIGHT_SOURCE} RERANK_TAU=${RERANK_TAU:-<default>}"
echo "[dataset_env] TEACHER_TAG=${TEACHER_TAG}"
echo "[dataset_env] MEMORY_SOURCE_TAG=${MEMORY_SOURCE_TAG}"
echo "[dataset_env] MEMORY_CKPT=${MEMORY_CKPT}"
echo "[dataset_env] BETA_SCHEDULE=${BETA_SCHEDULE} CONF_TYPE=${CONF_TYPE} DISTILL_TARGET=${DISTILL_TARGET} DISTILL_ABS_WEIGHT=${DISTILL_ABS_WEIGHT} DISTILL_DELTA_WEIGHT=${DISTILL_DELTA_WEIGHT} DISTILL_MED_GATE=${DISTILL_MED_GATE} BASE_ANCHOR_LAMBDA=${BASE_ANCHOR_LAMBDA}"
