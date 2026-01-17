#!/usr/bin/env bash
#SBATCH --job-name=tsmemory-teacher
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
#   sbatch --export=ALL,DATASET_NAME=ETTm1,SPLIT=train script/build_teacher_ts_quantiles.sbatch
source "${REPO_DIR}/script/dataset_env.sh"

# Split to build (train|val|test).
# Supports Slurm arrays:
#   sbatch --array=0-2 --export=ALL,DATASET_NAME=ETTm1 script/build_teacher_ts_quantiles.sbatch
if [ -z "${SPLIT:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  case "${SLURM_ARRAY_TASK_ID}" in
    0) SPLIT="train" ;;
    1) SPLIT="val" ;;
    2) SPLIT="test" ;;
    *)
      echo "[error] unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0(train),1(val),2(test)" >&2
      exit 2
      ;;
  esac
fi
SPLIT="${SPLIT:-train}"

# Resource / output knobs
SHARD_SIZE="${SHARD_SIZE:-5000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
QUERY_STRIDE="${QUERY_STRIDE:-${TEACHER_QUERY_STRIDE:-1}}"
TEACHER_MODE="${TEACHER_MODE:-weighted_quantile}"  # weighted_quantile|rag_output
TEACHER_ALIGN="${TEACHER_ALIGN:-none}"              # none|retrieved_to_query|shift_last|shift_mean_last_m
SHIFT_LAST_M="${SHIFT_LAST_M:-16}"
RERANK_MODE="${RERANK_MODE:-none}"                  # none|raw_l1_shift|raw_l2_shift
RERANK_K0="${RERANK_K0:-0}"                         # 0 disables rerank
RERANK_WEIGHT_SOURCE="${RERANK_WEIGHT_SOURCE:-embedding}"  # embedding|raw
RERANK_TAU="${RERANK_TAU:-}"                        # optional; empty uses --tau
RETRIEVED_LOC_SCALE="${RETRIEVED_LOC_SCALE:-full}"  # full|context (only for teacher_align=retrieved_to_query)
RAG_MODEL_CKPT="${RAG_MODEL_CKPT:-${REPO_DIR}/checkpoints/chronos-bolt/best.pth}"
RAG_AUGMENT_MODE="${RAG_AUGMENT_MODE:-moe2}"
DROP_SELF_MODE="${DROP_SELF_MODE:-dist}"        # dist|idx
DIST_TRANSFORM="${DIST_TRANSFORM:-none}"        # none|sqrt
DISTANCE_METRIC="${DISTANCE_METRIC:-l2}"        # l2|cosine
SAMPLE_MODE="${SAMPLE_MODE:-univariate}"        # univariate|multivariate
RETRIEVAL_FEATURE_ID="${RETRIEVAL_FEATURE_ID:-}"  # only for sample_mode=multivariate; empty=auto
RETRIEVAL_DB_PATH="${RETRIEVAL_DB_PATH:-}"        # optional explicit retrieval DB path (pkl or dir)
# Retrieval-embedding augmentation (offline teacher only; default off)
RETRIEVAL_EMBED_TIME_FEATURES="${RETRIEVAL_EMBED_TIME_FEATURES:-0}"  # 0/1
RETRIEVAL_EMBED_TIME_WEIGHT="${RETRIEVAL_EMBED_TIME_WEIGHT:-2.0}"
RETRIEVAL_EMBED_STATS_FEATURES="${RETRIEVAL_EMBED_STATS_FEATURES:-0}"  # 0/1
RETRIEVAL_EMBED_STATS_WEIGHT="${RETRIEVAL_EMBED_STATS_WEIGHT:-2.0}"
RETRIEVAL_EMBED_STATS_LAST_M="${RETRIEVAL_EMBED_STATS_LAST_M:-}"  # empty => use SHIFT_LAST_M

# Basic file checks (fully offline)
if [ ! -f "${ROOT_PATH}/${DATA_PATH}" ]; then
  echo "[error] dataset CSV not found: ${ROOT_PATH}/${DATA_PATH}" >&2
  exit 2
fi
if [ ! -f "${BASE_MODEL_PATH}/config.json" ]; then
  echo "[error] base model config not found: ${BASE_MODEL_PATH}/config.json" >&2
  exit 2
fi
if [ "${TEACHER_MODE}" = "rag_output" ] && [ ! -f "${RAG_MODEL_CKPT}" ]; then
  echo "[error] rag teacher ckpt not found: ${RAG_MODEL_CKPT}" >&2
  exit 2
fi
shopt -s nullglob
db_candidates=("${RETRIEVAL_DATABASE_DIR}/${DATASET_NAME}_"*"_${SEQ_LEN}.pkl")
db_dir_candidates=("${RETRIEVAL_DATABASE_DIR}/${DATASET_NAME}_"*"_${SEQ_LEN}")
shopt -u nullglob
if [ -n "${RETRIEVAL_DB_PATH}" ]; then
  if [ -f "${RETRIEVAL_DB_PATH}" ]; then
    echo "[info] using explicit retrieval_database pkl: ${RETRIEVAL_DB_PATH}"
  elif [ -d "${RETRIEVAL_DB_PATH}" ] && [ -d "${RETRIEVAL_DB_PATH}/embeddings" ]; then
    echo "[info] using explicit retrieval_database dir: ${RETRIEVAL_DB_PATH}"
  else
    echo "[error] RETRIEVAL_DB_PATH must be a .pkl file or a dir with embeddings/: ${RETRIEVAL_DB_PATH}" >&2
    exit 2
  fi
else
  if [ ${#db_candidates[@]} -eq 0 ]; then
    found_dir=""
    for d in "${db_dir_candidates[@]}"; do
      if [ -d "${d}" ] && [ -d "${d}/embeddings" ]; then
        found_dir="${d}"
        break
      fi
    done
    if [ -z "${found_dir}" ]; then
      echo "[error] retrieval_database cache not found for ${DATASET_NAME} seq_len=${SEQ_LEN} in ${RETRIEVAL_DATABASE_DIR}" >&2
      echo "        Expected either: ${DATASET_NAME}_*_${SEQ_LEN}.pkl OR ${DATASET_NAME}_*_${SEQ_LEN}/embeddings/feat_*.npy" >&2
      exit 2
    fi
    echo "[info] using retrieval_database dir cache: ${found_dir}"
  else
    echo "[info] using retrieval_database pkl cache: ${db_candidates[0]}"
  fi
fi

# Safety: enforce leakage prevention
if [ "${KB_SPLIT}" != "train" ]; then
  echo "[error] KB_SPLIT must be train (leakage prevention). Got: ${KB_SPLIT}" >&2
  exit 2
fi
# ------------------------------------------------------------------------

EXTRA_ARGS=(
  --drop_self_mode "${DROP_SELF_MODE}"
  --dist_transform "${DIST_TRANSFORM}"
  --distance_metric "${DISTANCE_METRIC}"
  --sample_mode "${SAMPLE_MODE}"
  --shift_last_m "${SHIFT_LAST_M}"
  --rerank_mode "${RERANK_MODE}"
  --rerank_k0 "${RERANK_K0}"
  --rerank_weight_source "${RERANK_WEIGHT_SOURCE}"
)
if [ -n "${RETRIEVAL_FEATURE_ID}" ]; then
  EXTRA_ARGS+=(--retrieval_feature_id "${RETRIEVAL_FEATURE_ID}")
fi
if [ -n "${RETRIEVAL_DB_PATH}" ]; then
  EXTRA_ARGS+=(--retrieval_db_path "${RETRIEVAL_DB_PATH}")
fi
if [ "${RETRIEVAL_EMBED_TIME_FEATURES}" = "1" ]; then
  EXTRA_ARGS+=(--retrieval_embed_time_features --retrieval_embed_time_weight "${RETRIEVAL_EMBED_TIME_WEIGHT}")
fi
if [ "${RETRIEVAL_EMBED_STATS_FEATURES}" = "1" ]; then
  EXTRA_ARGS+=(--retrieval_embed_stats_features --retrieval_embed_stats_weight "${RETRIEVAL_EMBED_STATS_WEIGHT}")
  if [ -n "${RETRIEVAL_EMBED_STATS_LAST_M}" ]; then
    EXTRA_ARGS+=(--retrieval_embed_stats_last_m "${RETRIEVAL_EMBED_STATS_LAST_M}")
  fi
fi
if [ -n "${RERANK_TAU}" ]; then
  EXTRA_ARGS+=(--rerank_tau "${RERANK_TAU}")
fi

python -u scripts/build_teacher_ts_quantiles.py \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --features "${FEATURES}" \
  --target "${TARGET}" \
  --split "${SPLIT}" \
  --kb_split "${KB_SPLIT}" \
  --seq_len "${SEQ_LEN}" \
  --pred_len "${PRED_LEN}" \
  --k "${K}" \
  --tau "${TAU}" \
  --drop_self_eps "${DROP_SELF_EPS}" \
  --teacher_mode "${TEACHER_MODE}" \
  --teacher_align "${TEACHER_ALIGN}" \
  --retrieved_loc_scale "${RETRIEVED_LOC_SCALE}" \
  --query_embedding_source "${QUERY_EMBEDDING_SOURCE}" \
  --base_model_path "${BASE_MODEL_PATH}" \
  --rag_model_ckpt "${RAG_MODEL_CKPT}" \
  --rag_augment_mode "${RAG_AUGMENT_MODE}" \
  --retrieval_database_dir "${RETRIEVAL_DATABASE_DIR}" \
  --save_dir "${TEACHER_SAVE_DIR}" \
  --query_stride "${QUERY_STRIDE}" \
  --shard_size "${SHARD_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"
