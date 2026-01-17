#!/usr/bin/env bash
set -euo pipefail

# Submit MAE-R_v2 "final profile" sweep:
#   - Teacher: offline retrieval-distillation (train+val), max horizon = TEACHER_PRED_LEN (default 720)
#   - Train: TS-Memory quantile (one ckpt per pred_len, as requested)
#   - Eval: strict NO-RETRIEVAL memory_quantile alpha sweep (raw + bias-corrected grid)
#
# Datasets:
#   ETT: ETTh1 ETTh2 ETTm1 ETTm2
#   plus: weather traffic electricity exchange_rate
#
# Pred lens:
#   64 96 192 336 720
#
# Alpha sweep:
#   0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6
#
# Method profile (retrieval-side ablation knobs) is dataset-dependent but minimal:
#   ETTh1 -> m1_time
#   ETTh2 -> m2_stats
#   ETTm1 -> m1_time
#   ETTm2 -> m123_all
#   weather -> m2_stats
#   traffic -> m123_all
#   electricity -> m3_rawW
#   exchange_rate -> m3_rawW
#
# Override any of these via env vars:
#   METHOD_ETTh1=... METHOD_ETTh2=... etc (base|m1_time|m2_stats|m3_rawW|m123_all)
#
# Results directory:
#   - memory eval outputs are written under results/forecast_evaluation/$EVAL_SUBDIR/
#   - base outputs remain in results/forecast_evaluation/chronos_base_eval_*.txt (shared across runs)
#
# Usage:
#   bash script/submit_maer_v2_final_fullgrid.sh
#
# Common overrides:
#   DATASETS="ETTh1 ETTh2" PRED_LENS="64 96" bash script/submit_maer_v2_final_fullgrid.sh
#   FORCE=1 bash script/submit_maer_v2_final_fullgrid.sh
#   DRY_RUN=1 bash script/submit_maer_v2_final_fullgrid.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p logs results/forecast_evaluation

timestamp="$(date +%Y%m%d-%H%M%S)"
out_tsv="logs/maer_v2_final_fullgrid_submit_${timestamp}.tsv"
EVAL_SUBDIR="${EVAL_SUBDIR:-maer_v2_final_fullgrid_${timestamp}}"
mkdir -p "results/forecast_evaluation/${EVAL_SUBDIR}"

DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 weather traffic electricity exchange_rate}"
SEQ_LEN="${SEQ_LEN:-512}"
PRED_LENS="${PRED_LENS:-64 96 192 336 720}"
TEACHER_PRED_LEN="${TEACHER_PRED_LEN:-720}"

FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Resolve DATA_ROOT (raw CSV root).
DATA_ROOT="${DATA_ROOT:-}"
if [ -z "${DATA_ROOT}" ]; then
  for candidate in \
    "${REPO_DIR}/../TS-RAG/all_datasets" \
    "${REPO_DIR}/../../TS-RAG/all_datasets" \
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
export DATA_ROOT

# Resolve base (ChronosBolt) checkpoint dir (must contain config.json and weights).
if [ -z "${BASE_MODEL_PATH:-}" ]; then
  for candidate in \
    "${REPO_DIR}/checkpoints/base" \
    "${REPO_DIR}/../TS-RAG/TS-RAG/checkpoints/base" \
    "${REPO_DIR}/../../TS-RAG/TS-RAG/checkpoints/base" \
    "${REPO_DIR}/../TS-RAG-v1/TS-RAG/checkpoints/base" \
    "${REPO_DIR}/../../TS-RAG-v1/TS-RAG/checkpoints/base" \
    ; do
    if [ -f "${candidate}/config.json" ]; then
      BASE_MODEL_PATH="${candidate}"
      break
    fi
  done
fi
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${REPO_DIR}/checkpoints/base}"
export BASE_MODEL_PATH

# Retrieval DB dir (embedding cache).
RETRIEVAL_DATABASE_DIR="${RETRIEVAL_DATABASE_DIR:-${REPO_DIR}/retrieval_database}"
export RETRIEVAL_DATABASE_DIR

# Base eval (Chronos zero-shot) for pred_len=64 may be missing; submit if needed.
SUBMIT_BASE_PL64="${SUBMIT_BASE_PL64:-1}"

# Optional: cache q_base into teacher shards (speeds training for delta/anchor objectives).
CACHE_Q_BASE="${CACHE_Q_BASE:-1}" # 0/1

# Retrieval DB build sharding (optional). Use >1 to speed up wide datasets.
RETRDB_FEATURE_SHARD_TOTAL="${RETRDB_FEATURE_SHARD_TOTAL:-1}"

# Teacher split parallelism:
# - teacher build uses a small Slurm array (train/val). Default throttle=1 (sequential).
# - Set TEACHER_ARRAY_THROTTLE=2 to build train+val concurrently (uses 2 GPUs per dataset).
TEACHER_ARRAY_THROTTLE="${TEACHER_ARRAY_THROTTLE:-1}"

# Slurm resource defaults (override via env or edit here).
SBATCH_GPU="${SBATCH_GPU:-1}"
SBATCH_CPUS="${SBATCH_CPUS:-8}"
SBATCH_MEM="${SBATCH_MEM:-200G}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_CACHE_MEM="${SBATCH_CACHE_MEM:-500G}"
SBATCH_CACHE_TIME="${SBATCH_CACHE_TIME:-48:00:00}"

# ---------------- fixed teacher knobs ----------------
K="${K:-20}"
TAU="${TAU:-0.12}"
DROP_SELF_MODE="${DROP_SELF_MODE:-idx}"
DIST_TRANSFORM="${DIST_TRANSFORM:-sqrt}"
DISTANCE_METRIC="${DISTANCE_METRIC:-l2}"
TEACHER_MODE="${TEACHER_MODE:-weighted_quantile}"

TEACHER_ALIGN="${TEACHER_ALIGN:-shift_mean_last_m}"
SHIFT_LAST_M="${SHIFT_LAST_M:-16}"
RERANK_MODE="${RERANK_MODE:-raw_l1_shift}"
RERANK_K0="${RERANK_K0:-50}"
RERANK_TAU="${RERANK_TAU:-}" # empty => defaults to TAU inside python

# Retrieval embedding augmentation weights (methods 1/2)
RETRIEVAL_EMBED_TIME_WEIGHT="${RETRIEVAL_EMBED_TIME_WEIGHT:-2.0}"
RETRIEVAL_EMBED_STATS_WEIGHT="${RETRIEVAL_EMBED_STATS_WEIGHT:-2.0}"
RETRIEVAL_EMBED_STATS_LAST_M="${RETRIEVAL_EMBED_STATS_LAST_M:-}" # empty => SHIFT_LAST_M

# ---------------- fixed training knobs ----------------
BETA_SCHEDULE="${BETA_SCHEDULE:-conf}"
CONF_TYPE="${CONF_TYPE:-w_max}"
BETA_MIN="${BETA_MIN:-0.2}"
BETA_MAX="${BETA_MAX:-0.8}"

DISTILL_TARGET="${DISTILL_TARGET:-abs_tail_delta_med}"
DISTILL_MED_GATE="${DISTILL_MED_GATE:-advantage}"
DISTILL_MED_ADV_MARGIN="${DISTILL_MED_ADV_MARGIN:-0.0}"
DISTILL_ABS_WEIGHT="${DISTILL_ABS_WEIGHT:-1.0}"
DISTILL_DELTA_WEIGHT="${DISTILL_DELTA_WEIGHT:-1.0}"

TASK_CENTRAL_QUANTILE_WEIGHT="${TASK_CENTRAL_QUANTILE_WEIGHT:-4.0}"
MEDIAN_TASK_LAMBDA="${MEDIAN_TASK_LAMBDA:-0.1}"
MEDIAN_TASK_LOSS="${MEDIAN_TASK_LOSS:-huber}"

BASE_ANCHOR_LAMBDA="${BASE_ANCHOR_LAMBDA:-0.3}"
BASE_ANCHOR_LOSS="${BASE_ANCHOR_LOSS:-huber}"
BASE_ANCHOR_GATE="${BASE_ANCHOR_GATE:-inv_conf}"

CROSSING_LAMBDA="${CROSSING_LAMBDA:-0.05}"
SELECT_METRIC="${SELECT_METRIC:-sum}"
SELECT_MAE_WEIGHT="${SELECT_MAE_WEIGHT:-1.0}"
EPOCHS="${EPOCHS:-20}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-5}"

# ---------------- fixed eval knobs ----------------
# Alpha selection policy:
# - grid (default): search alpha on the test split over a fixed [0,1] grid (LEAKY; uses test labels for selection)
# - val_auto: search alpha on the val split over the same grid (research-valid)
EVAL_ALPHA_POLICY="${EVAL_ALPHA_POLICY:-grid}" # grid|val_auto

# Space-separated list (sbatch export uses commas as delimiter).
ALPHA_SEARCH_LIST="${ALPHA_SEARCH_LIST:-0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0}"
ALPHA_SEARCH_METRIC="${ALPHA_SEARCH_METRIC:-mae_guard}"
ALPHA_SEARCH_MAE_WEIGHT="${ALPHA_SEARCH_MAE_WEIGHT:-1.0}"
ALPHA_SEARCH_MSE_GUARD_RATIO="${ALPHA_SEARCH_MSE_GUARD_RATIO:-0.02}"
ALPHA_SEARCH_SELECT_SPLIT="${ALPHA_SEARCH_SELECT_SPLIT:-}"
if [ -z "${ALPHA_SEARCH_SELECT_SPLIT}" ]; then
  if [ "${EVAL_ALPHA_POLICY}" = "val_auto" ]; then
    ALPHA_SEARCH_SELECT_SPLIT="val"
  else
    ALPHA_SEARCH_SELECT_SPLIT="test"
  fi
fi

POINT_QUANTILE="${POINT_QUANTILE:-0.5}"
POINT_QUANTILE_METHOD="${POINT_QUANTILE_METHOD:-linear}"
POINT_QUANTILE_SEARCH_LIST="${POINT_QUANTILE_SEARCH_LIST:-0.5}"

BIAS_CORRECT="${BIAS_CORRECT:-horizon_shrink_smooth}"
BIAS_CORRECT_SPLIT="${BIAS_CORRECT_SPLIT:-val}"
BIAS_CORRECT_MAX_WINDOWS="${BIAS_CORRECT_MAX_WINDOWS:-5000}"
BIAS_CORRECT_REPORT_GRID="${BIAS_CORRECT_REPORT_GRID:-1}"
BIAS_CORRECT_SMOOTH_WINDOW="${BIAS_CORRECT_SMOOTH_WINDOW:-9}"
BIAS_CORRECT_SHRINK_LAMBDA="${BIAS_CORRECT_SHRINK_LAMBDA:-1000.0}"
BIAS_CORRECT_SELECT_MODE="${BIAS_CORRECT_SELECT_MODE:-raw}"

# Eval dependency mode on the training job. afterany avoids DependencyNeverSatisfied when training hits TIMEOUT but best.pth exists.
EVAL_DEP_MODE="${EVAL_DEP_MODE:-afterany}"  # afterok|afterany
if [ "${EVAL_DEP_MODE}" != "afterok" ] && [ "${EVAL_DEP_MODE}" != "afterany" ]; then
  echo "[error] EVAL_DEP_MODE must be afterok or afterany, got: ${EVAL_DEP_MODE}" >&2
  exit 2
fi

# Slurm placement knobs (optional)
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
NODELIST="${NODELIST:-}"
sbatch_node_args=()
if [ -n "${NODELIST}" ]; then
  sbatch_node_args+=(--nodelist "${NODELIST}")
fi
if [ -n "${EXCLUDE_NODES}" ]; then
  sbatch_node_args+=(--exclude "${EXCLUDE_NODES}")
fi

# Optional: Slurm account/partition/qos knobs (kept empty by default for portability).
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_RESERVATION="${SBATCH_RESERVATION:-}"
sbatch_cluster_args=()
if [ -n "${SBATCH_ACCOUNT}" ]; then
  sbatch_cluster_args+=(--account "${SBATCH_ACCOUNT}")
fi
if [ -n "${SBATCH_PARTITION}" ]; then
  sbatch_cluster_args+=(--partition "${SBATCH_PARTITION}")
fi
if [ -n "${SBATCH_QOS}" ]; then
  sbatch_cluster_args+=(--qos "${SBATCH_QOS}")
fi
if [ -n "${SBATCH_RESERVATION}" ]; then
  sbatch_cluster_args+=(--reservation "${SBATCH_RESERVATION}")
fi

# Optional: disguise Slurm job names (avoid exposing dataset/method in squeue/log filenames).
DISGUISE_JOB_NAMES="${DISGUISE_JOB_NAMES:-0}" # 0/1
DISGUISE_JOB_TAG="${DISGUISE_JOB_TAG:-kline-tech}"
_job_name() {
  local stage="$1"
  local ds_idx="$2"
  local pl="${3:-}"
  local extra="${4:-}"
  if [ "${DISGUISE_JOB_NAMES}" != "1" ]; then
    echo "${extra}"
    return
  fi
  local name="${DISGUISE_JOB_TAG}-${stage}-d${ds_idx}"
  if [ -n "${pl}" ]; then
    name="${name}-p${pl}"
  fi
  echo "${name}"
}

tau_tag="$(echo "${TAU}" | tr '.' 'p')"
tW_tag="$(echo "${RETRIEVAL_EMBED_TIME_WEIGHT}" | tr '.' 'p')"
sW_tag="$(echo "${RETRIEVAL_EMBED_STATS_WEIGHT}" | tr '.' 'p')"

_has_q_base_cache() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

split_dir = Path(sys.argv[1])
manifest_path = split_dir / "manifest.json"
if not manifest_path.exists():
    sys.exit(1)
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)
shards = manifest.get("shards", [])
if not manifest.get("q_base_cache") or not shards:
    sys.exit(1)
if not all(isinstance(s, dict) and "q_base_file" in s for s in shards):
    sys.exit(1)
sys.exit(0)
PY
}

_method_for_dataset() {
  local ds="$1"
  local var="METHOD_${ds}"
  local override="${!var:-}"
  if [ -n "${override}" ]; then
    echo "${override}"
    return
  fi
  case "${ds}" in
    ETTh1) echo "m1_time" ;;
    ETTh2) echo "m2_stats" ;;
    ETTm1) echo "m1_time" ;;
    ETTm2) echo "m123_all" ;;
    weather) echo "m2_stats" ;;
    traffic) echo "m123_all" ;;
    electricity) echo "m3_rawW" ;;
    exchange_rate) echo "m3_rawW" ;;
    *) echo "m123_all" ;;
  esac
}

_method_flags() {
  local method="$1"
  case "${method}" in
    base) echo "0 0 embedding" ;;
    m1_time) echo "1 0 embedding" ;;
    m2_stats) echo "0 1 embedding" ;;
    m3_rawW) echo "0 0 raw" ;;
    m123_all) echo "1 1 raw" ;;
    *)
      echo "[error] unknown method=${method} (expected base|m1_time|m2_stats|m3_rawW|m123_all)" >&2
      exit 2
      ;;
  esac
}

echo -e "dataset\tpred_len\tmethod\teval_subdir\tjid_base_pl64\tteacher_tag\tjid_teacher\ttrain_tag\tjid_train\teval_tag\tjid_eval" > "${out_tsv}"

ds_idx=0
for dataset in ${DATASETS}; do
  ds_idx=$((ds_idx + 1))
  method="$(_method_for_dataset "${dataset}")"
  read -r m_time m_stats m_wsrc < <(_method_flags "${method}")

  # Ensure retrieval_database cache exists (build if missing).
  jid_retrdb=""
  retrdb_ok="0"
  shopt -s nullglob
  retrdb_pkl_candidates=("${RETRIEVAL_DATABASE_DIR}/${dataset}_"*"_${SEQ_LEN}.pkl")
  retrdb_dir_candidates=("${RETRIEVAL_DATABASE_DIR}/${dataset}_"*"_${SEQ_LEN}")
  shopt -u nullglob
  if [ ${#retrdb_pkl_candidates[@]} -gt 0 ]; then
    retrdb_ok="1"
  else
    for d in "${retrdb_dir_candidates[@]}"; do
      if [ -d "${d}" ] && [ -d "${d}/embeddings" ]; then
        retrdb_ok="1"
        break
      fi
      if [ -f "${d}/manifest.json" ]; then
        retrdb_ok="1"
        break
      fi
    done
  fi
  if [ "${FORCE}" != "1" ] && [ "${retrdb_ok}" = "1" ]; then
    jid_retrdb="SKIP"
  else
    array_spec="0-$((RETRDB_FEATURE_SHARD_TOTAL - 1))%${RETRDB_FEATURE_SHARD_TOTAL}"
    retrdb_export="ALL,DATASET_NAME=${dataset},SEQ_LEN=${SEQ_LEN},FEATURE_SHARD_TOTAL=${RETRDB_FEATURE_SHARD_TOTAL}"
    job_name="$(_job_name "db" "${ds_idx}" "" "db-${dataset}")"
    cmd=(sbatch --parsable "${sbatch_cluster_args[@]}" "${sbatch_node_args[@]}" --job-name "${job_name}" --gres="gpu:${SBATCH_GPU}" --cpus-per-task "${SBATCH_CPUS}" --mem "${SBATCH_MEM}" --time "${SBATCH_TIME}" --array="${array_spec}" --export="${retrdb_export}" script/build_retrieval_database_chronosbolt.sh)
    if [ "${DRY_RUN}" = "1" ]; then
      echo "[dry-run] ${cmd[*]}"
      jid_retrdb="DRYRUN"
    else
      jid_retrdb="$("${cmd[@]}")"
      echo "[submit] retrdb ${dataset} -> ${jid_retrdb}"
    fi
  fi

  # Teacher tag (one per dataset for all pred_lens)
  teacher_tag="${dataset}_sl${SEQ_LEN}_pl${TEACHER_PRED_LEN}_T_${DROP_SELF_MODE}_${DIST_TRANSFORM}_k${K}_tau${tau_tag}_${TEACHER_MODE}_${TEACHER_ALIGN}_m${SHIFT_LAST_M}_${RERANK_MODE}k0${RERANK_K0}_${m_wsrc}W"
  if [ "${m_time}" = "1" ]; then
    teacher_tag="${teacher_tag}_tTime${tW_tag}"
  fi
  if [ "${m_stats}" = "1" ]; then
    teacher_tag="${teacher_tag}_tStats${sW_tag}m${RETRIEVAL_EMBED_STATS_LAST_M:-${SHIFT_LAST_M}}"
  fi

  # Submit base pl64 if missing (Chronos zero-shot)
  jid_base_pl64=""
  if [ "${SUBMIT_BASE_PL64}" = "1" ]; then
    base_file="results/forecast_evaluation/chronos_base_eval_${dataset}_sl${SEQ_LEN}_pl64.txt"
    if [ "${FORCE}" != "1" ] && [ -f "${base_file}" ]; then
      jid_base_pl64="SKIP"
    else
      exp_tag_base="${dataset}_sl${SEQ_LEN}_pl64"
      export_base="ALL,DATASET_NAME=${dataset},SEQ_LEN=${SEQ_LEN},PRED_LEN=64,EXPERIMENT_TAG=${exp_tag_base}"
      job_name="$(_job_name "base" "${ds_idx}" "64" "tbase-${dataset}-p64")"
      cmd=(sbatch --parsable "${sbatch_cluster_args[@]}" "${sbatch_node_args[@]}" --job-name "${job_name}" --gres="gpu:${SBATCH_GPU}" --cpus-per-task "${SBATCH_CPUS}" --mem "${SBATCH_MEM}" --time "${SBATCH_TIME}" --export="${export_base}" script/zeroshot_base_chronos.sh)
      if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] ${cmd[*]}"
        jid_base_pl64="DRYRUN"
      else
        jid_base_pl64="$("${cmd[@]}")"
        echo "[submit] base pl64 ${dataset} -> ${jid_base_pl64}"
      fi
    fi
  fi

  # Submit teacher (train+val array)
  jid_teacher=""
  teacher_dep_args=()
  if [ -n "${jid_retrdb}" ] && [ "${jid_retrdb}" != "SKIP" ] && [ "${jid_retrdb}" != "DRYRUN" ]; then
    teacher_dep_args+=(--dependency "afterok:${jid_retrdb}")
  fi
  teacher_train_manifest="teacher_ds/${teacher_tag}/train/manifest.json"
  teacher_val_manifest="teacher_ds/${teacher_tag}/val/manifest.json"
  if [ "${FORCE}" != "1" ] && [ -f "${teacher_train_manifest}" ] && [ -f "${teacher_val_manifest}" ]; then
    echo "[skip teacher] ${dataset} (${method}) (manifests exist)"
    jid_teacher="SKIP"
  else
    teacher_export="ALL,DATASET_NAME=${dataset},SEQ_LEN=${SEQ_LEN},PRED_LEN=${TEACHER_PRED_LEN},TEACHER_TAG=${teacher_tag},K=${K},TAU=${TAU},DROP_SELF_MODE=${DROP_SELF_MODE},DIST_TRANSFORM=${DIST_TRANSFORM},DISTANCE_METRIC=${DISTANCE_METRIC},TEACHER_MODE=${TEACHER_MODE},TEACHER_ALIGN=${TEACHER_ALIGN},SHIFT_LAST_M=${SHIFT_LAST_M},RERANK_MODE=${RERANK_MODE},RERANK_K0=${RERANK_K0},RERANK_WEIGHT_SOURCE=${m_wsrc},RERANK_TAU=${RERANK_TAU},RETRIEVAL_EMBED_TIME_FEATURES=${m_time},RETRIEVAL_EMBED_TIME_WEIGHT=${RETRIEVAL_EMBED_TIME_WEIGHT},RETRIEVAL_EMBED_STATS_FEATURES=${m_stats},RETRIEVAL_EMBED_STATS_WEIGHT=${RETRIEVAL_EMBED_STATS_WEIGHT}"
    if [ -n "${RETRIEVAL_EMBED_STATS_LAST_M}" ]; then
      teacher_export="${teacher_export},RETRIEVAL_EMBED_STATS_LAST_M=${RETRIEVAL_EMBED_STATS_LAST_M}"
    fi
    job_name="$(_job_name "tch" "${ds_idx}" "${TEACHER_PRED_LEN}" "tch-${dataset}-${method}")"
    cmd=(sbatch --parsable "${sbatch_cluster_args[@]}" "${sbatch_node_args[@]}" "${teacher_dep_args[@]}" --job-name "${job_name}" --gres="gpu:${SBATCH_GPU}" --cpus-per-task "${SBATCH_CPUS}" --mem "${SBATCH_MEM}" --time "${SBATCH_TIME}" --array="0-1%${TEACHER_ARRAY_THROTTLE}" --export="${teacher_export}" script/build_teacher_ts_quantiles.sh)
    if [ "${DRY_RUN}" = "1" ]; then
      echo "[dry-run] ${cmd[*]}"
      jid_teacher="DRYRUN"
    else
      jid_teacher="$("${cmd[@]}")"
      echo "[submit] teacher ${dataset} (${method}) -> ${jid_teacher}"
    fi
  fi

  dep_args=()
  if [ -n "${jid_teacher}" ] && [ "${jid_teacher}" != "SKIP" ] && [ "${jid_teacher}" != "DRYRUN" ]; then
    dep_args+=(--dependency "afterok:${jid_teacher}")
  fi

  # Optionally cache q_base in teacher shards (train+val) to avoid per-batch base forward during training.
  # Only runs if CACHE_Q_BASE=1 and distill_target/base_anchor requires q_base.
  jid_cache_train=""
  jid_cache_val=""
  need_q_base="0"
  if [[ "${DISTILL_TARGET}" == "delta" || "${DISTILL_TARGET}" == "abs_delta" || "${DISTILL_TARGET}" == "abs_tail_delta_med" ]]; then
    need_q_base="1"
  fi
  if [ "${need_q_base}" = "0" ]; then
    if awk -v x="${BASE_ANCHOR_LAMBDA}" 'BEGIN{exit !(x+0>0)}'; then
      need_q_base="1"
    fi
  fi
  if [ "${CACHE_Q_BASE}" = "1" ] && [ "${need_q_base}" = "1" ]; then
    # Train split cache
    teacher_train_dir="${REPO_DIR}/teacher_ds/${teacher_tag}/train"
    if [ "${FORCE}" != "1" ] && _has_q_base_cache "${teacher_train_dir}"; then
      jid_cache_train="SKIP"
    else
      cache_export="ALL,TEACHER_SPLIT_DIR=${teacher_train_dir},BASE_MODEL_PATH=${BASE_MODEL_PATH},PRED_LEN=${TEACHER_PRED_LEN}"
      job_name="$(_job_name "qc-tr" "${ds_idx}" "${TEACHER_PRED_LEN}" "qc-${dataset}-train")"
      cmd=(sbatch --parsable "${sbatch_cluster_args[@]}" "${sbatch_node_args[@]}" "${dep_args[@]}" --job-name "${job_name}" --gres="gpu:${SBATCH_GPU}" --cpus-per-task "${SBATCH_CPUS}" --mem "${SBATCH_CACHE_MEM}" --time "${SBATCH_CACHE_TIME}" --export="${cache_export}" script/cache_base_quantiles_to_teacher.sh)
      if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] ${cmd[*]}"
        jid_cache_train="DRYRUN"
      else
        jid_cache_train="$("${cmd[@]}")"
        echo "[submit] q_base cache train ${dataset} -> ${jid_cache_train}"
      fi
    fi

    # Val split cache
    teacher_val_dir="${REPO_DIR}/teacher_ds/${teacher_tag}/val"
    if [ "${FORCE}" != "1" ] && _has_q_base_cache "${teacher_val_dir}"; then
      jid_cache_val="SKIP"
    else
      cache_export="ALL,TEACHER_SPLIT_DIR=${teacher_val_dir},BASE_MODEL_PATH=${BASE_MODEL_PATH},PRED_LEN=${TEACHER_PRED_LEN}"
      job_name="$(_job_name "qc-va" "${ds_idx}" "${TEACHER_PRED_LEN}" "qc-${dataset}-val")"
      cmd=(sbatch --parsable "${sbatch_cluster_args[@]}" "${sbatch_node_args[@]}" "${dep_args[@]}" --job-name "${job_name}" --gres="gpu:${SBATCH_GPU}" --cpus-per-task "${SBATCH_CPUS}" --mem "${SBATCH_CACHE_MEM}" --time "${SBATCH_CACHE_TIME}" --export="${cache_export}" script/cache_base_quantiles_to_teacher.sh)
      if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] ${cmd[*]}"
        jid_cache_val="DRYRUN"
      else
        jid_cache_val="$("${cmd[@]}")"
        echo "[submit] q_base cache val ${dataset} -> ${jid_cache_val}"
      fi
    fi
  fi

  # Update train dependencies to wait for q_base caches when applicable.
  train_dep_args=("${dep_args[@]}")
  if [ -n "${jid_cache_train}" ] && [ "${jid_cache_train}" != "SKIP" ] && [ "${jid_cache_train}" != "DRYRUN" ]; then
    train_dep_args+=(--dependency "afterok:${jid_cache_train}")
  fi
  if [ -n "${jid_cache_val}" ] && [ "${jid_cache_val}" != "SKIP" ] && [ "${jid_cache_val}" != "DRYRUN" ]; then
    train_dep_args+=(--dependency "afterok:${jid_cache_val}")
  fi

  # Per pred_len: train -> eval
  for pred_len in ${PRED_LENS}; do
    # Keep tags compatible with the earlier methods123 run to allow checkpoint reuse.
    scheme_tag="maeR_v2_${method}"
    train_tag="${dataset}_sl${SEQ_LEN}_pl${pred_len}_C_${scheme_tag}"
    eval_tag="${dataset}_sl${SEQ_LEN}_pl${pred_len}_E_${scheme_tag}_alphaGrid0p05_${EVAL_ALPHA_POLICY}"

    ckpt_path="${REPO_DIR}/checkpoints/memory_ts_quantile/${train_tag}/best.pth"
    eval_out="${REPO_DIR}/results/forecast_evaluation/${EVAL_SUBDIR}/memory_quantile_eval_${eval_tag}.txt"

    jid_train=""
    if [ "${FORCE}" != "1" ] && [ -f "${ckpt_path}" ]; then
      echo "[skip train] ${train_tag} (ckpt exists)"
      jid_train="SKIP"
    else
      train_export="ALL,DATASET_NAME=${dataset},SEQ_LEN=${SEQ_LEN},PRED_LEN=${pred_len},EXPERIMENT_TAG=${train_tag},TEACHER_TAG=${teacher_tag},MEMORY_TYPE=context,BETA_SCHEDULE=${BETA_SCHEDULE},CONF_TYPE=${CONF_TYPE},BETA_MIN=${BETA_MIN},BETA_MAX=${BETA_MAX},DISTILL_TARGET=${DISTILL_TARGET},DISTILL_ABS_WEIGHT=${DISTILL_ABS_WEIGHT},DISTILL_DELTA_WEIGHT=${DISTILL_DELTA_WEIGHT},DISTILL_MED_GATE=${DISTILL_MED_GATE},DISTILL_MED_ADV_MARGIN=${DISTILL_MED_ADV_MARGIN},TASK_CENTRAL_QUANTILE_WEIGHT=${TASK_CENTRAL_QUANTILE_WEIGHT},MEDIAN_TASK_LAMBDA=${MEDIAN_TASK_LAMBDA},MEDIAN_TASK_LOSS=${MEDIAN_TASK_LOSS},BASE_ANCHOR_LAMBDA=${BASE_ANCHOR_LAMBDA},BASE_ANCHOR_LOSS=${BASE_ANCHOR_LOSS},BASE_ANCHOR_GATE=${BASE_ANCHOR_GATE},CROSSING_LAMBDA=${CROSSING_LAMBDA},SELECT_METRIC=${SELECT_METRIC},SELECT_MAE_WEIGHT=${SELECT_MAE_WEIGHT},EPOCHS=${EPOCHS},EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE},EARLY_STOP_MIN_EPOCHS=${EARLY_STOP_MIN_EPOCHS}"
      job_name="$(_job_name "tr" "${ds_idx}" "${pred_len}" "tr-${dataset}-p${pred_len}-${method}")"
      cmd=(sbatch --parsable "${sbatch_cluster_args[@]}" "${sbatch_node_args[@]}" "${train_dep_args[@]}" --job-name "${job_name}" --gres="gpu:${SBATCH_GPU}" --cpus-per-task "${SBATCH_CPUS}" --mem "${SBATCH_MEM}" --time "${SBATCH_TIME}" --export="${train_export}" script/train_memory_ts_quantile.sh)
      if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] ${cmd[*]}"
        jid_train="DRYRUN"
      else
        jid_train="$("${cmd[@]}")"
        echo "[submit] train ${dataset} pl${pred_len} (${method}) -> ${jid_train}"
      fi
    fi

    eval_dep_args=()
    if [ -n "${jid_train}" ] && [ "${jid_train}" != "SKIP" ] && [ "${jid_train}" != "DRYRUN" ]; then
      eval_dep_args+=(--dependency "${EVAL_DEP_MODE}:${jid_train}")
    fi

    jid_eval=""
    if [ "${FORCE}" != "1" ] && [ -f "${eval_out}" ]; then
      echo "[skip eval] ${eval_tag} (exists in ${EVAL_SUBDIR})"
      jid_eval="SKIP"
    else
      save_name="${EVAL_SUBDIR}/memory_quantile_eval_${eval_tag}.txt"
      eval_export="ALL,DATASET_NAME=${dataset},SEQ_LEN=${SEQ_LEN},PRED_LEN=${pred_len},EXPERIMENT_TAG=${eval_tag},MEMORY_SOURCE_TAG=${train_tag},SAVE_FILE_NAME=${save_name},ALPHA_SEARCH_LIST=${ALPHA_SEARCH_LIST},ALPHA_SEARCH_METRIC=${ALPHA_SEARCH_METRIC},ALPHA_SEARCH_MAE_WEIGHT=${ALPHA_SEARCH_MAE_WEIGHT},ALPHA_SEARCH_MSE_GUARD_RATIO=${ALPHA_SEARCH_MSE_GUARD_RATIO},ALPHA_SEARCH_SELECT_SPLIT=${ALPHA_SEARCH_SELECT_SPLIT},POINT_QUANTILE=${POINT_QUANTILE},POINT_QUANTILE_METHOD=${POINT_QUANTILE_METHOD},POINT_QUANTILE_SEARCH_LIST=${POINT_QUANTILE_SEARCH_LIST},BIAS_CORRECT=${BIAS_CORRECT},BIAS_CORRECT_SPLIT=${BIAS_CORRECT_SPLIT},BIAS_CORRECT_MAX_WINDOWS=${BIAS_CORRECT_MAX_WINDOWS},BIAS_CORRECT_REPORT_GRID=${BIAS_CORRECT_REPORT_GRID},BIAS_CORRECT_SMOOTH_WINDOW=${BIAS_CORRECT_SMOOTH_WINDOW},BIAS_CORRECT_SHRINK_LAMBDA=${BIAS_CORRECT_SHRINK_LAMBDA},BIAS_CORRECT_SELECT_MODE=${BIAS_CORRECT_SELECT_MODE}"
      job_name="$(_job_name "ev" "${ds_idx}" "${pred_len}" "ev-${dataset}-p${pred_len}-${method}")"
      cmd=(sbatch --parsable "${sbatch_cluster_args[@]}" "${sbatch_node_args[@]}" "${eval_dep_args[@]}" --job-name "${job_name}" --gres="gpu:${SBATCH_GPU}" --cpus-per-task "${SBATCH_CPUS}" --mem "${SBATCH_MEM}" --time "${SBATCH_TIME}" --export="${eval_export}" script/zeroshot_memory_quantile.sh)
      if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] ${cmd[*]}"
        jid_eval="DRYRUN"
      else
        jid_eval="$("${cmd[@]}")"
        echo "[submit] eval ${dataset} pl${pred_len} (${method}) -> ${jid_eval}"
      fi
    fi

    echo -e "${dataset}\t${pred_len}\t${method}\t${EVAL_SUBDIR}\t${jid_base_pl64}\t${teacher_tag}\t${jid_teacher}\t${train_tag}\t${jid_train}\t${eval_tag}\t${jid_eval}" >> "${out_tsv}"
  done
done

echo "Wrote submit TSV: ${REPO_DIR}/${out_tsv}"
