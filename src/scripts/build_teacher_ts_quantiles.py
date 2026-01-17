#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import faiss
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.tools import get_borders
from utils.weighted_quantile import (
    numpy_weighted_quantile_reference,
    weighted_quantile_torch,
)

def _load_datetime_index(root_path: str, data_path: str) -> "np.ndarray":
    """
    Load the first (timestamp) column as pandas datetime64 and return as a numpy array.
    """
    import pandas as pd

    df = pd.read_csv(os.path.join(root_path, data_path), usecols=[0])
    if df.shape[1] != 1:
        raise ValueError(f"Expected a single timestamp column at index 0, got shape={df.shape}")
    ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"Failed to parse {bad} timestamps from {root_path}/{data_path}")
    return ts.to_numpy()


def _compute_time_phase_features(ts: "np.ndarray") -> np.ndarray:
    """
    Compute periodic time features:
      - sin/cos of time-of-day (continuous hour)
      - sin/cos of day-of-week

    Args:
        ts: numpy datetime64 array of shape (T,)
    Returns:
        feats: float32 array (T, 4)
    """
    import pandas as pd

    s = pd.Series(ts)
    # Continuous hour-of-day (supports minute/second-level datasets).
    tod = s.dt.hour.to_numpy(dtype=np.float32) + s.dt.minute.to_numpy(dtype=np.float32) / 60.0 + s.dt.second.to_numpy(dtype=np.float32) / 3600.0
    dow = s.dt.dayofweek.to_numpy(dtype=np.float32)
    two_pi = float(2.0 * math.pi)
    sin_h = np.sin(two_pi * tod / 24.0).astype(np.float32)
    cos_h = np.cos(two_pi * tod / 24.0).astype(np.float32)
    sin_d = np.sin(two_pi * dow / 7.0).astype(np.float32)
    cos_d = np.cos(two_pi * dow / 7.0).astype(np.float32)
    return np.stack([sin_h, cos_h, sin_d, cos_d], axis=1).astype(np.float32, copy=False)


def _rolling_mean_std_slope(x: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rolling mean/std/slope over the last-m points ending at each time index.

    x: float32 array (T, C) in *dataset scale* (StandardScaler output).

    Returns arrays of shape (T, C), with zeros for indices < m-1.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (T,C), got {x.shape}")
    T, C = x.shape
    m = int(m)
    if m <= 0:
        raise ValueError(f"m must be >0, got {m}")
    if m > T:
        raise ValueError(f"m={m} exceeds T={T}")

    eps = 1e-6
    x = np.asarray(x, dtype=np.float32)

    # Prefix sums for mean/std.
    prefix = np.concatenate([np.zeros((1, C), dtype=np.float32), np.cumsum(x, axis=0, dtype=np.float64).astype(np.float32)], axis=0)
    prefix2 = np.concatenate([np.zeros((1, C), dtype=np.float32), np.cumsum((x * x), axis=0, dtype=np.float64).astype(np.float32)], axis=0)

    # Prefix sums for slope: sum(k * x_k) with absolute index k.
    k = np.arange(T, dtype=np.float32).reshape(T, 1)
    prefix_kx = np.concatenate([np.zeros((1, C), dtype=np.float32), np.cumsum((x * k), axis=0, dtype=np.float64).astype(np.float32)], axis=0)

    mean = np.zeros((T, C), dtype=np.float32)
    std = np.zeros((T, C), dtype=np.float32)
    slope = np.zeros((T, C), dtype=np.float32)

    # Indices t where the window [t-m+1, t] is valid.
    t = np.arange(m - 1, T, dtype=np.int64)
    t1 = t + 1
    t0 = t1 - m

    sum_x = prefix[t1] - prefix[t0]
    sum_x2 = prefix2[t1] - prefix2[t0]

    mean_t = sum_x / float(m)
    var_t = sum_x2 / float(m) - mean_t * mean_t
    var_t = np.maximum(var_t, 0.0)
    std_t = np.sqrt(var_t + eps, dtype=np.float32)

    mean[t] = mean_t.astype(np.float32, copy=False)
    std[t] = std_t.astype(np.float32, copy=False)

    # Slope with x_j = 0..m-1.
    # Sxy_rel = sum_{k=t-m+1..t} (k - (t-m+1)) * x_k = sum(k*x_k) - start * sum(x_k)
    sum_kx = prefix_kx[t1] - prefix_kx[t0]
    start = (t - m + 1).astype(np.float32).reshape(-1, 1)
    sxy = sum_kx - start * sum_x

    Sx = float(m * (m - 1) / 2.0)
    Sxx = float((m - 1) * m * (2 * m - 1) / 6.0)
    denom = float(m * Sxx - Sx * Sx)
    if denom <= 0:
        raise ValueError(f"Invalid slope denom for m={m}: {denom}")
    slope_t = (float(m) * sxy - Sx * sum_x) / denom
    slope[t] = slope_t.astype(np.float32, copy=False)

    return mean, std, slope

def _instance_norm_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    InstanceNorm stats matching TS-RAG ChronosBolt InstanceNorm semantics.

    Returns (loc, scale, is_constant) where reduction is over the last dimension.
    For constant inputs, scale is forced to 1 and normalized values are overridden to 1.
    """
    loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
    scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
    is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True)
    scale = torch.where(is_constant, torch.ones_like(scale), scale)
    return loc, scale, is_constant


def _instance_norm_apply(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, is_constant: torch.Tensor) -> torch.Tensor:
    normalized = (x - loc) / scale
    normalized = torch.where(is_constant, torch.ones_like(normalized), normalized)
    return normalized


def _instance_norm_inverse(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    is_constant = scale == 1
    return torch.where(is_constant, loc, x * scale + loc)


@dataclass(frozen=True)
class TeacherMeta:
    dataset: str
    split: str
    kb_split: str
    root_path: str
    data_path: str
    features: str
    target: str
    context_len: int
    pred_len: int
    k: int
    tau: float
    drop_self_eps: float
    drop_self_mode: str
    query_stride: int
    quantiles: List[float]
    base_model_path: str
    embedding_model_path: str
    teacher_mode: str
    rag_model_ckpt: str
    rag_augment_mode: str
    teacher_align: str
    shift_last_m: int
    rerank_mode: str
    rerank_k0: int
    rerank_weight_source: str
    rerank_tau: float
    retrieved_loc_scale: str
    dist_transform: str
    distance_metric: str


def _compute_level_shift_univariate(
    ctx: torch.Tensor, retrieved_ctx: torch.Tensor, *, mode: str, last_m: int
) -> torch.Tensor:
    if mode not in {"shift_last", "shift_mean_last_m"}:
        raise ValueError(f"Unsupported level-shift mode={mode}")
    m = 1 if mode == "shift_last" else int(last_m)
    m = max(1, min(m, int(ctx.shape[1])))
    q_ref = ctx[:, -m:].mean(dim=1, keepdim=True)  # (B,1)
    r_ref = retrieved_ctx[:, :, -m:].mean(dim=2)  # (B,K)
    return q_ref - r_ref  # (B,K)


def _compute_level_shift_multivariate(
    ctx: torch.Tensor, retrieved_ctx: torch.Tensor, *, mode: str, last_m: int
) -> torch.Tensor:
    if mode not in {"shift_last", "shift_mean_last_m"}:
        raise ValueError(f"Unsupported level-shift mode={mode}")
    m = 1 if mode == "shift_last" else int(last_m)
    m = max(1, min(m, int(ctx.shape[1])))
    q_ref = ctx[:, -m:, :].mean(dim=1)  # (B,C)
    r_ref = retrieved_ctx[:, :, -m:, :].mean(dim=2)  # (B,K,C)
    return q_ref.unsqueeze(1) - r_ref  # (B,K,C)


def _parse_split(split: str) -> int:
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train/val/test, got: {split}")
    return {"train": 0, "val": 1, "test": 2}[split]


def _load_quantiles(base_model_path: str, override: Optional[str]) -> List[float]:
    if override:
        return [float(x) for x in override.split(",") if x.strip() != ""]
    config = AutoConfig.from_pretrained(base_model_path)
    quantiles = config.chronos_config["quantiles"]
    return [float(q) for q in quantiles]


def _load_raw_series(
    root_path: str, data_path: str, features: str, target: str
) -> Tuple[np.ndarray, List[str]]:
    import pandas as pd

    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    if features in {"M", "MS"}:
        cols = list(df_raw.columns[1:])
        data = df_raw[cols].values.astype(np.float32)
        return data, cols
    if features == "S":
        data = df_raw[[target]].values.astype(np.float32)
        return data, [target]
    raise ValueError(f"Unsupported features={features} (expected S/M/MS)")


def _save_shard(
    out_dir: Path,
    split: str,
    shard_idx: int,
    context: torch.Tensor,
    target: torch.Tensor,
    q_teacher: torch.Tensor,
    meta: Dict,
    *,
    conf_w_max: Optional[torch.Tensor] = None,
    conf_entropy: Optional[torch.Tensor] = None,
    conf_effective_k: Optional[torch.Tensor] = None,
) -> str:
    out_dir = out_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"shard_{shard_idx:04d}.pt"
    payload = {
        "context": context.cpu(),
        "target": target.cpu(),
        "q_teacher": q_teacher.cpu(),
        "meta": meta,
    }
    if conf_w_max is not None:
        payload["conf_w_max"] = conf_w_max.cpu()
    if conf_entropy is not None:
        payload["conf_entropy"] = conf_entropy.cpu()
    if conf_effective_k is not None:
        payload["conf_effective_k"] = conf_effective_k.cpu()

    torch.save(payload, out_dir / shard_name)
    return shard_name


def sanity_check_weighted_quantile(device: torch.device) -> None:
    torch.manual_seed(0)
    B, K, T = 2, 6, 5
    quantiles = [0.1, 0.5, 0.9]
    y = torch.randn(B, K, T, device=device)
    w = torch.rand(B, K, device=device)
    q = torch.tensor(quantiles, device=device)

    out = weighted_quantile_torch(y, w, q).detach().cpu().numpy()
    ref = numpy_weighted_quantile_reference(
        y.detach().cpu().numpy(), w.detach().cpu().numpy(), quantiles
    )
    max_diff = np.max(np.abs(out - ref))
    print(f"[sanity] weighted_quantile max_diff={max_diff:.6g}")
    assert max_diff < 1e-5, f"weighted_quantile mismatch: max_diff={max_diff}"


def main():
    parser = argparse.ArgumentParser("Build TS-Memory teacher dataset (weighted quantiles)")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default="S", choices=["S", "M", "MS"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="univariate",
        choices=["univariate", "multivariate"],
        help="univariate: TS-RAG-style (treat each variable as an independent sample). "
        "multivariate: build samples with full (L,C) windows and teacher (Q,T,C) (reduces teacher_ds size).",
    )
    parser.add_argument(
        "--retrieval_feature_id",
        type=int,
        default=None,
        help="Only for sample_mode=multivariate: which feature id's cached embeddings to use for retrieval. "
        "Default: target column index if exists, else 0.",
    )
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--kb_split", type=str, default="train", choices=["train", "val", "test"])

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=None, help="Default: read from base_model_path config")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--drop_self_eps", type=float, default=1e-8)
    parser.add_argument(
        "--drop_self_mode",
        type=str,
        default="dist",
        choices=["dist", "idx"],
        help="How to drop self-retrieval neighbors when building teacher. "
        "dist: backward-compatible (drop dist<=eps, may incorrectly drop 0-dist duplicates). "
        "idx: drop only true self neighbor (idx==query_start) when split=train and query in KB.",
    )
    parser.add_argument(
        "--dist_transform",
        type=str,
        default="none",
        choices=["none", "sqrt"],
        help="Optional transform applied to FAISS distances before softmax weighting (teacher_mode=weighted_quantile).",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="l2",
        choices=["l2", "cosine"],
        help="Distance metric in embedding space. cosine is implemented by L2-normalizing embeddings then using L2.",
    )
    parser.add_argument(
        "--teacher_mode",
        type=str,
        default="weighted_quantile",
        choices=["weighted_quantile", "rag_output"],
        help="weighted_quantile: teacher is weighted quantiles of retrieved horizons. "
        "rag_output: teacher is the final TS-RAG output (ChronosBoltModelForForecastingWithRetrieval) run offline.",
    )
    parser.add_argument(
        "--teacher_align",
        type=str,
        default="none",
        choices=["none", "instance", "retrieved_to_query", "shift_last", "shift_mean_last_m"],
        help="Align retrieved horizons before weighted-quantile teacher computation. "
        "'instance' is an alias of 'retrieved_to_query'.",
    )
    parser.add_argument(
        "--shift_last_m",
        type=int,
        default=16,
        help="When teacher_align=shift_mean_last_m (and/or rerank uses shift), compute shift using mean of last-m context points.",
    )
    parser.add_argument(
        "--rerank_mode",
        type=str,
        default="none",
        choices=["none", "raw_l1_shift", "raw_l2_shift"],
        help="Optional 2-stage rerank: retrieve top-(rerank_k0) by embedding, then rerank by raw-space distance after shift alignment.",
    )
    parser.add_argument(
        "--rerank_k0",
        type=int,
        default=0,
        help="Candidate pool size for rerank (K0). 0 disables rerank. Must be >=k when rerank_mode!=none.",
    )
    parser.add_argument(
        "--rerank_weight_source",
        type=str,
        default="embedding",
        choices=["embedding", "raw"],
        help="How to compute softmax weights when rerank_mode!=none: embedding distances (backward-compatible) or raw rerank distances.",
    )
    parser.add_argument(
        "--rerank_tau",
        type=float,
        default=None,
        help="Softmax temperature for raw-distance weighting when rerank_weight_source=raw (default: use --tau).",
    )
    parser.add_argument(
        "--retrieved_loc_scale",
        type=str,
        default="full",
        choices=["full", "context"],
        help="When teacher_align=retrieved_to_query, compute retrieved loc/scale on full (ctx+pred) or context only.",
    )

    parser.add_argument(
        "--query_embedding_source",
        type=str,
        default="cache",
        choices=["cache", "model"],
        help="cache: reuse embeddings from retrieval_database pkl (recommended/offline). "
        "model: compute embeddings via ChronosPipeline.embed (requires embedding model weights).",
    )
    parser.add_argument("--base_model_path", type=str, default="./checkpoints/base")
    parser.add_argument("--override_quantiles", type=str, default=None)
    parser.add_argument("--embedding_model_path", type=str, default="amazon/chronos-t5-base")
    parser.add_argument(
        "--rag_model_ckpt",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "chronos-bolt" / "best.pth"),
        help="Path to TS-RAG retrieval-augmented ChronosBolt checkpoint (used when teacher_mode=rag_output).",
    )
    parser.add_argument(
        "--rag_augment_mode",
        type=str,
        default="moe2",
        help="Augment mode used by the TS-RAG checkpoint (used when teacher_mode=rag_output).",
    )
    parser.add_argument("--retrieval_database_dir", type=str, default="../retrieval_database/")
    parser.add_argument(
        "--retrieval_db_path",
        type=str,
        default=None,
        help="Optional explicit retrieval DB cache path (pkl file or dir). "
        "When set, bypasses auto-discovery under --retrieval_database_dir. "
        "Useful to force a specific cache variant when multiple exist (e.g., ETTh1/ETTh2 legacy pkl vs dir cache).",
    )
    parser.add_argument(
        "--retrieval_embed_time_features",
        action="store_true",
        help="Augment retrieval embeddings with time-phase features (sin/cos TOD + sin/cos DOW) computed from the CSV timestamp column. "
        "Applied to both KB and query embeddings consistently; affects neighbor selection.",
    )
    parser.add_argument(
        "--retrieval_embed_time_weight",
        type=float,
        default=2.0,
        help="Scale applied to time features before concatenation (only when --retrieval_embed_time_features is set).",
    )
    parser.add_argument(
        "--retrieval_embed_stats_features",
        action="store_true",
        help="Augment retrieval embeddings with per-window level/scale/trend stats from the last-m context points "
        "(mean, log(std), slope). Stats are z-scored using KB(train) windows, then concatenated; affects neighbor selection.",
    )
    parser.add_argument(
        "--retrieval_embed_stats_weight",
        type=float,
        default=2.0,
        help="Scale applied to stats features after z-score (only when --retrieval_embed_stats_features is set).",
    )
    parser.add_argument(
        "--retrieval_embed_stats_last_m",
        type=int,
        default=None,
        help="Last-m context points used to compute stats features. Default: use --shift_last_m.",
    )
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument(
        "--query_stride",
        type=int,
        default=1,
        help="Subsample query window start indices by this stride to control teacher dataset size (1 = use all).",
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sanity_check", action="store_true")

    args = parser.parse_args()

    if args.teacher_align == "instance":
        args.teacher_align = "retrieved_to_query"

    if args.rerank_tau is None:
        args.rerank_tau = float(args.tau)
    if int(args.rerank_k0) > 0 and args.rerank_mode == "none":
        raise ValueError("--rerank_k0>0 requires --rerank_mode != none")
    if args.rerank_mode != "none":
        if int(args.rerank_k0) <= 0:
            raise ValueError("--rerank_mode != none requires --rerank_k0 > 0")
        if int(args.rerank_k0) < int(args.k):
            raise ValueError(f"--rerank_k0 ({args.rerank_k0}) must be >= --k ({args.k})")
        if args.teacher_mode != "weighted_quantile":
            raise ValueError("--rerank_mode is only supported for teacher_mode=weighted_quantile")
        if args.rerank_weight_source == "raw" and float(args.rerank_tau) <= 0:
            raise ValueError("--rerank_tau must be > 0 when rerank_weight_source=raw")
    else:
        if args.rerank_weight_source != "embedding":
            raise ValueError("--rerank_weight_source!=embedding requires --rerank_mode != none")

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.sanity_check:
        sanity_check_weighted_quantile(device=device)
        return

    # Hard guard: prevent leakage by restricting KB to train split.
    if args.kb_split != "train":
        raise ValueError("--kb_split must be 'train' to avoid leakage (val/test teacher must retrieve from train KB).")

    if args.sample_mode == "multivariate":
        if args.teacher_mode != "weighted_quantile":
            raise ValueError("sample_mode=multivariate currently supports teacher_mode=weighted_quantile only")
        if args.query_embedding_source != "cache":
            raise ValueError("sample_mode=multivariate currently supports query_embedding_source=cache only")

    dataset_name = Path(args.data_path).stem
    quantiles = _load_quantiles(args.base_model_path, args.override_quantiles)
    quantiles_t = torch.tensor(quantiles, dtype=torch.float32, device=device)
    q50_idx = int(np.argmin(np.abs(np.asarray(quantiles, dtype=np.float32) - 0.5)))
    q50_val = float(quantiles[q50_idx])
    if abs(q50_val - 0.5) > 1e-6:
        print(f"[warn] quantiles has no exact 0.5; using closest q={q50_val} at idx={q50_idx} for teacher_q50 metrics")

    base_config = AutoConfig.from_pretrained(args.base_model_path)
    base_pred_len = int(base_config.chronos_config["prediction_length"])
    pred_len = base_pred_len if args.pred_len is None else int(args.pred_len)
    if pred_len != base_pred_len:
        print(f"[warn] pred_len={pred_len} != base prediction_length={base_pred_len}")
    if args.teacher_mode == "rag_output" and pred_len != base_pred_len:
        raise ValueError(
            "teacher_mode=rag_output currently requires pred_len == base prediction_length "
            f"(pred_len={pred_len}, base={base_pred_len}). "
            "TS-RAG ChronosBolt retrieval augmentation is trained for a fixed horizon (typically 64)."
        )
    if args.teacher_mode == "rag_output" and args.teacher_align != "none":
        print(
            f"[warn] teacher_align={args.teacher_align} ignored when teacher_mode=rag_output "
            "(teacher comes from TS-RAG forward pass)."
        )

    # Borders and scaling (fit scaler on TRAIN split to avoid leakage).
    raw, var_names = _load_raw_series(args.root_path, args.data_path, args.features, args.target)
    total_length = raw.shape[0]
    n_windows = int(total_length) - int(args.seq_len) + 1
    if n_windows <= 0:
        raise ValueError(f"Series too short for seq_len={args.seq_len}: total_length={total_length}")
    retrieval_feat_id: Optional[int] = None
    retrieval_var_name: Optional[str] = None
    if args.sample_mode == "multivariate":
        if args.retrieval_feature_id is not None:
            retrieval_feat_id = int(args.retrieval_feature_id)
        elif args.target in var_names:
            retrieval_feat_id = int(var_names.index(args.target))
        else:
            retrieval_feat_id = 0
        if retrieval_feat_id < 0 or retrieval_feat_id >= len(var_names):
            raise ValueError(f"retrieval_feature_id out of range: {retrieval_feat_id} for num_channels={len(var_names)}")
        retrieval_var_name = var_names[retrieval_feat_id]
    try:
        border1s, border2s = get_borders(dataset_name, args.seq_len, total_length=total_length)
    except ValueError as e:
        # Fall back to the same generic 0.7/0.1/0.2 split used by Dataset_Custom_S for unknown datasets.
        if "Unknown dataset name" not in str(e):
            raise
        num_train = int(total_length * 0.7)
        num_test = int(total_length * 0.2)
        num_vali = total_length - num_train - num_test
        border1s = [0, num_train - args.seq_len, total_length - num_test - args.seq_len]
        border2s = [num_train, num_train + num_vali, total_length]
        print(f"[warn] get_borders failed for dataset_name={dataset_name}: {e}; using generic split.")

    kb_type = _parse_split(args.kb_split)
    kb_end = int(border2s[kb_type])
    kb_max_start = kb_end - (args.seq_len + pred_len)
    if kb_max_start <= 0:
        raise ValueError(f"KB split too short: kb_end={kb_end}, need >= seq_len+pred_len")
    kb_slice_end = kb_max_start + 1  # exclusive end for embedding start indices

    scaler = StandardScaler()
    scaler.fit(raw[border1s[0] : border2s[0]])
    means_np = scaler.mean_.astype(np.float32, copy=False)
    scales_np = scaler.scale_.astype(np.float32, copy=False)
    scales_np = np.where(scales_np == 0, 1.0, scales_np).astype(np.float32, copy=False)

    # Optional: augment retrieval embeddings with extra features (time/stats) to improve MAE.
    # These features are used ONLY for offline retrieval in teacher building.
    use_time_feats = bool(args.retrieval_embed_time_features)
    use_stats_feats = bool(args.retrieval_embed_stats_features)
    time_feats_start: Optional[np.ndarray] = None  # (n_windows, 4)
    stats_mean_start: Optional[np.ndarray] = None  # (n_windows, C)
    stats_logstd_start: Optional[np.ndarray] = None  # (n_windows, C)
    stats_slope_start: Optional[np.ndarray] = None  # (n_windows, C)
    stats_mu: Optional[np.ndarray] = None  # (C, 3)
    stats_sigma: Optional[np.ndarray] = None  # (C, 3)
    stats_last_m = int(args.shift_last_m) if args.retrieval_embed_stats_last_m is None else int(args.retrieval_embed_stats_last_m)
    if use_time_feats:
        ts = _load_datetime_index(args.root_path, args.data_path)
        tf = _compute_time_phase_features(ts)  # (T,4)
        offset = int(args.seq_len) - 1
        time_feats_start = tf[offset : offset + n_windows].astype(np.float32, copy=False)
        time_feats_start = time_feats_start * float(args.retrieval_embed_time_weight)

    if use_stats_feats:
        # Compute stats on dataset scale (StandardScaler output) to match teacher/base outputs.
        z = ((raw - means_np) / scales_np).astype(np.float32, copy=False)  # (T,C)
        mean_t, std_t, slope_t = _rolling_mean_std_slope(z, m=stats_last_m)  # (T,C)
        logstd_t = np.log(std_t + 1e-6, dtype=np.float32)
        del z, std_t

        offset = int(args.seq_len) - 1
        stats_mean_start = mean_t[offset : offset + n_windows].astype(np.float32, copy=False)
        stats_logstd_start = logstd_t[offset : offset + n_windows].astype(np.float32, copy=False)
        stats_slope_start = slope_t[offset : offset + n_windows].astype(np.float32, copy=False)
        del mean_t, logstd_t, slope_t

        # Z-score stats per feature using KB(train) windows only (no leakage).
        kb_slice = slice(0, int(kb_slice_end))
        mu_mean = stats_mean_start[kb_slice].mean(axis=0)
        mu_logstd = stats_logstd_start[kb_slice].mean(axis=0)
        mu_slope = stats_slope_start[kb_slice].mean(axis=0)
        sig_mean = stats_mean_start[kb_slice].std(axis=0) + 1e-6
        sig_logstd = stats_logstd_start[kb_slice].std(axis=0) + 1e-6
        sig_slope = stats_slope_start[kb_slice].std(axis=0) + 1e-6
        stats_mu = np.stack([mu_mean, mu_logstd, mu_slope], axis=1).astype(np.float32, copy=False)
        stats_sigma = np.stack([sig_mean, sig_logstd, sig_slope], axis=1).astype(np.float32, copy=False)

    split_type = _parse_split(args.split)
    q_border1 = int(border1s[split_type])
    q_border2 = int(border2s[split_type])

    # Prepare output dir + manifest.
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_obj = TeacherMeta(
        dataset=dataset_name,
        split=args.split,
        kb_split=args.kb_split,
        root_path=args.root_path,
        data_path=args.data_path,
        features=args.features,
        target=args.target,
        context_len=args.seq_len,
        pred_len=pred_len,
        k=args.k,
        tau=float(args.tau),
        drop_self_eps=float(args.drop_self_eps),
        drop_self_mode=str(args.drop_self_mode),
        query_stride=int(args.query_stride),
        quantiles=quantiles,
        base_model_path=args.base_model_path,
        embedding_model_path=args.embedding_model_path,
        teacher_mode=str(args.teacher_mode),
        rag_model_ckpt=str(args.rag_model_ckpt),
        rag_augment_mode=str(args.rag_augment_mode),
        teacher_align=str(args.teacher_align),
        shift_last_m=int(args.shift_last_m),
        rerank_mode=str(args.rerank_mode),
        rerank_k0=int(args.rerank_k0),
        rerank_weight_source=str(args.rerank_weight_source),
        rerank_tau=float(args.rerank_tau),
        retrieved_loc_scale=str(args.retrieved_loc_scale),
        dist_transform=str(args.dist_transform),
        distance_metric=str(args.distance_metric),
    )
    meta = meta_obj.__dict__
    meta["query_embedding_source"] = args.query_embedding_source
    # Map FAISS neighbor indices -> global window start indices.
    # We build the index from emb_all[:kb_slice_end], so local index == global start (offset=0).
    meta["kb_end"] = kb_end
    meta["kb_slice_end"] = kb_slice_end
    meta["kb_window_start_offset"] = 0
    meta["conf_fields"] = ["conf_w_max", "conf_entropy", "conf_effective_k"]
    meta["sample_mode"] = args.sample_mode
    meta["num_channels"] = len(var_names)
    meta["retrieval_embed_time_features"] = bool(use_time_feats)
    meta["retrieval_embed_time_weight"] = float(args.retrieval_embed_time_weight)
    meta["retrieval_embed_stats_features"] = bool(use_stats_feats)
    meta["retrieval_embed_stats_weight"] = float(args.retrieval_embed_stats_weight)
    meta["retrieval_embed_stats_last_m"] = int(stats_last_m)
    if retrieval_feat_id is not None:
        meta["retrieval_feature_id"] = int(retrieval_feat_id)
        meta["retrieval_feature_name"] = str(retrieval_var_name)

    manifest_path = out_dir / args.split / "manifest.json"
    (out_dir / args.split).mkdir(parents=True, exist_ok=True)

    # Load retrieval database cache.
    # Prefer the legacy monolithic pickle (<dataset>_*_<seq_len>.pkl) for small datasets,
    # but also support the directory cache format:
    #   <dataset>_*_<seq_len>/embeddings/feat_XXXX.npy
    if args.retrieval_db_path:
        retrieval_db_path = Path(args.retrieval_db_path)
        if not retrieval_db_path.exists():
            raise FileNotFoundError(f"--retrieval_db_path not found: {retrieval_db_path}")
        db_candidates_pkl = [retrieval_db_path] if retrieval_db_path.suffix == ".pkl" else []
        db_candidates_dir = [retrieval_db_path] if retrieval_db_path.is_dir() else []
        if not db_candidates_pkl and not db_candidates_dir:
            raise ValueError(
                f"--retrieval_db_path must point to a .pkl file or a directory cache, got: {retrieval_db_path}"
            )
    else:
        db_candidates_pkl = sorted(
            Path(args.retrieval_database_dir).glob(f"{dataset_name}_*_{args.seq_len}.pkl")
        )
        db_candidates_dir = sorted(
            p
            for p in Path(args.retrieval_database_dir).glob(f"{dataset_name}_*_{args.seq_len}")
            if p.is_dir()
        )

    retrieval_db_path: Path
    retrieval_db_mode: str
    embeddings_dir: Optional[Path] = None

    # Build one FAISS index per variable from cached embeddings (KB=train only).
    faiss_indices: List[faiss.IndexFlatL2] = []
    var_embeddings: List[np.ndarray] = []
    d_infer: Optional[int] = None
    retrieval_index: Optional[faiss.IndexFlatL2] = None
    retrieval_emb_all: Optional[np.ndarray] = None
    extra_dim = (4 if use_time_feats else 0) + (3 if use_stats_feats else 0)

    if db_candidates_pkl:
        if len(db_candidates_pkl) > 1:
            print(
                f"[warn] multiple retrieval DB pkl candidates found: {[str(p) for p in db_candidates_pkl]} ; using {db_candidates_pkl[0]}"
            )
        retrieval_db_path = db_candidates_pkl[0]
        retrieval_db_mode = "pkl"
        meta["retrieval_db_path"] = str(retrieval_db_path)
        meta["retrieval_db_format"] = "pkl"
        print(f"[info] using cached retrieval DB (pkl): {retrieval_db_path}")

        import pickle

        with open(retrieval_db_path, "rb") as f:
            databases = pickle.load(f)

        if args.sample_mode == "multivariate":
            assert retrieval_var_name is not None
            if retrieval_var_name not in databases:
                raise KeyError(f"Variable {retrieval_var_name} not found in retrieval database {retrieval_db_path}")
            emb = np.asarray(databases[retrieval_var_name]["embeddings"], dtype=np.float32)
            if emb.ndim != 2:
                raise ValueError(f"Expected embeddings to be 2D (n_windows,d), got {emb.shape} for var={retrieval_var_name}")
            if int(emb.shape[0]) != int(n_windows):
                raise ValueError(
                    f"Retrieval embeddings length mismatch for {retrieval_var_name}: "
                    f"emb_windows={emb.shape[0]} != n_windows={n_windows} (seq_len={args.seq_len})"
                )
            d_infer = int(emb.shape[1])
            d_index = int(d_infer) + int(extra_dim)
            index = faiss.IndexFlatL2(int(d_index))
            emb_kb = np.asarray(emb[:kb_slice_end], dtype=np.float32)
            extra_parts: List[np.ndarray] = []
            if use_time_feats:
                assert time_feats_start is not None
                extra_parts.append(np.asarray(time_feats_start[:kb_slice_end], dtype=np.float32))
            if use_stats_feats:
                assert stats_mean_start is not None and stats_logstd_start is not None and stats_slope_start is not None
                assert stats_mu is not None and stats_sigma is not None
                assert retrieval_feat_id is not None
                s = np.stack(
                    [
                        stats_mean_start[:kb_slice_end, retrieval_feat_id],
                        stats_logstd_start[:kb_slice_end, retrieval_feat_id],
                        stats_slope_start[:kb_slice_end, retrieval_feat_id],
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
                mu = stats_mu[retrieval_feat_id]
                sig = stats_sigma[retrieval_feat_id]
                s = ((s - mu) / sig) * float(args.retrieval_embed_stats_weight)
                extra_parts.append(s.astype(np.float32, copy=False))
            if extra_parts:
                emb_kb = np.concatenate([emb_kb, *extra_parts], axis=1).astype(np.float32, copy=False)
            if args.distance_metric == "cosine":
                emb_kb = emb_kb.copy()
                faiss.normalize_L2(emb_kb)
            index.add(emb_kb)
            retrieval_index = index
            retrieval_emb_all = emb
            meta["retrieval_embed_dim_base"] = int(d_infer)
            meta["retrieval_embed_extra_dim"] = int(extra_dim)
            meta["retrieval_embed_dim"] = int(d_index)
        else:
            for feat_id, var in enumerate(var_names):
                if var not in databases:
                    raise KeyError(f"Variable {var} not found in retrieval database {retrieval_db_path}")
                emb = np.asarray(databases[var]["embeddings"], dtype=np.float32)
                if emb.ndim != 2:
                    raise ValueError(f"Expected embeddings to be 2D (n_windows,d), got {emb.shape} for var={var}")
                if int(emb.shape[0]) != int(n_windows):
                    raise ValueError(
                        f"Retrieval embeddings length mismatch for {var}: "
                        f"emb_windows={emb.shape[0]} != n_windows={n_windows} (seq_len={args.seq_len})"
                    )
                if d_infer is None:
                    d_infer = int(emb.shape[1])
                elif int(emb.shape[1]) != int(d_infer):
                    raise ValueError(f"Embedding dimension mismatch across variables: {var} has {emb.shape[1]} != {d_infer}")

                d_index = int(d_infer) + int(extra_dim)
                index = faiss.IndexFlatL2(int(d_index))
                emb_kb = np.asarray(emb[:kb_slice_end], dtype=np.float32)
                extra_parts = []
                if use_time_feats:
                    assert time_feats_start is not None
                    extra_parts.append(np.asarray(time_feats_start[:kb_slice_end], dtype=np.float32))
                if use_stats_feats:
                    assert stats_mean_start is not None and stats_logstd_start is not None and stats_slope_start is not None
                    assert stats_mu is not None and stats_sigma is not None
                    s = np.stack(
                        [
                            stats_mean_start[:kb_slice_end, feat_id],
                            stats_logstd_start[:kb_slice_end, feat_id],
                            stats_slope_start[:kb_slice_end, feat_id],
                        ],
                        axis=1,
                    ).astype(np.float32, copy=False)
                    mu = stats_mu[feat_id]
                    sig = stats_sigma[feat_id]
                    s = ((s - mu) / sig) * float(args.retrieval_embed_stats_weight)
                    extra_parts.append(s.astype(np.float32, copy=False))
                if extra_parts:
                    emb_kb = np.concatenate([emb_kb, *extra_parts], axis=1).astype(np.float32, copy=False)
                if args.distance_metric == "cosine":
                    emb_kb = emb_kb.copy()
                    faiss.normalize_L2(emb_kb)
                index.add(emb_kb)

                faiss_indices.append(index)
                var_embeddings.append(emb)
            if d_infer is not None:
                meta["retrieval_embed_dim_base"] = int(d_infer)
                meta["retrieval_embed_extra_dim"] = int(extra_dim)
                meta["retrieval_embed_dim"] = int(d_infer) + int(extra_dim)
    else:
        # Directory cache format.
        if not db_candidates_dir:
            raise FileNotFoundError(
                f"No retrieval database cache found for dataset={dataset_name} seq_len={args.seq_len} in {args.retrieval_database_dir}. "
                f"Expected either {dataset_name}_*_{args.seq_len}.pkl OR {dataset_name}_*_{args.seq_len}/embeddings/feat_*.npy"
            )
        if len(db_candidates_dir) > 1:
            print(
                f"[warn] multiple retrieval DB dir candidates found: {[str(p) for p in db_candidates_dir]} ; using {db_candidates_dir[0]}"
            )
        retrieval_db_path = db_candidates_dir[0]
        retrieval_db_mode = "dir"
        embeddings_dir = retrieval_db_path / "embeddings"
        if not embeddings_dir.is_dir():
            raise FileNotFoundError(f"retrieval DB dir missing embeddings/: {retrieval_db_path}")
        meta["retrieval_db_path"] = str(retrieval_db_path)
        meta["retrieval_db_format"] = "dir"
        print(f"[info] using cached retrieval DB (dir): {retrieval_db_path}")

        if args.sample_mode == "multivariate":
            assert embeddings_dir is not None
            assert retrieval_feat_id is not None
            emb_path = embeddings_dir / f"feat_{int(retrieval_feat_id):04d}.npy"
            if not emb_path.exists():
                raise FileNotFoundError(f"Missing retrieval embeddings for retrieval_feature_id={retrieval_feat_id}: {emb_path}")
            retrieval_emb_all = np.load(emb_path, mmap_mode="r")
            if retrieval_emb_all.ndim != 2:
                raise ValueError(
                    f"Expected retrieval embeddings to be 2D (n_windows,d), got {retrieval_emb_all.shape} for {emb_path}"
                )
            if int(retrieval_emb_all.shape[0]) != int(n_windows):
                raise ValueError(
                    f"Retrieval embeddings length mismatch for {emb_path}: "
                    f"emb_windows={retrieval_emb_all.shape[0]} != n_windows={n_windows} (seq_len={args.seq_len})"
                )
            d_infer = int(retrieval_emb_all.shape[1])
            d_index = int(d_infer) + int(extra_dim)
            retrieval_index = faiss.IndexFlatL2(int(d_index))
            emb_kb = np.asarray(retrieval_emb_all[:kb_slice_end], dtype=np.float32)
            extra_parts = []
            if use_time_feats:
                assert time_feats_start is not None
                extra_parts.append(np.asarray(time_feats_start[:kb_slice_end], dtype=np.float32))
            if use_stats_feats:
                assert stats_mean_start is not None and stats_logstd_start is not None and stats_slope_start is not None
                assert stats_mu is not None and stats_sigma is not None
                s = np.stack(
                    [
                        stats_mean_start[:kb_slice_end, retrieval_feat_id],
                        stats_logstd_start[:kb_slice_end, retrieval_feat_id],
                        stats_slope_start[:kb_slice_end, retrieval_feat_id],
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
                mu = stats_mu[retrieval_feat_id]
                sig = stats_sigma[retrieval_feat_id]
                s = ((s - mu) / sig) * float(args.retrieval_embed_stats_weight)
                extra_parts.append(s.astype(np.float32, copy=False))
            if extra_parts:
                emb_kb = np.concatenate([emb_kb, *extra_parts], axis=1).astype(np.float32, copy=False)
            if args.distance_metric == "cosine":
                emb_kb = emb_kb.copy()
                faiss.normalize_L2(emb_kb)
            retrieval_index.add(emb_kb)
            meta["retrieval_feature_embeddings_path"] = str(emb_path)
            meta["retrieval_embed_dim_base"] = int(d_infer)
            meta["retrieval_embed_extra_dim"] = int(extra_dim)
            meta["retrieval_embed_dim"] = int(d_index)

    # Precompute per-variable mean/scale for fast scaling.
    means = means_np
    scales = scales_np

    rag_model = None
    if args.teacher_mode == "rag_output":
        rag_ckpt_path = Path(args.rag_model_ckpt)
        if not rag_ckpt_path.exists():
            raise FileNotFoundError(f"rag_model_ckpt not found: {rag_ckpt_path}")

        # Lazily import to keep weighted-quantile teacher lightweight.
        from collections import OrderedDict
        from models.ChronosBolt import ChronosBoltModelForForecastingWithRetrieval

        rag_config = AutoConfig.from_pretrained(args.base_model_path)
        rag_model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
            args.base_model_path, config=rag_config, augment=str(args.rag_augment_mode)
        )
        state_dict = torch.load(rag_ckpt_path, map_location="cpu")
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_state_dict[key.replace("module.", "")] = value
        rag_model.load_state_dict(new_state_dict, strict=True)
        rag_model.to(device)
        rag_model.eval()
        print(f"[info] loaded rag teacher model ckpt={rag_ckpt_path} augment={args.rag_augment_mode}")

    # Build query start indices for this split (global indices).
    q_max_start = q_border2 - (args.seq_len + pred_len)
    if q_max_start < q_border1:
        raise ValueError(f"Split too short: {args.split} has no valid windows for seq_len+pred_len.")
    query_starts = np.arange(q_border1, q_max_start + 1, dtype=np.int64)
    if int(args.query_stride) > 1:
        query_starts = query_starts[:: int(args.query_stride)]

    shard_idx = 0
    shard_entries: List[Dict] = []
    shard_context: List[torch.Tensor] = []
    shard_target: List[torch.Tensor] = []
    shard_q_teacher: List[torch.Tensor] = []
    shard_conf_w_max: List[torch.Tensor] = []
    shard_conf_entropy: List[torch.Tensor] = []
    shard_conf_effective_k: List[torch.Tensor] = []

    written: List[Dict] = []
    total_queries = 0
    dropped_self = 0
    self_not_found = 0
    skipped_all_inf = 0
    teacher_q50_abs_sum = 0.0
    teacher_q50_sq_sum = 0.0
    teacher_q50_count = 0

    if args.sample_mode == "multivariate":
        if retrieval_index is None or retrieval_emb_all is None or retrieval_feat_id is None:
            raise ValueError("sample_mode=multivariate requires retrieval_index/retrieval_emb_all/retrieval_feat_id")

        # Use a single retrieval feature for neighbor search, but gather retrieved_y for all channels.
        mean_vec = torch.from_numpy(means).to(device=device, dtype=torch.float32).view(1, 1, -1)  # (1,1,C)
        scale_vec = torch.from_numpy(scales).to(device=device, dtype=torch.float32).view(1, 1, -1)  # (1,1,C)
        mean_bc = mean_vec.view(1, 1, 1, -1)  # (1,1,1,C)
        scale_bc = scale_vec.view(1, 1, 1, -1)  # (1,1,1,C)

        train_raw_all = torch.from_numpy(raw[:kb_end, :]).to(device=device, dtype=torch.float32)  # (kb_end,C)
        emb_all = retrieval_emb_all
        index = retrieval_index

        for start_i in range(0, len(query_starts), args.batch_size):
            starts = query_starts[start_i : start_i + args.batch_size]

            # Raw context/target windows for all channels (B,L,C) and (B,T,C)
            offsets_ctx = np.arange(args.seq_len, dtype=np.int64)
            idx_ctx = starts[:, None] + offsets_ctx[None, :]
            ctx_raw = torch.from_numpy(raw[idx_ctx]).to(device=device, dtype=torch.float32)

            offsets_tgt = np.arange(args.seq_len, args.seq_len + pred_len, dtype=np.int64)
            idx_tgt = starts[:, None] + offsets_tgt[None, :]
            tgt_raw = torch.from_numpy(raw[idx_tgt]).to(device=device, dtype=torch.float32)

            query_vec = np.asarray(emb_all[starts], dtype=np.float32)
            extra_parts = []
            if use_time_feats:
                assert time_feats_start is not None
                extra_parts.append(np.asarray(time_feats_start[starts], dtype=np.float32))
            if use_stats_feats:
                assert stats_mean_start is not None and stats_logstd_start is not None and stats_slope_start is not None
                assert stats_mu is not None and stats_sigma is not None
                s = np.stack(
                    [
                        stats_mean_start[starts, retrieval_feat_id],
                        stats_logstd_start[starts, retrieval_feat_id],
                        stats_slope_start[starts, retrieval_feat_id],
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
                mu = stats_mu[retrieval_feat_id]
                sig = stats_sigma[retrieval_feat_id]
                s = ((s - mu) / sig) * float(args.retrieval_embed_stats_weight)
                extra_parts.append(s.astype(np.float32, copy=False))
            if extra_parts:
                query_vec = np.concatenate([query_vec, *extra_parts], axis=1).astype(np.float32, copy=False)
            query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
            if args.distance_metric == "cosine":
                faiss.normalize_L2(query_vec)

            # Search k0+1 then drop self and keep top-k0 by embedding distance.
            k0 = int(args.rerank_k0) if args.rerank_mode != "none" else int(args.k)
            distances_raw, indices_raw = index.search(query_vec, k0 + 1)
            distances_np = distances_raw.astype(np.float64, copy=True)
            indices_np = indices_raw.astype(np.int64, copy=False)

            if args.drop_self_mode == "idx":
                in_kb = starts < kb_slice_end
                if args.split == "train" and in_kb.any():
                    self_idx = starts
                    mask = (indices_np == self_idx[:, None]) & in_kb[:, None]
                    dropped_self += int(mask.any(axis=1).sum())
                    self_not_found += int((in_kb & (~mask.any(axis=1))).sum())
                    distances_np[mask] = np.inf
            else:
                distances_np[distances_np <= float(args.drop_self_eps)] = np.inf

            order = np.argsort(distances_np, axis=1)[:, :k0]
            distances_np = np.take_along_axis(distances_np, order, axis=1).astype(np.float32)
            indices_np = np.take_along_axis(indices_np, order, axis=1).astype(np.int64)

            row_all_inf = np.isinf(distances_np).all(axis=1)
            if row_all_inf.any():
                skipped_all_inf += int(row_all_inf.sum())
                keep = ~row_all_inf
                distances_np = distances_np[keep]
                indices_np = indices_np[keep]
                keep_t = torch.from_numpy(keep).to(device=device)
                ctx_raw = ctx_raw[keep_t]
                tgt_raw = tgt_raw[keep_t]
                if distances_np.shape[0] == 0:
                    continue

            total_queries += int(distances_np.shape[0])
            distances = torch.from_numpy(distances_np).to(device=device, dtype=torch.float32)
            timestamp_idx = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)

            # Transform context/target to dataset scale (matches q_base output scale).
            ctx = (ctx_raw - mean_vec) / scale_vec  # (B,L,C)
            tgt = (tgt_raw - mean_vec) / scale_vec  # (B,pred_len,C)

            shift = None
            raw_dist = None
            shift_modes = {"shift_last", "shift_mean_last_m"}
            need_ctx = (
                args.teacher_align in shift_modes
                or args.rerank_mode != "none"
                or args.rerank_weight_source == "raw"
            )
            if need_ctx:
                offsets_ctx_t = torch.arange(0, args.seq_len, device=device, dtype=torch.long)
                idx_ctx_t = timestamp_idx.unsqueeze(-1) + offsets_ctx_t.view(1, 1, -1)  # (B,K0,L)
                retrieved_ctx_raw = train_raw_all[idx_ctx_t]  # (B,K0,L,C) RAW
                retrieved_ctx = (retrieved_ctx_raw - mean_bc) / scale_bc  # dataset scale

                shift_mode = (
                    args.teacher_align
                    if args.teacher_align in shift_modes
                    else "shift_mean_last_m"
                )
                shift = _compute_level_shift_multivariate(
                    ctx, retrieved_ctx, mode=str(shift_mode), last_m=int(args.shift_last_m)
                )  # (B,K0,C)

                if args.rerank_mode != "none" or args.rerank_weight_source == "raw":
                    retrieved_ctx_shifted = retrieved_ctx + shift.unsqueeze(2)  # (B,K0,L,C)
                    diff_ctx = ctx.unsqueeze(1) - retrieved_ctx_shifted
                    if args.rerank_mode == "raw_l2_shift":
                        raw_dist = (diff_ctx * diff_ctx).mean(dim=(2, 3))
                    else:
                        raw_dist = diff_ctx.abs().mean(dim=(2, 3))
                    raw_dist = torch.where(
                        torch.isfinite(distances),
                        raw_dist,
                        torch.full_like(raw_dist, float("inf")),
                    )

                if args.rerank_mode != "none":
                    sel = torch.topk(raw_dist, k=int(args.k), dim=1, largest=False).indices  # (B,K)
                    timestamp_idx = torch.gather(timestamp_idx, 1, sel)
                    distances = torch.gather(distances, 1, sel)
                    raw_dist = torch.gather(raw_dist, 1, sel)
                    assert shift is not None
                    sel_c = sel.unsqueeze(-1).expand(-1, -1, int(shift.shape[-1]))
                    shift = torch.gather(shift, 1, sel_c)

            # Gather retrieved horizons y_i (RAW scale) and transform to dataset scale.
            offsets_y = torch.arange(args.seq_len, args.seq_len + pred_len, device=device, dtype=torch.long)
            idx_y = timestamp_idx.unsqueeze(-1) + offsets_y.view(1, 1, -1)  # (B,K,T)
            retrieved_y_raw = train_raw_all[idx_y]  # (B,K,T,C) RAW
            retrieved_y = (retrieved_y_raw - mean_bc) / scale_bc  # dataset scale

            if args.teacher_align in shift_modes:
                assert shift is not None
                retrieved_y = retrieved_y + shift.unsqueeze(2)  # (B,K,1,C)

            # Build weights + confidence (used by dynamic-beta training).
            if args.rerank_mode != "none" and args.rerank_weight_source == "raw":
                assert raw_dist is not None
                distances_for_w = raw_dist
                tau_w = float(args.rerank_tau)
            else:
                distances_for_w = distances
                tau_w = float(args.tau)

            if args.dist_transform == "sqrt":
                distances_for_w = torch.sqrt(torch.clamp(distances_for_w, min=0.0))

            logits = -distances_for_w / tau_w
            logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -float("inf")))
            w = torch.softmax(logits, dim=1)

            bad = torch.isnan(w).any(dim=1)
            if bad.any():
                skipped_all_inf += int(bad.sum().item())
                keep = ~bad
                if keep.sum() == 0:
                    continue
                retrieved_y = retrieved_y[keep]
                ctx = ctx[keep]
                tgt = tgt[keep]
                w = w[keep]
                timestamp_idx = timestamp_idx[keep]
                distances = distances[keep]
                if shift is not None:
                    shift = shift[keep]
                if raw_dist is not None:
                    raw_dist = raw_dist[keep]

            conf_w_max = w.max(dim=1).values  # (B,)
            conf_entropy = -(w * torch.log(w + 1e-12)).sum(dim=1)  # (B,)
            conf_effective_k = 1.0 / (w.square().sum(dim=1) + 1e-12)  # (B,)

            if args.teacher_align in {"none", "shift_last", "shift_mean_last_m"}:
                q_teacher = weighted_quantile_torch(retrieved_y, w, quantiles_t)  # (B,Q,pred_len,C)
            elif args.teacher_align == "retrieved_to_query":
                # Per-channel InstanceNorm: normalize each retrieved sequence, inverse to query window scale.
                ctx_series = ctx.permute(0, 2, 1)  # (B,C,L)
                loc_q, scale_q, _is_const_q = _instance_norm_stats(ctx_series)  # (B,C,1)

                offsets_full = torch.arange(0, args.seq_len + pred_len, device=device, dtype=torch.long)
                idx_full = timestamp_idx.unsqueeze(-1) + offsets_full.view(1, 1, -1)  # (B,K,L+T)
                retrieved_seq_raw = train_raw_all[idx_full]  # (B,K,L+T,C) RAW
                retrieved_seq = (retrieved_seq_raw - mean_bc) / scale_bc  # dataset scale

                if args.retrieved_loc_scale == "context":
                    retrieved_for_stats = retrieved_seq[:, :, : args.seq_len, :]
                else:
                    retrieved_for_stats = retrieved_seq

                loc_r, scale_r, is_const_r = _instance_norm_stats(retrieved_for_stats.permute(0, 1, 3, 2))  # (B,K,C,1)
                retrieved_y_perm = retrieved_y.permute(0, 1, 3, 2)  # (B,K,C,T)
                retrieved_y_norm = _instance_norm_apply(retrieved_y_perm, loc_r, scale_r, is_const_r)  # (B,K,C,T)
                retrieved_y_aligned = _instance_norm_inverse(
                    retrieved_y_norm, loc_q.unsqueeze(1), scale_q.unsqueeze(1)
                ).permute(0, 1, 3, 2)  # (B,K,T,C)

                q_teacher = weighted_quantile_torch(retrieved_y_aligned, w, quantiles_t)  # (B,Q,T,C)
            else:
                raise ValueError(f"Unsupported teacher_align={args.teacher_align}")

            q50_pred = q_teacher[:, q50_idx, ...]
            diff = (q50_pred - tgt).to(dtype=torch.float32)
            teacher_q50_abs_sum += float(diff.abs().sum().item())
            teacher_q50_sq_sum += float((diff * diff).sum().item())
            teacher_q50_count += int(diff.numel())

            shard_context.append(ctx.detach().cpu())
            shard_target.append(tgt.detach().cpu())
            shard_q_teacher.append(q_teacher.detach().cpu())
            shard_conf_w_max.append(conf_w_max.detach().cpu())
            shard_conf_entropy.append(conf_entropy.detach().cpu())
            shard_conf_effective_k.append(conf_effective_k.detach().cpu())

            cur_n = sum(t.shape[0] for t in shard_context)
            if cur_n >= args.shard_size:
                context_cat = torch.cat(shard_context, dim=0)
                target_cat = torch.cat(shard_target, dim=0)
                q_teacher_cat = torch.cat(shard_q_teacher, dim=0)
                conf_w_max_cat = torch.cat(shard_conf_w_max, dim=0)
                conf_entropy_cat = torch.cat(shard_conf_entropy, dim=0)
                conf_effective_k_cat = torch.cat(shard_conf_effective_k, dim=0)
                shard_name = _save_shard(
                    out_dir,
                    args.split,
                    shard_idx,
                    context_cat,
                    target_cat,
                    q_teacher_cat,
                    meta,
                    conf_w_max=conf_w_max_cat,
                    conf_entropy=conf_entropy_cat,
                    conf_effective_k=conf_effective_k_cat,
                )
                written.append({"file": shard_name, "num_samples": int(context_cat.shape[0])})
                shard_idx += 1
                shard_context, shard_target, shard_q_teacher = [], [], []
                shard_conf_w_max, shard_conf_entropy, shard_conf_effective_k = [], [], []

        # Flush remaining.
        if shard_context:
            context_cat = torch.cat(shard_context, dim=0)
            target_cat = torch.cat(shard_target, dim=0)
            q_teacher_cat = torch.cat(shard_q_teacher, dim=0)
            conf_w_max_cat = torch.cat(shard_conf_w_max, dim=0)
            conf_entropy_cat = torch.cat(shard_conf_entropy, dim=0)
            conf_effective_k_cat = torch.cat(shard_conf_effective_k, dim=0)
            shard_name = _save_shard(
                out_dir,
                args.split,
                shard_idx,
                context_cat,
                target_cat,
                q_teacher_cat,
                meta,
                conf_w_max=conf_w_max_cat,
                conf_entropy=conf_entropy_cat,
                conf_effective_k=conf_effective_k_cat,
            )
            written.append({"file": shard_name, "num_samples": int(context_cat.shape[0])})

        teacher_q50_mae = float(teacher_q50_abs_sum / max(1, teacher_q50_count))
        teacher_q50_mse = float(teacher_q50_sq_sum / max(1, teacher_q50_count))
        manifest = {
            "meta": meta,
            "shards": written,
            "total_queries": total_queries,
            "dropped_self": dropped_self,
            "self_not_found": self_not_found,
            "skipped_all_inf": skipped_all_inf,
            "teacher_q50_mae": teacher_q50_mae,
            "teacher_q50_mse": teacher_q50_mse,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[done] wrote {len(written)} shard(s) to {out_dir / args.split}")
        if args.drop_self_mode == "idx" and args.split == "train":
            print(
                f"[drop_self_idx] dropped_self={dropped_self} self_not_found={self_not_found} total_queries={total_queries}"
            )
            if self_not_found:
                print(
                    "[warn] self_not_found>0: retrieval DB / window indexing may be misaligned; "
                    "fall back to drop_self_mode=dist if unsure."
                )
        if skipped_all_inf:
            print(f"[warn] skipped {skipped_all_inf} samples with no valid neighbors after drop_self")
        return

    # Process each variable independently (TS-RAG treats each variable as a separate sample).
    for feat_id, var in enumerate(var_names):
        mean = torch.tensor(means[feat_id], device=device)
        scale = torch.tensor(scales[feat_id], device=device)

        train_raw = torch.from_numpy(raw[:kb_end, feat_id]).to(device=device, dtype=torch.float32)

        if retrieval_db_mode == "pkl":
            index = faiss_indices[feat_id]
            emb_all = var_embeddings[feat_id]
        else:
            assert embeddings_dir is not None
            emb_path = embeddings_dir / f"feat_{feat_id:04d}.npy"
            if not emb_path.exists():
                raise FileNotFoundError(
                    f"Missing retrieval embeddings for feat_id={feat_id} var={var}: {emb_path}. "
                    f"Please (re)build retrieval_database cache for {dataset_name} seq_len={args.seq_len}."
                )
            emb_all = np.load(emb_path, mmap_mode="r")
            if emb_all.ndim != 2:
                raise ValueError(f"Expected embeddings to be 2D (n_windows,d), got {emb_all.shape} for var={var}")
            if int(emb_all.shape[0]) != int(n_windows):
                raise ValueError(
                    f"Retrieval embeddings length mismatch for {emb_path}: "
                    f"emb_windows={emb_all.shape[0]} != n_windows={n_windows} (seq_len={args.seq_len})"
                )
            if d_infer is None:
                d_infer = int(emb_all.shape[1])
                if "retrieval_embed_dim_base" not in meta:
                    meta["retrieval_embed_dim_base"] = int(d_infer)
                    meta["retrieval_embed_extra_dim"] = int(extra_dim)
                    meta["retrieval_embed_dim"] = int(d_infer) + int(extra_dim)
            elif int(emb_all.shape[1]) != int(d_infer):
                raise ValueError(f"Embedding dimension mismatch across features: feat_id={feat_id} has {emb_all.shape[1]} != {d_infer}")

            d_index = int(d_infer) + int(extra_dim)
            index = faiss.IndexFlatL2(int(d_index))
            emb_kb = np.asarray(emb_all[:kb_slice_end], dtype=np.float32)
            extra_parts = []
            if use_time_feats:
                assert time_feats_start is not None
                extra_parts.append(np.asarray(time_feats_start[:kb_slice_end], dtype=np.float32))
            if use_stats_feats:
                assert stats_mean_start is not None and stats_logstd_start is not None and stats_slope_start is not None
                assert stats_mu is not None and stats_sigma is not None
                s = np.stack(
                    [
                        stats_mean_start[:kb_slice_end, feat_id],
                        stats_logstd_start[:kb_slice_end, feat_id],
                        stats_slope_start[:kb_slice_end, feat_id],
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
                mu = stats_mu[feat_id]
                sig = stats_sigma[feat_id]
                s = ((s - mu) / sig) * float(args.retrieval_embed_stats_weight)
                extra_parts.append(s.astype(np.float32, copy=False))
            if extra_parts:
                emb_kb = np.concatenate([emb_kb, *extra_parts], axis=1).astype(np.float32, copy=False)
            if args.distance_metric == "cosine":
                emb_kb = emb_kb.copy()
                faiss.normalize_L2(emb_kb)
            index.add(emb_kb)

        # Batch over time windows.
        for start_i in range(0, len(query_starts), args.batch_size):
            starts = query_starts[start_i : start_i + args.batch_size]
            # Raw context/target (for embedding + later scaling).
            ctx_raw = torch.from_numpy(
                np.stack([raw[s : s + args.seq_len, feat_id] for s in starts], axis=0)
            ).to(device=device, dtype=torch.float32)
            tgt_raw = torch.from_numpy(
                np.stack([raw[s + args.seq_len : s + args.seq_len + pred_len, feat_id] for s in starts], axis=0)
            ).to(device=device, dtype=torch.float32)

            if args.query_embedding_source == "cache":
                query_vec = np.asarray(emb_all[starts], dtype=np.float32)
            else:
                # Fallback: compute query embeddings with ChronosPipeline.embed.
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                from chronos import ChronosPipeline

                embedding_model = ChronosPipeline.from_pretrained(
                    args.embedding_model_path,
                    device_map=str(device),
                    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                )
                query_vec, _ = embedding_model.embed(ctx_raw)
                query_vec = query_vec[:, -1, :].float().cpu().numpy()

            extra_parts = []
            if use_time_feats:
                assert time_feats_start is not None
                extra_parts.append(np.asarray(time_feats_start[starts], dtype=np.float32))
            if use_stats_feats:
                assert stats_mean_start is not None and stats_logstd_start is not None and stats_slope_start is not None
                assert stats_mu is not None and stats_sigma is not None
                s = np.stack(
                    [
                        stats_mean_start[starts, feat_id],
                        stats_logstd_start[starts, feat_id],
                        stats_slope_start[starts, feat_id],
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
                mu = stats_mu[feat_id]
                sig = stats_sigma[feat_id]
                s = ((s - mu) / sig) * float(args.retrieval_embed_stats_weight)
                extra_parts.append(s.astype(np.float32, copy=False))
            if extra_parts:
                query_vec = np.concatenate([query_vec, *extra_parts], axis=1).astype(np.float32, copy=False)

            query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
            if args.distance_metric == "cosine":
                faiss.normalize_L2(query_vec)

            # Search k+1 then drop self and keep top-k.
            distances_raw, indices_raw = index.search(query_vec, args.k + 1)
            distances_np = distances_raw.astype(np.float64, copy=True)
            indices_np = indices_raw.astype(np.int64, copy=False)

            if args.drop_self_mode == "idx":
                # Only applies to train split queries that are inside KB window range.
                in_kb = starts < kb_slice_end
                if args.split == "train" and in_kb.any():
                    # Index is built from emb_all[:kb_slice_end], so local idx == global window start (offset=0).
                    self_idx = starts
                    mask = (indices_np == self_idx[:, None]) & in_kb[:, None]
                    dropped_self += int(mask.any(axis=1).sum())
                    self_not_found += int((in_kb & (~mask.any(axis=1))).sum())
                    distances_np[mask] = np.inf
            else:
                # Backward compatible behavior: filter exact/near-exact matches by distance.
                distances_np[distances_np <= float(args.drop_self_eps)] = np.inf

            order = np.argsort(distances_np, axis=1)[:, : args.k]
            distances_np = np.take_along_axis(distances_np, order, axis=1).astype(np.float32)
            indices_np = np.take_along_axis(indices_np, order, axis=1).astype(np.int64)
            # Drop only rows with no valid neighbors after filtering; keep partial inf rows.
            row_all_inf = np.isinf(distances_np).all(axis=1)
            if row_all_inf.any():
                skipped_all_inf += int(row_all_inf.sum())
                keep = ~row_all_inf
                distances_np = distances_np[keep]
                indices_np = indices_np[keep]
                keep_t = torch.from_numpy(keep).to(device=device)
                ctx_raw = ctx_raw[keep_t]
                tgt_raw = tgt_raw[keep_t]
                if distances_np.shape[0] == 0:
                    continue

            total_queries += int(distances_np.shape[0])
            distances = torch.from_numpy(distances_np).to(device=device, dtype=torch.float32)
            timestamp_idx = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)

            # Transform context/target to dataset scale (matches q_base output scale).
            ctx = (ctx_raw - mean) / scale
            tgt = (tgt_raw - mean) / scale

            shift = None
            raw_dist = None
            shift_modes = {"shift_last", "shift_mean_last_m"}
            need_ctx = (
                args.teacher_align in shift_modes
                or args.rerank_mode != "none"
                or args.rerank_weight_source == "raw"
            )
            if need_ctx:
                offsets_ctx = torch.arange(0, args.seq_len, device=device, dtype=torch.long)
                idx_ctx = timestamp_idx.unsqueeze(-1) + offsets_ctx.view(1, 1, -1)  # (B,K0,L)
                retrieved_ctx_raw = train_raw[idx_ctx]  # (B,K0,L) RAW
                retrieved_ctx = (retrieved_ctx_raw - mean) / scale  # dataset scale

                shift_mode = (
                    args.teacher_align
                    if args.teacher_align in shift_modes
                    else "shift_mean_last_m"
                )
                shift = _compute_level_shift_univariate(
                    ctx, retrieved_ctx, mode=str(shift_mode), last_m=int(args.shift_last_m)
                )  # (B,K0)

                if args.rerank_mode != "none" or args.rerank_weight_source == "raw":
                    retrieved_ctx_shifted = retrieved_ctx + shift.unsqueeze(-1)  # (B,K0,L)
                    diff_ctx = ctx.unsqueeze(1) - retrieved_ctx_shifted
                    if args.rerank_mode == "raw_l2_shift":
                        raw_dist = (diff_ctx * diff_ctx).mean(dim=2)
                    else:
                        raw_dist = diff_ctx.abs().mean(dim=2)
                    raw_dist = torch.where(
                        torch.isfinite(distances),
                        raw_dist,
                        torch.full_like(raw_dist, float("inf")),
                    )

                if args.rerank_mode != "none":
                    sel = torch.topk(raw_dist, k=int(args.k), dim=1, largest=False).indices  # (B,K)
                    timestamp_idx = torch.gather(timestamp_idx, 1, sel)
                    distances = torch.gather(distances, 1, sel)
                    shift = torch.gather(shift, 1, sel)
                    raw_dist = torch.gather(raw_dist, 1, sel)

            # Gather retrieved horizons y_i (RAW scale) and transform to dataset scale.
            offsets_y = torch.arange(args.seq_len, args.seq_len + pred_len, device=device, dtype=torch.long)
            idx_y = timestamp_idx.unsqueeze(-1) + offsets_y.view(1, 1, -1)  # (B,K,T)
            retrieved_y_raw = train_raw[idx_y]  # (B,K,T) RAW
            retrieved_y = (retrieved_y_raw - mean) / scale  # dataset scale

            if args.teacher_mode != "rag_output" and args.teacher_align in shift_modes:
                assert shift is not None
                retrieved_y = retrieved_y + shift.unsqueeze(-1)

            # Build weights + confidence (used by dynamic-beta training).
            if args.rerank_mode != "none" and args.rerank_weight_source == "raw":
                assert raw_dist is not None
                distances_for_w = raw_dist
                tau_w = float(args.rerank_tau)
            else:
                distances_for_w = distances
                tau_w = float(args.tau)

            if args.dist_transform == "sqrt":
                distances_for_w = torch.sqrt(torch.clamp(distances_for_w, min=0.0))

            logits = -distances_for_w / tau_w
            logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -float("inf")))
            w = torch.softmax(logits, dim=1)

            bad = torch.isnan(w).any(dim=1)
            if bad.any():
                skipped_all_inf += int(bad.sum().item())
                keep = ~bad
                if keep.sum() == 0:
                    continue
                retrieved_y = retrieved_y[keep]
                ctx = ctx[keep]
                tgt = tgt[keep]
                w = w[keep]
                timestamp_idx = timestamp_idx[keep]
                distances = distances[keep]
                if shift is not None:
                    shift = shift[keep]
                if raw_dist is not None:
                    raw_dist = raw_dist[keep]

            conf_w_max = w.max(dim=1).values  # (B,)
            conf_entropy = -(w * torch.log(w + 1e-12)).sum(dim=1)  # (B,), in nats
            conf_effective_k = 1.0 / (w.square().sum(dim=1) + 1e-12)  # (B,)

            if args.teacher_mode == "rag_output":
                assert rag_model is not None
                offsets_full = torch.arange(0, args.seq_len + pred_len, device=device, dtype=torch.long)
                idx_full = timestamp_idx.unsqueeze(-1) + offsets_full.view(1, 1, -1)
                retrieved_seq_raw = train_raw[idx_full]  # (B, K, seq_len+pred_len) RAW
                retrieved_seq = (retrieved_seq_raw - mean) / scale  # dataset scale
                with torch.no_grad():
                    q_teacher = rag_model(
                        context=ctx,
                        retrieved_seq=retrieved_seq,
                        distances=distances,
                    ).quantile_preds.to(dtype=torch.float32)
            else:
                if args.teacher_align in {"none", "shift_last", "shift_mean_last_m"}:
                    q_teacher = weighted_quantile_torch(retrieved_y, w, quantiles_t)  # (B, Q, pred_len)
                elif args.teacher_align == "retrieved_to_query":
                    # Align retrieved horizons by transferring normalized shapes from each retrieved sequence
                    # to the query window scale (InstanceNorm per retrieved seq, inverse using query loc/scale).
                    loc_q, scale_q, _is_const_q = _instance_norm_stats(ctx)  # (B,1)
                    loc_q = loc_q.unsqueeze(1)  # (B,1,1) for broadcasting
                    scale_q = scale_q.unsqueeze(1)  # (B,1,1)

                    offsets_full = torch.arange(0, args.seq_len + pred_len, device=device, dtype=torch.long)
                    idx_full = timestamp_idx.unsqueeze(-1) + offsets_full.view(1, 1, -1)
                    retrieved_seq_raw = train_raw[idx_full]  # (B, K, seq_len+pred_len) RAW
                    retrieved_seq = (retrieved_seq_raw - mean) / scale  # dataset scale

                    if args.retrieved_loc_scale == "context":
                        retrieved_for_stats = retrieved_seq[:, :, : args.seq_len]
                    else:
                        retrieved_for_stats = retrieved_seq

                    loc_r, scale_r, is_const_r = _instance_norm_stats(retrieved_for_stats)  # (B,K,1)
                    retrieved_y_norm = _instance_norm_apply(retrieved_y, loc_r, scale_r, is_const_r)  # (B,K,pred_len)
                    retrieved_y_aligned = _instance_norm_inverse(retrieved_y_norm, loc_q, scale_q)  # (B,K,pred_len)

                    q_teacher = weighted_quantile_torch(retrieved_y_aligned, w, quantiles_t)  # (B, Q, pred_len)
                else:
                    raise ValueError(f"Unsupported teacher_align={args.teacher_align}")

            q50_pred = q_teacher[:, q50_idx, :]
            diff = (q50_pred - tgt).to(dtype=torch.float32)
            teacher_q50_abs_sum += float(diff.abs().sum().item())
            teacher_q50_sq_sum += float((diff * diff).sum().item())
            teacher_q50_count += int(diff.numel())

            shard_context.append(ctx.detach().cpu())
            shard_target.append(tgt.detach().cpu())
            shard_q_teacher.append(q_teacher.detach().cpu())
            shard_conf_w_max.append(conf_w_max.detach().cpu())
            shard_conf_entropy.append(conf_entropy.detach().cpu())
            shard_conf_effective_k.append(conf_effective_k.detach().cpu())

            # Flush shard.
            cur_n = sum(t.shape[0] for t in shard_context)
            if cur_n >= args.shard_size:
                context_cat = torch.cat(shard_context, dim=0)
                target_cat = torch.cat(shard_target, dim=0)
                q_teacher_cat = torch.cat(shard_q_teacher, dim=0)
                conf_w_max_cat = torch.cat(shard_conf_w_max, dim=0)
                conf_entropy_cat = torch.cat(shard_conf_entropy, dim=0)
                conf_effective_k_cat = torch.cat(shard_conf_effective_k, dim=0)
                shard_name = _save_shard(
                    out_dir,
                    args.split,
                    shard_idx,
                    context_cat,
                    target_cat,
                    q_teacher_cat,
                    meta,
                    conf_w_max=conf_w_max_cat,
                    conf_entropy=conf_entropy_cat,
                    conf_effective_k=conf_effective_k_cat,
                )
                written.append({"file": shard_name, "num_samples": int(context_cat.shape[0])})
                shard_idx += 1
                shard_context, shard_target, shard_q_teacher = [], [], []
                shard_conf_w_max, shard_conf_entropy, shard_conf_effective_k = [], [], []

    # Flush remaining.
    if shard_context:
        context_cat = torch.cat(shard_context, dim=0)
        target_cat = torch.cat(shard_target, dim=0)
        q_teacher_cat = torch.cat(shard_q_teacher, dim=0)
        conf_w_max_cat = torch.cat(shard_conf_w_max, dim=0)
        conf_entropy_cat = torch.cat(shard_conf_entropy, dim=0)
        conf_effective_k_cat = torch.cat(shard_conf_effective_k, dim=0)
        shard_name = _save_shard(
            out_dir,
            args.split,
            shard_idx,
            context_cat,
            target_cat,
            q_teacher_cat,
            meta,
            conf_w_max=conf_w_max_cat,
            conf_entropy=conf_entropy_cat,
            conf_effective_k=conf_effective_k_cat,
        )
        written.append({"file": shard_name, "num_samples": int(context_cat.shape[0])})

    teacher_q50_mae = float(teacher_q50_abs_sum / max(1, teacher_q50_count))
    teacher_q50_mse = float(teacher_q50_sq_sum / max(1, teacher_q50_count))
    manifest = {
        "meta": meta,
        "shards": written,
        "total_queries": total_queries,
        "dropped_self": dropped_self,
        "self_not_found": self_not_found,
        "skipped_all_inf": skipped_all_inf,
        "teacher_q50_mae": teacher_q50_mae,
        "teacher_q50_mse": teacher_q50_mse,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] wrote {len(written)} shard(s) to {out_dir / args.split}")
    if args.drop_self_mode == "idx" and args.split == "train":
        print(
            f"[drop_self_idx] dropped_self={dropped_self} self_not_found={self_not_found} total_queries={total_queries}"
        )
        if self_not_found:
            print(
                "[warn] self_not_found>0: retrieval DB / window indexing may be misaligned; "
                "fall back to drop_self_mode=dist if unsure."
            )
    if skipped_all_inf:
        print(f"[warn] skipped {skipped_all_inf} samples with no valid neighbors after drop_self")


if __name__ == "__main__":
    main()
