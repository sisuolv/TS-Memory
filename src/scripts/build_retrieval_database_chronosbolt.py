#!/usr/bin/env python
"""
Build retrieval_database embeddings cache **offline** using the local Chronos-Bolt backbone.

This produces a directory cache:
  <retrieval_database_dir>/<dataset>_<frequency>_<seq_len>/
    - manifest.json
    - embeddings/feat_XXXX.npy   # (num_windows, d_model)

Each feature id corresponds to the CSV column order excluding the first 'date' column:
  feat_id == i  <=>  var_names[i] == df.columns[1 + i]

This format avoids huge monolithic pickles for large multivariate datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from numpy.lib.format import open_memmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ChronosBolt import ChronosBoltModelForForecasting, ChronosBoltPipeline


def _guess_frequency(dataset_name: str) -> str:
    name = dataset_name.lower()
    if name.startswith("etth"):
        return "hour"
    if name.startswith("ettm"):
        return "minute"
    if name == "weather":
        return "10minutes"
    if name in {"electricity", "traffic"}:
        return "hour"
    if name in {"exchange_rate", "exchange-rate"}:
        return "hour"
    return "custom"


@torch.no_grad()
def _embed_chronosbolt_last_token(model: ChronosBoltModelForForecasting, context: torch.Tensor) -> torch.Tensor:
    """
    Return embeddings analogous to `ChronosPipeline.embed(...)[..., -1, :]`.

    We take the encoder last token hidden state (which corresponds to [REG] when enabled).

    Args:
        context: float tensor (B, L)
    Returns:
        embeddings: float32 tensor (B, d_model)
    """
    if context.ndim != 2:
        raise ValueError(f"context must be (B,L), got {tuple(context.shape)}")

    mask = torch.isnan(context).logical_not().to(context.dtype)
    batch_size = int(context.shape[0])

    # scaling in 32-bit precision
    context, _loc_scale = model.instance_norm(context)
    context = context.to(model.dtype)
    mask = mask.to(model.dtype)

    patched_context = model.patch(context)
    patched_mask = torch.nan_to_num(model.patch(mask), nan=0.0)
    patched_context[~(patched_mask > 0)] = 0.0
    patched_context = torch.cat([patched_context, patched_mask], dim=-1)

    attention_mask = patched_mask.sum(dim=-1) > 0
    input_embeds = model.input_patch_embedding(patched_context)

    if model.chronos_config.use_reg_token:
        reg_input_ids = torch.full((batch_size, 1), model.config.reg_token_id, device=input_embeds.device)
        reg_embeds = model.shared(reg_input_ids)
        input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, torch.ones_like(reg_input_ids, dtype=torch.bool)], dim=-1)

    encoder_outputs = model.encoder(attention_mask=attention_mask, inputs_embeds=input_embeds)
    hidden_states = encoder_outputs[0]
    return hidden_states[:, -1, :].float()


def _atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser("Build retrieval_database cache with Chronos-Bolt (offline)")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--base_model_path", type=str, default="./checkpoints/base")
    parser.add_argument("--retrieval_database_dir", type=str, default="../retrieval_database")
    parser.add_argument("--frequency", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--feature_shard_idx", type=int, default=0)
    parser.add_argument("--feature_shard_total", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_name = Path(args.data_path).stem
    frequency = args.frequency or _guess_frequency(dataset_name)

    csv_path = Path(args.root_path) / args.data_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load Chronos-Bolt model (offline, local path).
    # Use the same override weights as zeroshot.py to stay consistent.
    pipe = ChronosBoltPipeline.from_pretrained(args.base_model_path)
    autogluon_path = Path(args.base_model_path) / "autogluon_model.pth"
    if autogluon_path.exists():
        state = torch.load(autogluon_path, map_location="cpu")
        pipe.model.load_state_dict(state, strict=True)
    else:
        # Still usable if model.safetensors is present, but warn loudly.
        print(f"[warn] autogluon_model.pth not found at {autogluon_path}; using default from_pretrained weights only.")
    model: ChronosBoltModelForForecasting = pipe.model
    model.to(device)
    model.eval()

    d_model = int(getattr(model.config, "d_model", 768))
    dtype_np = np.float16 if args.dtype == "float16" else np.float32

    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns (date + variables), got columns={df.columns.tolist()}")
    var_names: List[str] = list(df.columns[1:])

    total_len = int(len(df))
    num_windows = total_len - int(args.seq_len) + 1
    if num_windows <= 0:
        raise ValueError(f"Series too short: len={total_len} < seq_len={args.seq_len}")

    out_dir = Path(args.retrieval_database_dir) / f"{dataset_name}_{frequency}_{args.seq_len}"
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Write/refresh manifest (safe even with shards; content is deterministic).
    manifest = {
        "format": "retrieval_db_dir_v1",
        "dataset": dataset_name,
        "root_path": str(Path(args.root_path).resolve()),
        "data_path": args.data_path,
        "frequency": frequency,
        "seq_len": int(args.seq_len),
        "num_windows": num_windows,
        "num_features": len(var_names),
        "embedding_dim": d_model,
        "dtype": args.dtype,
        "feature_file_pattern": "embeddings/feat_{feat_id:04d}.npy",
        "feature_shard_total": int(args.feature_shard_total),
    }
    _atomic_write_json(out_dir / "manifest.json", manifest)

    shard_idx = int(args.feature_shard_idx)
    shard_total = int(args.feature_shard_total)
    if shard_total <= 0:
        raise ValueError("--feature_shard_total must be >= 1")
    if not (0 <= shard_idx < shard_total):
        raise ValueError("--feature_shard_idx must be in [0, feature_shard_total)")

    feat_ids = [i for i in range(len(var_names)) if (i % shard_total) == shard_idx]
    print(
        f"[info] dataset={dataset_name} freq={frequency} seq_len={args.seq_len} windows={num_windows} "
        f"features={len(var_names)} shard={shard_idx}/{shard_total} (num_feat_ids={len(feat_ids)})"
    )

    for feat_id in feat_ids:
        var = var_names[feat_id]
        out_path = emb_dir / f"feat_{feat_id:04d}.npy"

        if out_path.exists() and not args.overwrite:
            try:
                existing = np.load(out_path, mmap_mode="r")
                if tuple(existing.shape) == (num_windows, d_model):
                    print(f"[skip] feat_id={feat_id} var={var} exists: {out_path}")
                    continue
            except Exception:
                pass
            print(f"[warn] feat_id={feat_id} var={var} has incompatible existing file; overwriting: {out_path}")

        series = df[var].to_numpy(dtype=np.float32, copy=True)
        series_t = torch.from_numpy(series)  # CPU
        windows = series_t.unfold(0, int(args.seq_len), 1)  # (num_windows, seq_len)

        mm = open_memmap(out_path, mode="w+", dtype=dtype_np, shape=(num_windows, d_model))

        for start in range(0, num_windows, int(args.batch_size)):
            end = min(num_windows, start + int(args.batch_size))
            ctx = windows[start:end].contiguous().to(device=device, dtype=torch.float32)
            emb = _embed_chronosbolt_last_token(model, ctx)  # (B, d_model)
            mm[start:end] = emb.detach().cpu().numpy().astype(dtype_np, copy=False)

        # Ensure data is flushed to disk.
        del mm

        print(f"[done] feat_id={feat_id} var={var} -> {out_path}")

    print(f"[done] retrieval_database cache written to: {out_dir}")


if __name__ == "__main__":
    main()
