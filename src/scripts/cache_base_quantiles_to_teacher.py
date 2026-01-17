#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_manifest(split_dir: Path) -> Dict[str, Any]:
    manifest_path = split_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _save_q_base_cache(
    cache_path: Path,
    *,
    q_base: torch.Tensor,
    pred_len: int,
    quantiles: List[float],
    base_model_path: str,
    shard_file: str,
    dtype_tag: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    payload = {
        "q_base": q_base,
        "pred_len": int(pred_len),
        "quantiles": list(quantiles),
        "base_model_path": str(base_model_path),
        "dtype": dtype_tag,
        "shard_file": str(shard_file),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "q_base_cache_v1",
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)


def _compute_q_base_for_context(
    *,
    base_pipe: Any,
    context_cpu: torch.Tensor,
    pred_len: int,
    batch_size: int,
    device: torch.device,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    n = int(context_cpu.shape[0])
    with torch.no_grad():
        probe = base_pipe.predict(context=context_cpu[:1].to(device=device, dtype=torch.float32), prediction_length=int(pred_len))
        if probe.ndim != 3:
            raise ValueError(f"Expected base predictions (B,Q,T), got {tuple(probe.shape)}")
        Q = int(probe.shape[1])
        T = int(probe.shape[2])
        if T != int(pred_len):
            raise ValueError(f"Base predict returned T={T}, expected pred_len={pred_len}")
        out = torch.empty((n, Q, T), dtype=out_dtype, device="cpu")

        for start in range(0, n, int(batch_size)):
            end = min(n, start + int(batch_size))
            ctx = context_cpu[start:end].to(device=device, dtype=torch.float32)
            q = base_pipe.predict(context=ctx, prediction_length=int(pred_len))
            if q.shape[1] != Q or q.shape[2] != T:
                raise ValueError(f"Unexpected base predict shape {tuple(q.shape)} (expected B,{Q},{T})")
            q = q.detach().to(device="cpu")
            if out_dtype != q.dtype:
                q = q.to(dtype=out_dtype)
            out[start:end] = q
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache base model quantile predictions (q_base) for teacher shards.")
    ap.add_argument(
        "--teacher_split_dir", type=str, required=True, help="Teacher split dir containing manifest.json + shard files."
    )
    ap.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="ChronosBolt base checkpoint dir (config.json + autogluon_model.pth).",
    )
    ap.add_argument("--pred_len", type=int, required=True, help="Prediction length used for q_base cache.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--force", type=int, default=0, help="Recompute even if cache exists.")
    ap.add_argument("--resume", type=int, default=1, help="Skip shards with existing cache file.")
    ap.add_argument("--device", type=str, default=None, help="Device string (default: cuda if available else cpu).")
    args = ap.parse_args()

    split_dir = Path(args.teacher_split_dir)
    if not (split_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Missing manifest: {split_dir}/manifest.json")
    if not Path(args.base_model_path).exists():
        raise FileNotFoundError(f"Missing base_model_path: {args.base_model_path}")
    if int(args.pred_len) <= 0:
        raise ValueError("--pred_len must be > 0")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    manifest = _load_manifest(split_dir)
    meta = manifest.get("meta", {})
    quantiles = meta.get("quantiles", None)
    if not isinstance(quantiles, list) or not quantiles:
        raise ValueError("Teacher manifest meta.quantiles must be a non-empty list")

    from models.ChronosBolt import ChronosBoltPipeline

    print(f"[cache_q_base] split_dir={split_dir}")
    print(f"[cache_q_base] base_model_path={args.base_model_path}")
    print(f"[cache_q_base] pred_len={args.pred_len} batch_size={args.batch_size} dtype={args.dtype} device={device}")

    base_pipe = ChronosBoltPipeline.from_pretrained(args.base_model_path)
    base_pipe.model.load_state_dict(torch.load(os.path.join(args.base_model_path, "autogluon_model.pth"), map_location="cpu"))
    base_pipe.model.to(device)
    base_pipe.model.eval()
    for p in base_pipe.model.parameters():
        p.requires_grad_(False)

    cache_dir = split_dir / "q_base_cache" / f"pl{int(args.pred_len)}_{args.dtype}"

    shards: List[Dict[str, Any]] = manifest.get("shards", [])
    if not shards:
        raise ValueError(f"Empty manifest shards: {split_dir}/manifest.json")

    n_total = 0
    n_cached = 0
    n_skipped = 0
    t0 = time.time()

    for shard_entry in shards:
        shard_file = shard_entry.get("file", None)
        if not shard_file:
            raise ValueError("Each shard entry must contain 'file'")
        shard_path = split_dir / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file: {shard_path}")

        cache_name = f"{Path(shard_file).stem}_qbase.pt"
        cache_path = cache_dir / cache_name
        cache_rel = cache_path.relative_to(split_dir).as_posix()

        if int(args.force) == 0 and int(args.resume) == 1 and cache_path.exists():
            shard_entry["q_base_file"] = cache_rel
            n_skipped += 1
            continue

        shard = torch.load(shard_path, map_location="cpu")
        if "context" not in shard:
            raise KeyError(f"Missing 'context' in shard: {shard_path}")
        context = shard["context"]
        if not isinstance(context, torch.Tensor):
            raise TypeError(f"Shard context must be torch.Tensor, got {type(context)} in {shard_path}")

        q_base = _compute_q_base_for_context(
            base_pipe=base_pipe,
            context_cpu=context,
            pred_len=int(args.pred_len),
            batch_size=int(args.batch_size),
            device=device,
            out_dtype=out_dtype,
        )
        n = int(context.shape[0])
        n_total += n

        _save_q_base_cache(
            cache_path,
            q_base=q_base,
            pred_len=int(args.pred_len),
            quantiles=quantiles,
            base_model_path=str(args.base_model_path),
            shard_file=str(shard_file),
            dtype_tag=args.dtype,
        )
        shard_entry["q_base_file"] = cache_rel
        n_cached += 1

        if n_cached % 5 == 0:
            dt = time.time() - t0
            print(f"[cache_q_base] cached_shards={n_cached}/{len(shards)} samples={n_total} elapsed={dt/60.0:.1f}m")

    manifest["q_base_cache"] = {
        "pred_len": int(args.pred_len),
        "dtype": args.dtype,
        "base_model_path": str(args.base_model_path),
        "cache_dir": cache_dir.relative_to(split_dir).as_posix(),
        "format": "q_base_cache_v1",
    }
    _atomic_write_json(split_dir / "manifest.json", manifest)

    dt = time.time() - t0
    print(
        f"[cache_q_base] done split={meta.get('split','?')} shards={len(shards)} "
        f"cached={n_cached} skipped={n_skipped} total_samples={n_total} elapsed={dt/60.0:.1f}m"
    )


if __name__ == "__main__":
    main()

