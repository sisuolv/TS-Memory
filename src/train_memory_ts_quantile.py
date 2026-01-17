#!/usr/bin/env python
import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig

from models.memory_ts_quantile import MemoryTSQuantile, MemoryTSQuantileConfig
from models.memory_ts_quantile_multivar import (
    MemoryTSQuantileMultivar,
    MemoryTSQuantileMultivarConfig,
)
from models.memory_ts_quantile_hidden import (
    MemoryTSHiddenDelta,
    MemoryTSHiddenDeltaConfig,
    instance_norm_stats,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_manifest(split_dir: str) -> Dict:
    manifest_path = Path(split_dir) / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TeacherShardIterableDataset(IterableDataset):
    def __init__(self, split_dir: str, shuffle: bool, seed: int):
        super().__init__()
        self.split_dir = Path(split_dir)
        self.shuffle = shuffle
        self.seed = seed
        self.manifest = load_manifest(split_dir)
        self.shards = list(self.manifest["shards"])

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + int(worker.id) if worker else self.seed)
        shard_entries = list(self.shards)
        if self.shuffle:
            rng.shuffle(shard_entries)

        for shard_entry in shard_entries:
            shard_file = shard_entry["file"]
            shard_path = self.split_dir / shard_file
            q_base_path = None
            if shard_entry.get("q_base_file"):
                q_base_path = self.split_dir / shard_entry["q_base_file"]

            data = torch.load(shard_path, map_location="cpu")
            context = data["context"].float()
            target = data["target"].float()
            q_teacher = data["q_teacher"].float()
            q_base = data.get("q_base", None)
            if q_base is None and q_base_path is not None:
                if not q_base_path.exists():
                    raise FileNotFoundError(f"q_base_file missing: {q_base_path} (shard={shard_file})")
                q_obj = torch.load(q_base_path, map_location="cpu")
                if isinstance(q_obj, torch.Tensor):
                    q_base = q_obj
                elif isinstance(q_obj, dict) and "q_base" in q_obj:
                    q_base = q_obj["q_base"]
                else:
                    raise TypeError(f"Unexpected q_base cache type={type(q_obj)} at {q_base_path}")
            conf_w_max = data.get("conf_w_max", None)
            conf_entropy = data.get("conf_entropy", None)
            conf_effective_k = data.get("conf_effective_k", None)
            if conf_w_max is not None:
                conf_w_max = conf_w_max.float()
            if conf_entropy is not None:
                conf_entropy = conf_entropy.float()
            if conf_effective_k is not None:
                conf_effective_k = conf_effective_k.float()
            n = context.shape[0]
            idx = list(range(n))
            if self.shuffle:
                rng.shuffle(idx)
            for i in idx:
                sample = {
                    "context": context[i],
                    "target": target[i],
                    "q_teacher": q_teacher[i],
                }
                if q_base is not None:
                    sample["q_base"] = q_base[i]
                if conf_w_max is not None:
                    sample["conf_w_max"] = conf_w_max[i]
                if conf_entropy is not None:
                    sample["conf_entropy"] = conf_entropy[i]
                if conf_effective_k is not None:
                    sample["conf_effective_k"] = conf_effective_k[i]
                yield sample


def pinball_loss(q_pred: torch.Tensor, y_true: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    q_pred: (B, Q, T)
    y_true: (B, T)
    quantiles: (Q,)
    """
    if q_pred.ndim != 3:
        raise ValueError(f"q_pred must be (B,Q,T), got {q_pred.shape}")
    if y_true.ndim != 2:
        raise ValueError(f"y_true must be (B,T), got {y_true.shape}")
    if quantiles.ndim != 1:
        raise ValueError(f"quantiles must be (Q,), got {quantiles.shape}")

    y = y_true.unsqueeze(1)  # (B,1,T)
    q = quantiles.view(1, -1, 1)  # (1,Q,1)
    diff = y - q_pred
    loss = torch.maximum(q * diff, (q - 1) * diff)  # (B,Q,T)
    return loss.mean()


def pinball_loss_per_sample(q_pred: torch.Tensor, y_true: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    Per-sample pinball loss.

    q_pred: (B, Q, T) or (B, Q, T, C)
    y_true: (B, T) or (B, T, C)
    quantiles: (Q,)

    Returns:
        (B,)
    """
    if q_pred.ndim == 3:
        if y_true.ndim != 2:
            raise ValueError(f"y_true must be (B,T) for q_pred (B,Q,T), got {y_true.shape}")
        y = y_true.unsqueeze(1)  # (B,1,T)
        q = quantiles.view(1, -1, 1)  # (1,Q,1)
    elif q_pred.ndim == 4:
        if y_true.ndim != 3:
            raise ValueError(f"y_true must be (B,T,C) for q_pred (B,Q,T,C), got {y_true.shape}")
        y = y_true.unsqueeze(1)  # (B,1,T,C)
        q = quantiles.view(1, -1, 1, 1)  # (1,Q,1,1)
    else:
        raise ValueError(f"q_pred must be 3D or 4D, got {q_pred.shape}")
    if quantiles.ndim != 1:
        raise ValueError(f"quantiles must be (Q,), got {quantiles.shape}")

    diff = y - q_pred
    loss = torch.maximum(q * diff, (q - 1) * diff)
    return loss.reshape(loss.shape[0], -1).mean(dim=1)


def weighted_pinball_loss(
    q_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: torch.Tensor,
    quantile_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Weighted pinball loss (for emphasizing central quantiles / MAE).

    q_pred: (B, Q, T)
    y_true: (B, T)
    quantiles: (Q,)
    quantile_weights: (Q,) or None (uniform)
    """
    if quantile_weights is None:
        return pinball_loss(q_pred, y_true, quantiles)

    if quantile_weights.ndim != 1 or quantile_weights.shape[0] != quantiles.shape[0]:
        raise ValueError(
            f"quantile_weights must be (Q,), got {quantile_weights.shape} for Q={quantiles.shape[0]}"
        )

    y = y_true.unsqueeze(1)  # (B,1,T)
    q = quantiles.view(1, -1, 1)  # (1,Q,1)
    diff = y - q_pred
    loss = torch.maximum(q * diff, (q - 1) * diff)  # (B,Q,T)
    w = quantile_weights.view(1, -1, 1).to(dtype=loss.dtype, device=loss.device)
    loss = (loss * w).sum(dim=1) / w.sum()  # (B,T)
    return loss.mean()


def weighted_pinball_loss_per_sample(
    q_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: torch.Tensor,
    quantile_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Per-sample weighted pinball loss.

    q_pred: (B, Q, T) or (B, Q, T, C)
    y_true: (B, T) or (B, T, C)
    quantiles: (Q,)
    quantile_weights: (Q,) or None

    Returns:
        (B,)
    """
    if quantile_weights is None:
        return pinball_loss_per_sample(q_pred, y_true, quantiles)
    if quantile_weights.ndim != 1 or quantile_weights.shape[0] != quantiles.shape[0]:
        raise ValueError(
            f"quantile_weights must be (Q,), got {quantile_weights.shape} for Q={quantiles.shape[0]}"
        )

    if q_pred.ndim == 3:
        y = y_true.unsqueeze(1)  # (B,1,T)
        q = quantiles.view(1, -1, 1)  # (1,Q,1)
        w = quantile_weights.view(1, -1, 1).to(dtype=q_pred.dtype, device=q_pred.device)
    elif q_pred.ndim == 4:
        y = y_true.unsqueeze(1)  # (B,1,T,C)
        q = quantiles.view(1, -1, 1, 1)  # (1,Q,1,1)
        w = quantile_weights.view(1, -1, 1, 1).to(dtype=q_pred.dtype, device=q_pred.device)
    else:
        raise ValueError(f"q_pred must be 3D or 4D, got {q_pred.shape}")

    diff = y - q_pred
    loss = torch.maximum(q * diff, (q - 1) * diff)  # (B,Q,...) 
    loss = (loss * w).sum(dim=1) / (w.sum() + 1e-12)  # (B,...) without Q
    return loss.reshape(loss.shape[0], -1).mean(dim=1)


def crossing_penalty(q_pred: torch.Tensor) -> torch.Tensor:
    """
    Penalize quantile crossing: q_{i+1} >= q_i
    q_pred: (B, Q, T) or (B, Q, T, C)
    """
    if q_pred.ndim not in (3, 4):
        raise ValueError(f"q_pred must be 3D or 4D, got {q_pred.shape}")
    diff = q_pred[:, 1:, ...] - q_pred[:, :-1, ...]
    return F.relu(-diff).mean()


def _slice_target_time(y: torch.Tensor, t: int) -> torch.Tensor:
    if y.ndim == 2:
        return y[:, :t]
    if y.ndim == 3:
        return y[:, :t, :]
    raise ValueError(f"target must be (B,T) or (B,T,C), got {tuple(y.shape)}")


def _slice_quantiles_time(q: torch.Tensor, t: int) -> torch.Tensor:
    if q.ndim == 3:
        return q[:, :, :t]
    if q.ndim == 4:
        return q[:, :, :t, :]
    raise ValueError(f"q must be (B,Q,T) or (B,Q,T,C), got {tuple(q.shape)}")


@torch.no_grad()
def evaluate_mse_mae(
    model: nn.Module,
    loader: DataLoader,
    quantiles: torch.Tensor,
    device: torch.device,
    *,
    memory_type: str,
    base_model: Optional[nn.Module] = None,
    base_hidden_source: str = "encoder",
) -> Tuple[float, float]:
    model.eval()
    q_levels = quantiles.to(device=device, dtype=torch.float32)
    central_idx = int(torch.argmin(torch.abs(q_levels - 0.5)).item())

    sum_abs = 0.0
    sum_sq = 0.0
    count = 0
    for batch in loader:
        context = batch["context"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        if memory_type in {"context", "context_multivar"}:
            q_pred = model(context)
        elif memory_type == "base_hidden_delta":
            if base_model is None:
                raise ValueError("base_model is required for memory_type=base_hidden_delta")
            base_out = base_model(context=context, return_hidden=True)
            q_base = base_out.quantile_preds.to(dtype=torch.float32)
            if base_hidden_source == "decoder":
                hidden = base_out.decoder_hidden_state
            else:
                hidden = base_out.encoder_hidden_states
            if hidden is None:
                raise ValueError(f"Missing hidden states from base_model (source={base_hidden_source})")
            _loc, scale, is_const = instance_norm_stats(context)
            delta = model(hidden, scale=scale.squeeze(-1), is_constant=is_const.squeeze(-1))
            q_pred = q_base + delta
        else:
            raise ValueError(f"Unknown memory_type: {memory_type}")
        # Support evaluating a pred_len-specific Memory model using teacher shards built at a longer horizon:
        # slice target to q_pred's time dimension (prefix).
        if q_pred.ndim not in (3, 4):
            raise ValueError(f"q_pred must be 3D or 4D, got {tuple(q_pred.shape)}")
        pred_len = int(q_pred.shape[2])
        target_t = int(target.shape[1])
        if target_t < pred_len:
            raise ValueError(f"target horizon too short: targetT={target_t} predT={pred_len}")
        if target_t != pred_len:
            target = _slice_target_time(target, pred_len)
        med = q_pred[:, central_idx, ...]
        err = med - target
        sum_abs += err.abs().sum().item()
        sum_sq += (err ** 2).sum().item()
        count += err.numel()
    mae = sum_abs / max(count, 1)
    mse = sum_sq / max(count, 1)
    return mse, mae


@torch.no_grad()
def evaluate_mse_mae_multi(
    model: nn.Module,
    loader: DataLoader,
    quantiles: torch.Tensor,
    device: torch.device,
    *,
    pred_lens: List[int],
    memory_type: str,
    base_model: Optional[nn.Module] = None,
    base_hidden_source: str = "encoder",
) -> Tuple[float, float, Dict[int, Tuple[float, float]]]:
    """
    Evaluate median MSE/MAE at multiple horizons in one pass.

    pred_lens are treated as prefixes: metrics for pl use the first pl steps.
    Returns:
      (mse_mean, mae_mean, per_pl_metrics)
    """
    if not pred_lens:
        raise ValueError("pred_lens must be non-empty")
    pred_lens_int = [int(x) for x in pred_lens]
    if any(x <= 0 for x in pred_lens_int):
        raise ValueError(f"pred_lens must be >0, got {pred_lens_int}")

    max_len = max(pred_lens_int)

    model.eval()
    q_levels = quantiles.to(device=device, dtype=torch.float32)
    central_idx = int(torch.argmin(torch.abs(q_levels - 0.5)).item())

    sum_abs: Dict[int, float] = {pl: 0.0 for pl in pred_lens_int}
    sum_sq: Dict[int, float] = {pl: 0.0 for pl in pred_lens_int}
    count: Dict[int, int] = {pl: 0 for pl in pred_lens_int}

    for batch in loader:
        context = batch["context"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)

        if memory_type in {"context", "context_multivar"}:
            try:
                q_pred = model(context, pred_len=max_len)
            except TypeError:
                q_pred = model(context)
        elif memory_type == "base_hidden_delta":
            if base_model is None:
                raise ValueError("base_model is required for memory_type=base_hidden_delta")
            if int(max_len) != int(target.shape[1]):
                raise ValueError("memory_type=base_hidden_delta only supports fixed horizon eval")
            base_out = base_model(context=context, return_hidden=True)
            q_base = base_out.quantile_preds.to(dtype=torch.float32)
            if base_hidden_source == "decoder":
                hidden = base_out.decoder_hidden_state
            else:
                hidden = base_out.encoder_hidden_states
            if hidden is None:
                raise ValueError(f"Missing hidden states from base_model (source={base_hidden_source})")
            _loc, scale, is_const = instance_norm_stats(context)
            delta = model(hidden, scale=scale.squeeze(-1), is_constant=is_const.squeeze(-1))
            q_pred = q_base + delta
        else:
            raise ValueError(f"Unknown memory_type: {memory_type}")

        q_pred = _slice_quantiles_time(q_pred, max_len)
        target = _slice_target_time(target, max_len)

        med = q_pred[:, central_idx, ...]
        for pl in pred_lens_int:
            med_pl = _slice_target_time(med, pl)
            tgt_pl = _slice_target_time(target, pl)
            err = med_pl - tgt_pl
            sum_abs[pl] += err.abs().sum().item()
            sum_sq[pl] += (err ** 2).sum().item()
            count[pl] += int(err.numel())

    per_pl: Dict[int, Tuple[float, float]] = {}
    for pl in pred_lens_int:
        mae = sum_abs[pl] / max(count[pl], 1)
        mse = sum_sq[pl] / max(count[pl], 1)
        per_pl[pl] = (float(mse), float(mae))

    mse_mean = float(sum(per_pl[pl][0] for pl in pred_lens_int) / len(pred_lens_int))
    mae_mean = float(sum(per_pl[pl][1] for pl in pred_lens_int) / len(pred_lens_int))
    return mse_mean, mae_mean, per_pl


def load_quantiles_from_base(base_model_path: str) -> List[float]:
    config = AutoConfig.from_pretrained(base_model_path)
    return [float(x) for x in config.chronos_config["quantiles"]]


def main():
    parser = argparse.ArgumentParser("Train TS-Memory (quantile) with offline teacher")
    parser.add_argument("--teacher_dir", type=str, required=True, help="Path to split dir with manifest.json (train)")
    parser.add_argument("--val_teacher_dir", type=str, required=True, help="Path to split dir with manifest.json (val)")
    parser.add_argument("--base_model_path", type=str, default="./checkpoints/base", help="Used to read quantiles by default")

    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="fixed",
        choices=["fixed", "conf"],
        help="How to set beta in the hybrid loss. fixed: use --beta. conf: per-sample beta_i from teacher confidence fields.",
    )
    parser.add_argument("--beta_min", type=float, default=0.1, help="Only for beta_schedule=conf")
    parser.add_argument("--beta_max", type=float, default=0.9, help="Only for beta_schedule=conf")
    parser.add_argument(
        "--conf_type",
        type=str,
        default="w_max",
        choices=["w_max", "entropy", "effective_k"],
        help="Teacher confidence type used for beta_schedule=conf.",
    )
    parser.add_argument("--conf_power", type=float, default=1.0, help="Only for beta_schedule=conf (conf^power)")
    parser.add_argument("--align_loss", type=str, default="huber", choices=["l2", "huber"])
    parser.add_argument("--crossing_lambda", type=float, default=0.0)

    # Emphasize the central (q≈0.5) quantile to improve MAE/median forecasts.
    parser.add_argument(
        "--task_central_quantile_weight",
        type=float,
        default=0.0,
        help="Extra weight added to the quantile closest to 0.5 in the pinball loss (0 disables).",
    )
    parser.add_argument(
        "--median_task_lambda",
        type=float,
        default=0.0,
        help="Additional loss weight on the median forecast (q closest to 0.5) vs y_true (0 disables).",
    )
    parser.add_argument(
        "--median_task_loss",
        type=str,
        default="huber",
        choices=["l1", "huber", "l2"],
        help="Loss type used for the extra median_task_lambda term.",
    )
    parser.add_argument(
        "--base_anchor_lambda",
        type=float,
        default=0.0,
        help="Additional loss weight to anchor the median forecast (q closest to 0.5) to the base model median (0 disables).",
    )
    parser.add_argument(
        "--base_anchor_loss",
        type=str,
        default="huber",
        choices=["l1", "huber", "l2"],
        help="Loss type used for base_anchor_lambda (median forecast vs q_base median).",
    )
    parser.add_argument(
        "--base_anchor_gate",
        type=str,
        default="none",
        choices=["none", "inv_conf"],
        help="Optional per-sample gate for base_anchor loss. inv_conf uses (1-conf)^conf_power where conf comes from teacher shards.",
    )
    parser.add_argument(
        "--distill_med_gate",
        type=str,
        default="none",
        choices=["none", "advantage"],
        help="Only for distill_target=abs_tail_delta_med: gate the central-quantile delta distillation. "
        "advantage: only distill median delta when teacher median MAE is better than base on the train batch.",
    )
    parser.add_argument(
        "--distill_med_adv_margin",
        type=float,
        default=0.0,
        help="Only for distill_med_gate=advantage: require (|y-base|-|y-teacher|) > margin to enable median distillation.",
    )

    parser.add_argument(
        "--select_metric",
        type=str,
        default="mse_then_mae",
        choices=["mse_then_mae", "mae_then_mse", "mse", "mae", "sum"],
        help="How to select best checkpoint on val.",
    )
    parser.add_argument(
        "--select_mae_weight",
        type=float,
        default=1.0,
        help="Only used when select_metric=sum: score = val_mse + select_mae_weight * val_mae.",
    )

    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=64)
    parser.add_argument(
        "--train_pred_lens",
        type=str,
        default=None,
        help="Optional comma-separated list of horizon lengths to sample per batch (prefixes). "
        "Requires the teacher shards and model to be built for max horizon == --pred_len.",
    )
    parser.add_argument(
        "--train_pred_lens_probs",
        type=str,
        default=None,
        help="Optional comma-separated sampling probabilities for --train_pred_lens (same length). "
        "If omitted, samples uniformly.",
    )
    parser.add_argument(
        "--val_pred_lens",
        type=str,
        default=None,
        help="Optional comma-separated list of horizons to evaluate/choose checkpoints on val. "
        "If set, uses the mean MSE/MAE over these horizons for --select_metric.",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=None,
        help="Only for memory_type=context_multivar. Default: infer from teacher shard context shape.",
    )
    parser.add_argument(
        "--memory_type",
        type=str,
        default="context",
        choices=["context", "context_multivar", "base_hidden_delta"],
        help="context: Memory consumes raw context (existing). "
        "context_multivar: Memory consumes full (B,L,C) context and predicts (B,Q,T,C). "
        "base_hidden_delta: Memory consumes frozen ChronosBolt hidden states and predicts delta vs base.",
    )
    parser.add_argument(
        "--distill_target",
        type=str,
        default="absolute",
        choices=["absolute", "delta", "abs_delta", "abs_tail_delta_med"],
        help="Alignment target used for the teacher term. "
        "absolute: align q_mem to q_teacher. "
        "delta: align (q_mem-q_base) to (q_teacher-q_base) (requires base model forward). "
        "abs_delta: align both absolute and delta targets (requires base model forward). "
        "abs_tail_delta_med: align tails (all quantiles except the central quantile) absolutely, "
        "and align the central quantile in delta space (requires base model forward).",
    )
    parser.add_argument(
        "--distill_abs_weight",
        type=float,
        default=1.0,
        help="Only for distill_target in {abs_delta, abs_tail_delta_med}: weight for absolute alignment loss.",
    )
    parser.add_argument(
        "--distill_delta_weight",
        type=float,
        default=1.0,
        help="Only for distill_target in {abs_delta, abs_tail_delta_med}: weight for delta alignment loss.",
    )
    parser.add_argument(
        "--base_hidden_source",
        type=str,
        default="encoder",
        choices=["encoder", "decoder"],
        help="Which hidden states from ChronosBolt to feed into memory_type=base_hidden_delta.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max tokens for hidden-based Memory positional embeddings (memory_type=base_hidden_delta).",
    )
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_decoder_layers", type=int, default=2)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Early stopping patience in epochs (0 disables). Stops when val score does not improve for this many epochs.",
    )
    parser.add_argument(
        "--early_stop_min_epochs",
        type=int,
        default=0,
        help="Minimum number of epochs before early stopping can trigger.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)

    if str(args.distill_med_gate) != "none" and str(args.distill_target) != "abs_tail_delta_med":
        raise ValueError("--distill_med_gate is only supported for distill_target=abs_tail_delta_med")
    if float(args.base_anchor_lambda) < 0.0:
        raise ValueError("--base_anchor_lambda must be >= 0")
    if int(args.early_stop_patience) < 0:
        raise ValueError("--early_stop_patience must be >= 0")
    if int(args.early_stop_min_epochs) < 0:
        raise ValueError("--early_stop_min_epochs must be >= 0")

    def _parse_int_csv(s: Optional[str]) -> Optional[List[int]]:
        if s is None:
            return None
        s = str(s).strip()
        if s == "":
            return None
        out: List[int] = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok == "":
                continue
            out.append(int(tok))
        return out

    def _parse_float_csv(s: Optional[str]) -> Optional[List[float]]:
        if s is None:
            return None
        s = str(s).strip()
        if s == "":
            return None
        out: List[float] = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok == "":
                continue
            out.append(float(tok))
        return out

    train_pred_lens = _parse_int_csv(args.train_pred_lens)
    train_pred_probs = _parse_float_csv(args.train_pred_lens_probs)
    if train_pred_lens is not None:
        if str(args.memory_type) == "base_hidden_delta":
            raise ValueError("train_pred_lens is not supported for memory_type=base_hidden_delta")
        if any(pl <= 0 for pl in train_pred_lens):
            raise ValueError(f"train_pred_lens must be >0, got {train_pred_lens}")
        if any(pl > int(args.pred_len) for pl in train_pred_lens):
            raise ValueError(f"train_pred_lens must be <= pred_len={args.pred_len}, got {train_pred_lens}")
        if train_pred_probs is not None:
            if len(train_pred_probs) != len(train_pred_lens):
                raise ValueError(
                    f"train_pred_lens_probs must match train_pred_lens length "
                    f"({len(train_pred_lens)}), got {len(train_pred_probs)}"
                )
            s = float(sum(train_pred_probs))
            if not (s > 0.0):
                raise ValueError(f"train_pred_lens_probs sum must be >0, got {train_pred_probs}")
            train_pred_probs = [float(x) / s for x in train_pred_probs]
        else:
            train_pred_probs = [1.0 for _ in train_pred_lens]

    val_pred_lens = _parse_int_csv(args.val_pred_lens)
    if val_pred_lens is not None:
        if any(pl <= 0 for pl in val_pred_lens):
            raise ValueError(f"val_pred_lens must be >0, got {val_pred_lens}")
        if any(pl > int(args.pred_len) for pl in val_pred_lens):
            raise ValueError(f"val_pred_lens must be <= pred_len={args.pred_len}, got {val_pred_lens}")
    else:
        val_pred_lens = [int(args.pred_len)]

    train_manifest = load_manifest(args.teacher_dir)
    val_manifest = load_manifest(args.val_teacher_dir)

    teacher_quantiles = [float(x) for x in train_manifest["meta"]["quantiles"]]
    base_quantiles = load_quantiles_from_base(args.base_model_path)

    if len(teacher_quantiles) != len(base_quantiles) or any(abs(a - b) > 1e-9 for a, b in zip(teacher_quantiles, base_quantiles)):
        raise ValueError(
            f"Quantiles mismatch between teacher and base config.\n"
            f"teacher={teacher_quantiles}\nbase={base_quantiles}"
        )

    quantiles_t = torch.tensor(teacher_quantiles, dtype=torch.float32, device=device)
    central_idx = int(torch.argmin(torch.abs(quantiles_t - 0.5)).item())
    tail_indices = [i for i in range(int(quantiles_t.numel())) if i != central_idx]
    task_q_weights = None
    if float(args.task_central_quantile_weight) > 0.0:
        task_q_weights = torch.ones_like(quantiles_t)
        task_q_weights[central_idx] += float(args.task_central_quantile_weight)

    # Infer teacher sample shape (used by multivariate Memory).
    if not train_manifest.get("shards"):
        raise ValueError(f"Empty teacher manifest (no shards): {args.teacher_dir}")
    first_shard_path = Path(args.teacher_dir) / train_manifest["shards"][0]["file"]
    first_shard = torch.load(first_shard_path, map_location="cpu")
    teacher_context = first_shard["context"]
    inferred_num_channels = int(teacher_context.shape[2]) if teacher_context.ndim == 3 else 1

    base_model = None
    base_pipe = None
    base_pred_len = int(AutoConfig.from_pretrained(args.base_model_path).chronos_config["prediction_length"])
    base_hidden_dim = int(AutoConfig.from_pretrained(args.base_model_path).d_model)
    need_q_base = str(args.distill_target) in {"delta", "abs_delta", "abs_tail_delta_med"} or float(args.base_anchor_lambda) > 0.0
    train_shards = list(train_manifest.get("shards", []))
    train_q_base_files = sum(1 for s in train_shards if s.get("q_base_file"))
    train_has_full_q_base_cache = bool(train_shards) and train_q_base_files == len(train_shards)
    if train_q_base_files > 0 or train_manifest.get("q_base_cache") is not None:
        print(
            f"[q_base_cache] train_shards={len(train_shards)} q_base_files={train_q_base_files} "
            f"full_cache={train_has_full_q_base_cache} need_q_base={need_q_base}"
        )

    if args.memory_type == "context":
        model_cfg = MemoryTSQuantileConfig(
            context_len=args.context_len,
            pred_len=args.pred_len,
            num_quantiles=len(teacher_quantiles),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_decoder_layers=args.n_decoder_layers,
            patch_len=args.patch_len,
            dropout=args.dropout,
        )
        model = MemoryTSQuantile(model_cfg).to(device)
        if need_q_base and not train_has_full_q_base_cache:
            # Need frozen base model to compute q_base for delta-distillation.
            from models.ChronosBolt import ChronosBoltPipeline

            base_pipe = ChronosBoltPipeline.from_pretrained(args.base_model_path)
            base_pipe.model.load_state_dict(
                torch.load(os.path.join(args.base_model_path, "autogluon_model.pth"), map_location="cpu")
            )
            base_pipe.model.to(device)
            base_pipe.model.eval()
            for p in base_pipe.model.parameters():
                p.requires_grad_(False)
    elif args.memory_type == "context_multivar":
        if float(args.base_anchor_lambda) > 0.0:
            raise ValueError("base_anchor_lambda is not supported for memory_type=context_multivar")
        if str(args.distill_target) != "absolute":
            raise ValueError("memory_type=context_multivar currently supports distill_target=absolute only")
        if teacher_context.ndim != 3:
            raise ValueError(
                f"memory_type=context_multivar requires teacher context to be 3D (N,L,C), got {tuple(teacher_context.shape)}"
            )
        num_channels = inferred_num_channels if args.num_channels is None else int(args.num_channels)
        model_cfg = MemoryTSQuantileMultivarConfig(
            context_len=args.context_len,
            pred_len=args.pred_len,
            num_quantiles=len(teacher_quantiles),
            num_channels=num_channels,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_decoder_layers=args.n_decoder_layers,
            patch_len=args.patch_len,
            dropout=args.dropout,
        )
        model = MemoryTSQuantileMultivar(model_cfg).to(device)
    elif args.memory_type == "base_hidden_delta":
        if int(args.pred_len) != int(base_pred_len):
            raise ValueError(
                f"memory_type=base_hidden_delta currently requires pred_len == base prediction_length "
                f"(pred_len={args.pred_len}, base={base_pred_len})."
            )

        # Lazily import to avoid heavy ChronosBolt imports unless requested.
        from models.ChronosBolt import ChronosBoltPipeline

        base_pipe = ChronosBoltPipeline.from_pretrained(args.base_model_path)
        base_pipe.model.load_state_dict(
            torch.load(os.path.join(args.base_model_path, "autogluon_model.pth"), map_location="cpu")
        )
        base_pipe.model.to(device)
        base_pipe.model.eval()
        for p in base_pipe.model.parameters():
            p.requires_grad_(False)
        base_model = base_pipe.model

        model_cfg = MemoryTSHiddenDeltaConfig(
            pred_len=args.pred_len,
            num_quantiles=len(teacher_quantiles),
            hidden_dim_in=base_hidden_dim,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_decoder_layers=args.n_decoder_layers,
            dropout=args.dropout,
            max_tokens=int(args.max_tokens),
            hidden_source=str(args.base_hidden_source),
        )
        model = MemoryTSHiddenDelta(model_cfg).to(device)
    else:
        raise ValueError(f"Unknown memory_type: {args.memory_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = TeacherShardIterableDataset(args.teacher_dir, shuffle=True, seed=args.seed)
    val_ds = TeacherShardIterableDataset(args.val_teacher_dir, shuffle=False, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _score(mse: float, mae: float):
        metric = str(args.select_metric)
        if metric == "mse_then_mae":
            return (mse, mae)
        if metric == "mae_then_mse":
            return (mae, mse)
        if metric == "mse":
            return (mse,)
        if metric == "mae":
            return (mae,)
        if metric == "sum":
            return (mse + float(args.select_mae_weight) * mae,)
        raise ValueError(f"Unknown select_metric: {metric}")

    best_score = None
    best_mse = float("inf")
    best_mae = float("inf")
    best_epoch = -1
    early_patience = int(args.early_stop_patience)
    early_min_epochs = int(args.early_stop_min_epochs)

    teacher_k = int(train_manifest["meta"].get("k", 0) or 0)
    if str(args.beta_schedule) == "conf":
        if not (0.0 <= float(args.beta_min) <= float(args.beta_max) <= 1.0):
            raise ValueError(
                f"Require 0<=beta_min<=beta_max<=1, got beta_min={args.beta_min} beta_max={args.beta_max}"
            )
        if teacher_k <= 0:
            raise ValueError("beta_schedule=conf requires teacher manifest to contain meta.k (>0)")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        align_sum = 0.0
        task_sum = 0.0
        median_sum = 0.0
        anchor_sum = 0.0
        cross_sum = 0.0
        beta_sum = 0.0
        conf_sum = 0.0
        med_gate_sum = 0.0
        med_gate_batches = 0
        beta_min_t = torch.tensor(float("inf"), device=device)
        beta_max_t = torch.tensor(float("-inf"), device=device)
        conf_min_t = torch.tensor(float("inf"), device=device)
        conf_max_t = torch.tensor(float("-inf"), device=device)
        conf_sum_t = torch.zeros((), device=device, dtype=torch.float64)
        conf_sq_sum_t = torch.zeros((), device=device, dtype=torch.float64)
        conf_count = 0
        n_batches = 0
        for batch in train_loader:
            context = batch["context"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            q_teacher = batch["q_teacher"].to(device=device, dtype=torch.float32)
            q_base = None
            if "q_base" in batch:
                q_base = batch["q_base"].to(device=device, dtype=torch.float32)

            pred_len_b = int(args.pred_len)
            if train_pred_lens is not None and train_pred_probs is not None:
                pred_len_b = int(random.choices(train_pred_lens, weights=train_pred_probs, k=1)[0])
            # Always slice to the chosen prefix horizon. This supports training per-horizon Memory models
            # from teacher shards built at a longer max horizon (e.g. build teacher at 720, train Memory at 96).
            target_t = int(target.shape[1])
            if target_t < pred_len_b:
                raise ValueError(f"target horizon too short: targetT={target_t} predT={pred_len_b}")
            if target_t != pred_len_b:
                target = _slice_target_time(target, pred_len_b)

            teacher_t = int(q_teacher.shape[2])
            if teacher_t < pred_len_b:
                raise ValueError(f"teacher horizon too short: teacherT={teacher_t} predT={pred_len_b}")
            if teacher_t != pred_len_b:
                q_teacher = _slice_quantiles_time(q_teacher, pred_len_b)

            if q_base is not None:
                base_t = int(q_base.shape[2])
                if base_t < pred_len_b:
                    raise ValueError(f"base cache horizon too short: baseT={base_t} predT={pred_len_b}")
                if base_t != pred_len_b:
                    q_base = _slice_quantiles_time(q_base, pred_len_b)

            if args.memory_type in {"context", "context_multivar"}:
                try:
                    q_mem = model(context, pred_len=pred_len_b)
                except TypeError:
                    q_mem = model(context)
                    if pred_len_b != int(args.pred_len):
                        q_mem = _slice_quantiles_time(q_mem, pred_len_b)
                distill_target = str(args.distill_target)
                if distill_target == "absolute":
                    align_abs_per = None
                    align_delta_per = None
                    align_lhs = q_mem
                    align_rhs = q_teacher
                    if float(args.base_anchor_lambda) > 0.0 and q_base is None:
                        if base_pipe is None:
                            raise RuntimeError("base_anchor_lambda>0 requires q_base (no cache and base_pipe missing)")
                        with torch.no_grad():
                            q_base = base_pipe.predict(context=context, prediction_length=int(pred_len_b))
                elif distill_target == "delta":
                    if q_base is None:
                        if base_pipe is None:
                            raise ValueError("distill_target=delta requires q_base (no cache and base_pipe missing)")
                        with torch.no_grad():
                            q_base = base_pipe.predict(context=context, prediction_length=int(pred_len_b))
                    align_abs_per = None
                    align_delta_per = None
                    align_lhs = q_mem - q_base
                    align_rhs = q_teacher - q_base
                elif distill_target == "abs_delta":
                    if q_base is None:
                        if base_pipe is None:
                            raise ValueError("distill_target=abs_delta requires q_base (no cache and base_pipe missing)")
                        with torch.no_grad():
                            q_base = base_pipe.predict(context=context, prediction_length=int(pred_len_b))
                    # We'll compute per-sample losses for abs and delta separately below.
                    align_abs_per = (q_mem, q_teacher)
                    align_delta_per = (q_mem - q_base, q_teacher - q_base)
                    align_lhs = None
                    align_rhs = None
                elif distill_target == "abs_tail_delta_med":
                    if q_base is None:
                        if base_pipe is None:
                            raise ValueError(
                                "distill_target=abs_tail_delta_med requires q_base (no cache and base_pipe missing)"
                            )
                        with torch.no_grad():
                            q_base = base_pipe.predict(context=context, prediction_length=int(pred_len_b))
                    # Tails: absolute alignment; Central quantile: delta alignment.
                    align_abs_per = None
                    if tail_indices:
                        align_abs_per = (q_mem[:, tail_indices, ...], q_teacher[:, tail_indices, ...])
                    align_delta_per = (
                        (q_mem - q_base)[:, central_idx, ...],
                        (q_teacher - q_base)[:, central_idx, ...],
                    )
                    align_lhs = None
                    align_rhs = None
                else:
                    raise ValueError(f"Unknown distill_target: {distill_target}")
            else:
                assert base_model is not None
                with torch.no_grad():
                    base_out = base_model(context=context, return_hidden=True)
                    q_base = base_out.quantile_preds.to(dtype=torch.float32)
                    if args.base_hidden_source == "decoder":
                        hidden = base_out.decoder_hidden_state
                    else:
                        hidden = base_out.encoder_hidden_states
                    if hidden is None:
                        raise ValueError(f"Missing hidden states from base_model (source={args.base_hidden_source})")

                _loc, scale, is_const = instance_norm_stats(context)
                delta = model(hidden, scale=scale.squeeze(-1), is_constant=is_const.squeeze(-1))
                q_mem = q_base + delta
                distill_target = str(args.distill_target)
                if distill_target == "absolute":
                    align_abs_per = None
                    align_delta_per = None
                    align_lhs = q_mem
                    align_rhs = q_teacher
                elif distill_target == "delta":
                    align_abs_per = None
                    align_delta_per = None
                    align_lhs = delta
                    align_rhs = q_teacher - q_base
                elif distill_target == "abs_delta":
                    align_abs_per = (q_mem, q_teacher)
                    align_delta_per = (delta, q_teacher - q_base)
                    align_lhs = None
                    align_rhs = None
                elif distill_target == "abs_tail_delta_med":
                    align_abs_per = None
                    if tail_indices:
                        align_abs_per = (q_mem[:, tail_indices, ...], q_teacher[:, tail_indices, ...])
                    align_delta_per = (delta[:, central_idx, ...], (q_teacher - q_base)[:, central_idx, ...])
                    align_lhs = None
                    align_rhs = None
                else:
                    raise ValueError(f"Unknown distill_target: {distill_target}")

            def _per_sample_align(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
                if args.align_loss == "huber":
                    err = F.smooth_l1_loss(lhs, rhs, reduction="none")
                else:
                    err = F.mse_loss(lhs, rhs, reduction="none")
                return err.reshape(err.shape[0], -1).mean(dim=1)

            if str(args.distill_target) in {"abs_delta", "abs_tail_delta_med"}:
                abs_w = float(args.distill_abs_weight)
                delta_w = float(args.distill_delta_weight)
                if abs_w < 0.0 or delta_w < 0.0:
                    raise ValueError(
                        f"distill_abs_weight and distill_delta_weight must be >=0, got {abs_w}, {delta_w}"
                    )
                if abs_w == 0.0 and delta_w == 0.0:
                    raise ValueError("At least one of distill_abs_weight/distill_delta_weight must be >0")
                if align_delta_per is None:
                    raise RuntimeError("Internal error: abs/delta hybrid requested but delta tensors are missing")
                l_delta_per = _per_sample_align(align_delta_per[0], align_delta_per[1])
                if distill_target == "abs_tail_delta_med" and str(args.distill_med_gate) == "advantage":
                    if q_base is None:
                        raise RuntimeError("distill_med_gate=advantage requires q_base (missing base forward)")
                    med_teacher = q_teacher[:, central_idx, ...]
                    med_base = q_base[:, central_idx, ...]
                    err_teacher = (target - med_teacher).abs().reshape(target.shape[0], -1).mean(dim=1)
                    err_base = (target - med_base).abs().reshape(target.shape[0], -1).mean(dim=1)
                    adv = err_base - err_teacher
                    gate = (adv > float(args.distill_med_adv_margin)).to(dtype=torch.float32)
                    l_delta_per = l_delta_per * gate
                    med_gate_sum += float(gate.mean().item())
                    med_gate_batches += 1
                if align_abs_per is None:
                    l_abs_per = torch.zeros_like(l_delta_per)
                else:
                    l_abs_per = _per_sample_align(align_abs_per[0], align_abs_per[1])
                l_align_per = abs_w * l_abs_per + delta_w * l_delta_per
            else:
                if align_lhs is None or align_rhs is None:
                    raise RuntimeError("Internal error: align tensors are missing")
                l_align_per = _per_sample_align(align_lhs, align_rhs)

            l_task_per = weighted_pinball_loss_per_sample(q_mem, target, quantiles_t, task_q_weights)

            conf_norm = None
            if str(args.beta_schedule) == "conf" or str(args.base_anchor_gate) == "inv_conf":
                conf_type = str(args.conf_type)
                if conf_type == "w_max":
                    if "conf_w_max" not in batch:
                        raise ValueError("beta_schedule/base_anchor_gate requires teacher shards to contain conf_w_max")
                    conf = batch["conf_w_max"].to(device=device, dtype=torch.float32).reshape(-1)
                    conf_norm = conf.clamp(0.0, 1.0)
                elif conf_type == "effective_k":
                    if "conf_effective_k" not in batch:
                        raise ValueError("beta_schedule/base_anchor_gate requires teacher shards to contain conf_effective_k")
                    eff_k = batch["conf_effective_k"].to(device=device, dtype=torch.float32).reshape(-1)
                    if teacher_k <= 1:
                        conf_norm = torch.ones_like(eff_k)
                    else:
                        conf_norm = ((eff_k - 1.0) / float(teacher_k - 1)).clamp(0.0, 1.0)
                else:
                    if "conf_entropy" not in batch:
                        raise ValueError("beta_schedule/base_anchor_gate requires teacher shards to contain conf_entropy")
                    ent = batch["conf_entropy"].to(device=device, dtype=torch.float32).reshape(-1)
                    if teacher_k <= 1:
                        conf_norm = torch.ones_like(ent)
                    else:
                        conf_norm = (1.0 - ent / float(math.log(teacher_k))).clamp(0.0, 1.0)

            if str(args.beta_schedule) == "fixed":
                beta_i = torch.full_like(l_align_per, float(args.beta))
            else:
                assert conf_norm is not None
                beta_i = float(args.beta_min) + (float(args.beta_max) - float(args.beta_min)) * conf_norm.pow(
                    float(args.conf_power)
                )
                beta_i = beta_i.clamp(0.0, 1.0)

            beta_min_t = torch.minimum(beta_min_t, beta_i.min())
            beta_max_t = torch.maximum(beta_max_t, beta_i.max())
            if str(args.beta_schedule) == "conf":
                conf_min_t = torch.minimum(conf_min_t, conf_norm.min())
                conf_max_t = torch.maximum(conf_max_t, conf_norm.max())
                conf_sum_t += conf_norm.double().sum()
                conf_sq_sum_t += conf_norm.double().square().sum()
                conf_count += int(conf_norm.numel())

            l_cross = crossing_penalty(q_mem) * float(args.crossing_lambda)
            l_median = torch.tensor(0.0, device=device)
            if float(args.median_task_lambda) > 0.0:
                med = q_mem[:, central_idx, ...]
                if args.median_task_loss == "l1":
                    l_median = F.l1_loss(med, target)
                elif args.median_task_loss == "l2":
                    l_median = F.mse_loss(med, target)
                else:
                    l_median = F.smooth_l1_loss(med, target)

            l_anchor = torch.tensor(0.0, device=device)
            if float(args.base_anchor_lambda) > 0.0:
                if q_base is None:
                    raise RuntimeError("base_anchor_lambda>0 requires q_base (missing base forward)")
                med_mem = q_mem[:, central_idx, ...]
                med_base = q_base[:, central_idx, ...].detach()
                if args.base_anchor_loss == "l1":
                    err = (med_mem - med_base).abs()
                elif args.base_anchor_loss == "l2":
                    diff = med_mem - med_base
                    err = diff * diff
                else:
                    err = F.smooth_l1_loss(med_mem, med_base, reduction="none")
                l_anchor_per = err.reshape(err.shape[0], -1).mean(dim=1)
                if str(args.base_anchor_gate) == "inv_conf":
                    assert conf_norm is not None
                    gate = (1.0 - conf_norm).pow(float(args.conf_power))
                else:
                    gate = torch.ones_like(l_anchor_per)
                l_anchor = (l_anchor_per * gate).mean()

            loss_main = (beta_i * l_align_per + (1.0 - beta_i) * l_task_per).mean()
            loss = (
                loss_main
                + float(args.median_task_lambda) * l_median
                + float(args.base_anchor_lambda) * l_anchor
                + l_cross
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_sum += loss.item()
            align_sum += l_align_per.mean().item()
            task_sum += l_task_per.mean().item()
            median_sum += l_median.item()
            anchor_sum += l_anchor.item()
            cross_sum += l_cross.item()
            beta_sum += beta_i.mean().item()
            if str(args.beta_schedule) == "conf":
                conf_sum += conf_norm.mean().item()
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)
        avg_align = align_sum / max(n_batches, 1)
        avg_task = task_sum / max(n_batches, 1)
        avg_median = median_sum / max(n_batches, 1)
        avg_anchor = anchor_sum / max(n_batches, 1)
        avg_cross = cross_sum / max(n_batches, 1)
        avg_beta = beta_sum / max(n_batches, 1)
        avg_conf = conf_sum / max(n_batches, 1) if str(args.beta_schedule) == "conf" else float("nan")
        avg_med_gate = med_gate_sum / max(med_gate_batches, 1) if med_gate_batches > 0 else float("nan")
        beta_min = beta_min_t.item() if torch.isfinite(beta_min_t) else float("nan")
        beta_max = beta_max_t.item() if torch.isfinite(beta_max_t) else float("nan")
        conf_min = float("nan")
        conf_max = float("nan")
        conf_std = float("nan")
        if str(args.beta_schedule) == "conf" and conf_count > 0:
            conf_min = conf_min_t.item() if torch.isfinite(conf_min_t) else float("nan")
            conf_max = conf_max_t.item() if torch.isfinite(conf_max_t) else float("nan")
            conf_mean_exact = (conf_sum_t / float(conf_count)).item()
            conf_var = (conf_sq_sum_t / float(conf_count)).item() - conf_mean_exact ** 2
            conf_std = math.sqrt(max(conf_var, 0.0))
        if val_pred_lens is not None and len(val_pred_lens) > 1 and str(args.memory_type) in {"context", "context_multivar"}:
            val_mse, val_mae, per_pl = evaluate_mse_mae_multi(
                model,
                val_loader,
                quantiles_t,
                device,
                pred_lens=list(val_pred_lens),
                memory_type=str(args.memory_type),
                base_model=base_model,
                base_hidden_source=str(args.base_hidden_source),
            )
            per_pl_str = " ".join([f"{pl}:{per_pl[pl][0]:.4f}/{per_pl[pl][1]:.4f}" for pl in sorted(per_pl.keys())])
            print(f"[val_pred_lens] {','.join([str(x) for x in sorted(per_pl.keys())])} {per_pl_str}")
        else:
            val_mse, val_mae = evaluate_mse_mae(
                model,
                val_loader,
                quantiles_t,
                device,
                memory_type=str(args.memory_type),
                base_model=base_model,
                base_hidden_source=str(args.base_hidden_source),
            )
        if str(args.beta_schedule) == "conf":
            print(
                f"epoch {epoch:03d} | train_loss={avg_loss:.6f} beta={avg_beta:.3f}[{beta_min:.3f},{beta_max:.3f}] "
                f"conf={avg_conf:.3f}[{conf_min:.3f},{conf_max:.3f}] std={conf_std:.3f} "
                f"(align={avg_align:.6f} task={avg_task:.6f} median={avg_median:.6f} anchor={avg_anchor:.6f} cross={avg_cross:.6f}) "
                f"med_gate={avg_med_gate:.3f} "
                f"| val_mse={val_mse:.6f} | val_mae={val_mae:.6f}"
            )
        else:
            print(
                f"epoch {epoch:03d} | train_loss={avg_loss:.6f} beta={avg_beta:.3f}[{beta_min:.3f},{beta_max:.3f}] "
                f"(align={avg_align:.6f} task={avg_task:.6f} median={avg_median:.6f} anchor={avg_anchor:.6f} cross={avg_cross:.6f}) "
                f"med_gate={avg_med_gate:.3f} "
                f"| val_mse={val_mse:.6f} | val_mae={val_mae:.6f}"
            )

        score = _score(val_mse, val_mae)
        if best_score is None or score < best_score:
            best_score = score
            best_epoch = epoch
            best_mse = val_mse
            best_mae = val_mae
            ckpt = {
                "state_dict": model.state_dict(),
                "model_config": asdict(model_cfg),
                "train_args": vars(args),
                "quantiles": teacher_quantiles,
                "memory_type": str(args.memory_type),
                "best_epoch": best_epoch,
                "val_mse": best_mse,
                "val_mae": best_mae,
                "val_score": best_score,
            }
            torch.save(ckpt, save_dir / "best.pth")

        # Always save last
        ckpt_last = {
            "state_dict": model.state_dict(),
            "model_config": asdict(model_cfg),
            "train_args": vars(args),
            "quantiles": teacher_quantiles,
            "memory_type": str(args.memory_type),
            "epoch": epoch,
            "val_mse": val_mse,
            "val_mae": val_mae,
        }
        torch.save(ckpt_last, save_dir / "last.pth")

        if early_patience > 0 and epoch >= early_min_epochs:
            if best_epoch >= 0 and (epoch - best_epoch) >= early_patience:
                print(
                    f"[early_stop] epoch={epoch} best_epoch={best_epoch} "
                    f"patience={early_patience} min_epochs={early_min_epochs} best_val_score={best_score}"
                )
                break

    print(f"[done] best_epoch={best_epoch} best_val_mse={best_mse:.6f} best_val_mae={best_mae:.6f}")


if __name__ == "__main__":
    main()
