import os
import time
import torch
import torch.nn.functional as F
import random
import argparse
import warnings
import numpy as np
import copy
import sys

from transformers import AutoConfig
from utils.tools import test, test_retrieve
from data_provider.data_factory import data_provider
from models.ChronosBolt import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval
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

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Chronos-bolt')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)
parser.add_argument('--all', type=int, default=0)
parser.add_argument(
    '--return_feature_id',
    type=int,
    default=0,
    help='For non-retrieval datasets: whether Dataset.__getitem__ returns feat_id (0/1).',
)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=20)

parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--train_ratio', type=float, default=1.0 , required=False)
parser.add_argument('--save_file_name', type=str, default=None)
parser.add_argument('--gpu_loc', type=int, default=1)
parser.add_argument('--n_scale', type=float, default=-1)
parser.add_argument('--method', type=str, default='')

# pipeline mode (keep --mode as retrieval mode for backward compatibility)
parser.add_argument(
    '--run_mode',
    type=str,
    default='auto',
    choices=['auto', 'rag', 'base', 'memory_quantile'],
    help="auto: keep original behavior; memory_quantile: NO RETRIEVAL inference",
)

# retrieve
parser.add_argument('--embedding_tuning', type=str, default=None)
parser.add_argument('--metadata', type=dict, default={})
parser.add_argument('--metadata_database_name', type=str, default='ETTh2')
parser.add_argument('--metadata_frequency', type=str, default='hour')
parser.add_argument('--mode', type=str, default='only_self_train')
parser.add_argument('--top_k', type=int, default=1)
parser.add_argument('--retrieval_database_dir', type=str, default='../retrieval_database/')
parser.add_argument('--dimension', type=int, default=768)
parser.add_argument('--embedding_model_type', type=str, default='chronos')
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--lookback_length', type=int, default=512)

# augment
parser.add_argument('--augment_mode', type=str, default='moe2')

parser.add_argument('--checkpoint_model_path', type=str, default='None')
parser.add_argument('--pretrained_model_path', type=str, default='./checkpoints/base')

# memory (quantile)
parser.add_argument('--memory_ckpt', type=str, default=None)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--alpha_search', type=str, default=None, help='comma-separated, e.g. "0.2,0.4,0.6,0.8"')
parser.add_argument(
    "--alpha_mode",
    type=str,
    default="scalar",
    choices=["scalar", "2segment"],
    help="How alpha is applied when fusing base and memory point forecasts. scalar: one alpha for all horizons. "
    "2segment: alpha_short for first --alpha_split steps, alpha_long for the rest (both searched from --alpha_search list).",
)
parser.add_argument(
    "--alpha_split",
    type=int,
    default=16,
    help="Only for --alpha_mode=2segment: number of early horizon steps using alpha_short.",
)
parser.add_argument(
    '--alpha_search_metric',
    type=str,
    default='mse_then_mae',
    choices=['mse_then_mae', 'mae_then_mse', 'mse', 'mae', 'sum', 'combo', 'mae_guard'],
    help='How to pick best alpha on val when --alpha_search is set.',
)
parser.add_argument(
    '--alpha_search_mae_weight',
    type=float,
    default=1.0,
    help='Only used when alpha_search_metric is sum/combo: score = val_mse + alpha_search_mae_weight * val_mae.',
)
parser.add_argument(
    '--alpha_search_mse_guard_ratio',
    type=float,
    default=0.02,
    help="Only used when alpha_search_metric=mae_guard. Select the (alpha,...) with best MAE among candidates whose "
    "MSE is within (1+ratio)*best_MSE on the selection split (ratio>=0).",
)
parser.add_argument(
    "--alpha_search_select_split",
    type=str,
    default="val",
    choices=["val", "test"],
    help="Which split is used to select the best (alpha, point_quantile[, alpha_short/long]). "
    "val is research-valid; test is leaky (uses test labels) and intended only for deployment-style tuning.",
)
parser.add_argument(
    "--point_quantile",
    type=float,
    default=0.5,
    help="Point-forecast quantile used to compute MSE/MAE from quantile preds (default: 0.5).",
)
parser.add_argument(
    "--point_quantile_search",
    type=str,
    default=None,
    help='Optional comma-separated list to jointly search with alpha on val, e.g. "0.4,0.45,0.5,0.55,0.6".',
)
parser.add_argument(
    "--point_quantile_method",
    type=str,
    default="nearest",
    choices=["nearest", "linear"],
    help="How to get point prediction from quantile grid. nearest preserves old behavior; linear enables interpolation.",
)
parser.add_argument(
    "--bias_correct",
    type=str,
    default="none",
    choices=["none", "global", "horizon", "horizon_shrink_smooth"],
    help="Optional median bias correction computed on val residuals and applied to test point forecasts.",
)
parser.add_argument(
    "--bias_correct_split",
    type=str,
    default="val",
    choices=["val", "test"],
    help="Which split residuals are used to estimate bias when --bias_correct != none. "
    "val is research-valid; test is leaky (uses test labels).",
)
parser.add_argument(
    "--bias_correct_max_windows",
    type=int,
    default=5000,
    help="Max number of val windows used to estimate bias (0 means all). Only used when --bias_correct != none.",
)
parser.add_argument(
    "--bias_correct_smooth_window",
    type=int,
    default=9,
    help="Only for bias_correct=horizon_shrink_smooth: moving-average window over horizons (odd >=1).",
)
parser.add_argument(
    "--bias_correct_shrink_lambda",
    type=float,
    default=1000.0,
    help="Only for bias_correct=horizon_shrink_smooth: shrink factor is n/(n+lambda), where n is #val windows used.",
)
parser.add_argument(
    "--bias_correct_report_grid",
    type=int,
    default=0,
    help="If 1 and --bias_correct != none, also compute and report bias-corrected test metrics for every alpha in "
    "--alpha_search (instead of only the selected alpha).",
)
parser.add_argument(
    "--bias_correct_select_mode",
    type=str,
    default="raw",
    choices=["raw", "bias", "auto"],
    help="How alpha/point_quantile selection is performed when --bias_correct != none. "
    "raw: select on raw metrics (backward-compatible), then optionally apply bias to test. "
    "bias: select on bias-corrected metrics. "
    "auto: pick whichever of raw vs bias-corrected gives a better selection metric on the selection split.",
)

# memory speed report (inference; NO-RETRIEVAL mode)
parser.add_argument(
    "--report_speed",
    type=int,
    default=0,
    help="If 1, benchmark base vs Memory forward latency (ms/iter) in memory_quantile mode and write to the result file.",
)
parser.add_argument(
    "--speed_split",
    type=str,
    default="test",
    choices=["val", "test"],
    help="Which split loader is used for the speed benchmark (memory_quantile mode only).",
)
parser.add_argument(
    "--speed_warmup_iters",
    type=int,
    default=5,
    help="Number of warmup iterations before timing (speed benchmark).",
)
parser.add_argument(
    "--speed_num_iters",
    type=int,
    default=20,
    help="Number of timed iterations (speed benchmark).",
)

args = parser.parse_args()

if args.save_file_name is not None : 
    log_fine_name = args.save_file_name
else:
    log_fine_name = f"{args.model_id}.txt"

# Backward-compatible alias: allow `--mode memory_quantile` when --run_mode is not set explicitly.
if args.run_mode == 'auto' and args.mode in {'rag', 'base', 'memory_quantile'}:
    args.run_mode = args.mode
    args.mode = 'only_self_train'

if torch.cuda.is_available():
    visible_gpus = torch.cuda.device_count()
    gpu_loc = int(getattr(args, "gpu_loc", 0))
    if gpu_loc < 0 or gpu_loc >= visible_gpus:
        print(
            f"Warning: gpu_loc={gpu_loc} is out of range for visible_gpus={visible_gpus}; "
            "falling back to gpu_loc=0 (note: CUDA_VISIBLE_DEVICES may mask GPUs)."
        )
        gpu_loc = 0
    device_address = f"cuda:{gpu_loc}"
    map_location = device_address
else:
    device_address = 'cpu'
    map_location = device_address
    print("Warning: CUDA is not available, falling back to CPU. Check GPU allocation and torch install.")
        
SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}
mses = []
maes = []
print(args.model_id)

args.metadata['lookback_length'] = args.lookback_length
args.metadata['frequency'] = args.metadata_frequency
args.metadata['database_name'] = args.metadata_database_name.split(' ')
ori_data_path = args.data_path

best_model_path = args.checkpoint_model_path
print(f'best_model_path: {best_model_path}')
if args.run_mode in {'rag', 'auto'} and 'retrieve' in args.model_id:
    if not os.path.exists(best_model_path):
        exit('no corresponding checkpoint!!')
    
if args.freq == 0:
    args.freq = 'h'

def _memory_eval(model_base, model_mem, loader, quantiles, device, alpha_list):
    q_levels = torch.tensor(quantiles, dtype=torch.float32, device=device)
    central_idx = int(torch.abs(q_levels - 0.5).argmin().item())

    # Accumulate on-device to avoid per-batch GPU sync from `.item()`.
    sum_abs = {a: torch.zeros((), device=device, dtype=torch.float64) for a in alpha_list}
    sum_sq = {a: torch.zeros((), device=device, dtype=torch.float64) for a in alpha_list}
    count = 0
    alpha0_checked = False

    for batch in loader:
        batch_x, batch_y, *_ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if batch_x.ndim == 3 and batch_x.shape[-1] == 1:
            batch_x = batch_x.squeeze(-1)
        if batch_y.ndim == 3 and batch_y.shape[-1] == 1:
            batch_y = batch_y.squeeze(-1)
        y_true = batch_y[:, -args.pred_len:]

        with torch.no_grad():
            if getattr(model_mem, "requires_base_hidden", False):
                if batch_x.ndim != 2:
                    raise ValueError("base_hidden_delta memory requires univariate (B,L) context")
                base_out = model_base.model(context=batch_x, return_hidden=True)
                q_base = base_out.quantile_preds.to(batch_x)  # (B,Q,T)
                hidden_source = getattr(model_mem, "config", None).hidden_source if hasattr(model_mem, "config") else "encoder"
                if hidden_source == "decoder":
                    hidden = base_out.decoder_hidden_state
                else:
                    hidden = base_out.encoder_hidden_states
                if hidden is None:
                    raise ValueError(f"Missing hidden states from base model (hidden_source={hidden_source})")

                _loc, scale, is_const = instance_norm_stats(batch_x)
                delta = model_mem(hidden, scale=scale.squeeze(-1), is_constant=is_const.squeeze(-1))
                assert delta.shape == q_base.shape, f"shape mismatch: base={q_base.shape}, delta={delta.shape}"
                q_mem = q_base + delta
            else:
                if batch_x.ndim == 2:
                    q_base = model_base.predict(context=batch_x, prediction_length=args.pred_len)  # (B,Q,T)
                elif batch_x.ndim == 3:
                    B, L, C = batch_x.shape
                    x_flat = batch_x.permute(0, 2, 1).reshape(B * C, L)
                    q_base_flat = model_base.predict(context=x_flat, prediction_length=args.pred_len)  # (B*C,Q,T)
                    q_base = q_base_flat.reshape(B, C, q_base_flat.shape[1], q_base_flat.shape[2]).permute(0, 2, 3, 1).contiguous()
                else:
                    raise ValueError(f"Unexpected batch_x shape: {batch_x.shape}")
                q_mem = model_mem(batch_x)

        assert q_base.shape == q_mem.shape, f"shape mismatch: base={q_base.shape}, mem={q_mem.shape}"

        # Sanity: alpha=0 must exactly match base without touching q_mem (avoid 0*NaN propagation).
        if not alpha0_checked:
            q0 = q_base
            assert q0.data_ptr() == q_base.data_ptr(), "alpha=0 should directly use q_base tensor"
            finite = torch.isfinite(q_base)
            if finite.any():
                max_diff0 = (q0[finite] - q_base[finite]).abs().max().item()
                assert max_diff0 < 1e-7, f"alpha=0 sanity failed: max_diff={max_diff0}"
            alpha0_checked = True

        for a in alpha_list:
            if a <= 0.0:
                q_final = q_base
            elif a >= 1.0:
                q_final = q_mem
            else:
                if getattr(model_mem, "outputs_delta", False):
                    # q_mem = q_base + delta  => q_final = q_base + a * delta
                    q_final = q_base + a * (q_mem - q_base)
                else:
                    q_final = (1.0 - a) * q_base + a * q_mem
            pred = q_final[:, central_idx, ...]
            err = pred - y_true
            sum_abs[a] += err.abs().sum().double()
            sum_sq[a] += (err ** 2).sum().double()

        count += y_true.numel()

    assert alpha0_checked, "alpha=0 sanity check did not run (empty loader?)"

    metrics = {}
    for a in alpha_list:
        denom = float(max(count, 1))
        mae = (sum_abs[a] / denom).item()
        mse = (sum_sq[a] / denom).item()
        metrics[a] = (float(mse), float(mae))
    return metrics


def _parse_csv_floats(s: str):
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(float(tok))
    return out


def _build_point_extractors(q_levels: torch.Tensor, point_q_list, method: str):
    # Returns a list of specs aligned with point_q_list.
    # spec: ("pick", idx, _) OR ("lerp", i0, i1, w)
    specs = []
    for pq in point_q_list:
        pq = float(pq)
        if method == "nearest":
            idx = int(torch.abs(q_levels - pq).argmin().item())
            specs.append(("pick", idx))
            continue
        if method != "linear":
            raise ValueError(f"Unknown point_quantile_method: {method}")

        q0 = float(q_levels[0].item())
        qn = float(q_levels[-1].item())
        if pq <= q0:
            specs.append(("pick", 0))
            continue
        if pq >= qn:
            specs.append(("pick", int(q_levels.numel() - 1)))
            continue

        ins = int(torch.searchsorted(q_levels, torch.tensor(pq, device=q_levels.device)).item())
        i1 = min(max(ins, 1), int(q_levels.numel() - 1))
        i0 = i1 - 1
        v0 = float(q_levels[i0].item())
        v1 = float(q_levels[i1].item())
        if v1 <= v0:
            specs.append(("pick", i0))
            continue
        w = (pq - v0) / (v1 - v0)
        # Avoid 0*NaN propagation when w==0 or w==1.
        if w <= 0.0:
            specs.append(("pick", i0))
            continue
        if w >= 1.0:
            specs.append(("pick", i1))
            continue
        specs.append(("lerp", i0, i1, float(w)))
    return specs


def _memory_forward_quantiles(model_base, model_mem, batch_x: torch.Tensor):
    with torch.no_grad():
        pred_len = int(args.pred_len)

        if getattr(model_mem, "requires_base_hidden", False):
            if batch_x.ndim != 2:
                raise ValueError("base_hidden_delta memory requires univariate (B,L) context")
            base_out = model_base.model(context=batch_x, return_hidden=True)
            q_base = base_out.quantile_preds.to(batch_x)  # (B,Q,T)
            hidden_source = getattr(model_mem, "config", None).hidden_source if hasattr(model_mem, "config") else "encoder"
            if hidden_source == "decoder":
                hidden = base_out.decoder_hidden_state
            else:
                hidden = base_out.encoder_hidden_states
            if hidden is None:
                raise ValueError(f"Missing hidden states from base model (hidden_source={hidden_source})")

            _loc, scale, is_const = instance_norm_stats(batch_x)
            delta = model_mem(hidden, scale=scale.squeeze(-1), is_constant=is_const.squeeze(-1))
            assert delta.shape == q_base.shape, f"shape mismatch: base={q_base.shape}, delta={delta.shape}"
            q_mem = q_base + delta
            return q_base, q_mem

        if batch_x.ndim == 2:
            q_base = model_base.predict(context=batch_x, prediction_length=pred_len)  # (B,Q,T)
            try:
                q_mem = model_mem(batch_x, pred_len=pred_len)
            except TypeError:
                q_mem = model_mem(batch_x)
            if q_mem.shape != q_base.shape:
                if q_mem.ndim != 3 or q_base.ndim != 3:
                    raise ValueError(f"Unexpected quantile shapes: base={q_base.shape} mem={q_mem.shape}")
                if q_mem.shape[:2] != q_base.shape[:2]:
                    raise ValueError(f"Batch/quantile mismatch: base={q_base.shape} mem={q_mem.shape}")
                if q_mem.shape[-1] < q_base.shape[-1]:
                    raise ValueError(f"memory horizon too short: baseT={q_base.shape[-1]} memT={q_mem.shape[-1]}")
                q_mem = q_mem[..., : q_base.shape[-1]]
            return q_base, q_mem

        if batch_x.ndim == 3:
            B, L, C = batch_x.shape
            x_flat = batch_x.permute(0, 2, 1).reshape(B * C, L)
            q_base_flat = model_base.predict(context=x_flat, prediction_length=pred_len)  # (B*C,Q,T)
            q_base = (
                q_base_flat.reshape(B, C, q_base_flat.shape[1], q_base_flat.shape[2]).permute(0, 2, 3, 1).contiguous()
            )
            try:
                q_mem = model_mem(batch_x, pred_len=pred_len)
            except TypeError:
                q_mem = model_mem(batch_x)
            if q_mem.shape != q_base.shape:
                if q_mem.ndim != 4 or q_base.ndim != 4:
                    raise ValueError(f"Unexpected quantile shapes: base={q_base.shape} mem={q_mem.shape}")
                if q_mem.shape[0] != q_base.shape[0] or q_mem.shape[1] != q_base.shape[1] or q_mem.shape[-1] != q_base.shape[-1]:
                    raise ValueError(f"Batch/quantile/channel mismatch: base={q_base.shape} mem={q_mem.shape}")
                if q_mem.shape[2] < q_base.shape[2]:
                    raise ValueError(f"memory horizon too short: baseT={q_base.shape[2]} memT={q_mem.shape[2]}")
                q_mem = q_mem[:, :, : q_base.shape[2], :]
            return q_base, q_mem

        raise ValueError(f"Unexpected batch_x shape: {batch_x.shape}")


def _benchmark_memory_speed(model_base, model_mem, loader, device, warmup_iters: int, num_iters: int):
    """
    Benchmark forward latency (ms/iter) for:
      - base quantile prediction
      - memory quantile prediction
      - combined (base + memory)

    Notes:
      - Measures only model forward time (batch already on device).
      - Uses torch.cuda.synchronize() for accurate timing on GPU.
    """

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    model_mem.eval()
    try:
        model_base.model.eval()
    except Exception:
        pass

    pred_len = int(args.pred_len)

    def _prep_batch(batch):
        batch_x, *_ = batch
        batch_x = batch_x.float().to(device)
        if batch_x.ndim == 3 and batch_x.shape[-1] == 1:
            batch_x = batch_x.squeeze(-1)
        return batch_x

    warmup_iters = max(int(warmup_iters), 0)
    num_iters = max(int(num_iters), 0)
    if num_iters <= 0:
        return None

    # Warmup (combined path).
    with torch.no_grad():
        seen = 0
        for batch in loader:
            batch_x = _prep_batch(batch)
            _memory_forward_quantiles(model_base, model_mem, batch_x)
            seen += 1
            if seen >= warmup_iters:
                break

    base_s = 0.0
    mem_s = 0.0
    iters = 0
    windows = 0

    with torch.no_grad():
        for batch in loader:
            if iters >= num_iters:
                break
            batch_x = _prep_batch(batch)

            if batch_x.ndim == 2:
                n_windows = int(batch_x.shape[0])
            elif batch_x.ndim == 3:
                # For (B,L,C) evaluation, Chronos base predicts over flattened B*C.
                n_windows = int(batch_x.shape[0] * batch_x.shape[2])
            else:
                n_windows = int(batch_x.shape[0])

            # Base forward
            _sync()
            t0 = time.perf_counter()
            hidden = None
            if getattr(model_mem, "requires_base_hidden", False):
                if batch_x.ndim != 2:
                    raise ValueError("base_hidden_delta memory requires univariate (B,L) context")
                base_out = model_base.model(context=batch_x, return_hidden=True)
                q_base = base_out.quantile_preds.to(batch_x)
                hidden_source = getattr(model_mem, "config", None).hidden_source if hasattr(model_mem, "config") else "encoder"
                hidden = base_out.decoder_hidden_state if hidden_source == "decoder" else base_out.encoder_hidden_states
                if hidden is None:
                    raise ValueError(f"Missing hidden states from base model (hidden_source={hidden_source})")
            elif batch_x.ndim == 2:
                q_base = model_base.predict(context=batch_x, prediction_length=pred_len)
            elif batch_x.ndim == 3:
                B, L, C = batch_x.shape
                x_flat = batch_x.permute(0, 2, 1).reshape(B * C, L)
                q_base_flat = model_base.predict(context=x_flat, prediction_length=pred_len)
                q_base = (
                    q_base_flat.reshape(B, C, q_base_flat.shape[1], q_base_flat.shape[2]).permute(0, 2, 3, 1).contiguous()
                )
            else:
                raise ValueError(f"Unexpected batch_x shape: {batch_x.shape}")
            _sync()
            t1 = time.perf_counter()
            base_s += float(t1 - t0)

            # Memory forward (quantiles)
            _sync()
            t0 = time.perf_counter()
            if getattr(model_mem, "requires_base_hidden", False):
                _loc, scale, is_const = instance_norm_stats(batch_x)
                delta = model_mem(hidden, scale=scale.squeeze(-1), is_constant=is_const.squeeze(-1))
                q_mem = q_base + delta
                _ = q_mem
            else:
                try:
                    q_mem = model_mem(batch_x, pred_len=pred_len)
                except TypeError:
                    q_mem = model_mem(batch_x)
                # Match base horizon shape (slice if memory emits a longer max_pred_len).
                if q_mem.shape != q_base.shape:
                    if batch_x.ndim == 2:
                        q_mem = q_mem[..., : q_base.shape[-1]]
                    else:
                        q_mem = q_mem[:, :, : q_base.shape[2], :]
                _ = q_mem
            _sync()
            t1 = time.perf_counter()
            mem_s += float(t1 - t0)

            iters += 1
            windows += n_windows

    if iters <= 0:
        return None

    base_ms_iter = 1000.0 * base_s / float(iters)
    mem_ms_iter = 1000.0 * mem_s / float(iters)
    total_ms_iter = base_ms_iter + mem_ms_iter

    windows = max(int(windows), 1)
    base_ms_win = 1000.0 * base_s / float(windows)
    mem_ms_win = 1000.0 * mem_s / float(windows)
    total_ms_win = base_ms_win + mem_ms_win

    return {
        "split": str(getattr(args, "speed_split", "test")),
        "warmup_iters": int(warmup_iters),
        "num_iters": int(iters),
        "batch_size": int(getattr(args, "batch_size", 0)),
        "windows": int(windows),
        "base_ms_per_iter": float(base_ms_iter),
        "mem_ms_per_iter": float(mem_ms_iter),
        "total_ms_per_iter": float(total_ms_iter),
        "base_ms_per_window": float(base_ms_win),
        "mem_ms_per_window": float(mem_ms_win),
        "total_ms_per_window": float(total_ms_win),
        "device": str(device),
    }


def _memory_point_pred_from_quantiles(q_preds: torch.Tensor, spec):
    if spec[0] == "pick":
        idx = int(spec[1])
        return q_preds[:, idx, ...]
    _, i0, i1, w = spec
    p0 = q_preds[:, int(i0), ...]
    p1 = q_preds[:, int(i1), ...]
    return (1.0 - float(w)) * p0 + float(w) * p1


def _memory_fuse_point(p_base: torch.Tensor, p_mem: torch.Tensor, alpha: float):
    a = float(alpha)
    if a <= 0.0:
        return p_base
    if a >= 1.0:
        return p_mem
    return p_base + a * (p_mem - p_base)


def _memory_fuse_quantiles(q_base: torch.Tensor, q_mem: torch.Tensor, alpha: float):
    a = float(alpha)
    if a <= 0.0:
        return q_base
    if a >= 1.0:
        return q_mem
    return q_base + a * (q_mem - q_base)


def _memory_fuse_quantiles_2segment(
    q_base: torch.Tensor, q_mem: torch.Tensor, alpha_short: float, alpha_long: float, split: int
):
    split = int(split)
    t = int(q_base.shape[2])
    if split <= 0:
        return _memory_fuse_quantiles(q_base, q_mem, float(alpha_long))
    if split >= t:
        return _memory_fuse_quantiles(q_base, q_mem, float(alpha_short))

    q0_b = q_base[:, :, :split, ...]
    q0_m = q_mem[:, :, :split, ...]
    q1_b = q_base[:, :, split:, ...]
    q1_m = q_mem[:, :, split:, ...]
    out0 = _memory_fuse_quantiles(q0_b, q0_m, float(alpha_short))
    out1 = _memory_fuse_quantiles(q1_b, q1_m, float(alpha_long))
    return torch.cat([out0, out1], dim=2)


def _memory_fuse_point_2segment(
    p_base: torch.Tensor, p_mem: torch.Tensor, alpha_short: float, alpha_long: float, split: int
):
    split = int(split)
    if split <= 0:
        return _memory_fuse_point(p_base, p_mem, float(alpha_long))
    if split >= int(p_base.shape[1]):
        return _memory_fuse_point(p_base, p_mem, float(alpha_short))

    p0_b = p_base[:, :split, ...]
    p0_m = p_mem[:, :split, ...]
    p1_b = p_base[:, split:, ...]
    p1_m = p_mem[:, split:, ...]
    out0 = _memory_fuse_point(p0_b, p0_m, float(alpha_short))
    out1 = _memory_fuse_point(p1_b, p1_m, float(alpha_long))
    return torch.cat([out0, out1], dim=1)


def _memory_eval_grid(
    model_base,
    model_mem,
    loader,
    quantiles,
    device,
    alpha_list,
    point_q_list,
    point_method: str,
):
    q_levels = torch.tensor(quantiles, dtype=torch.float32, device=device)

    alpha_list = [float(a) for a in alpha_list]
    point_q_list = [float(pq) for pq in point_q_list]

    extract_specs = _build_point_extractors(q_levels, point_q_list, point_method)

    keys = [(a, pq) for a in alpha_list for pq in point_q_list]
    sum_abs = {k: torch.zeros((), device=device, dtype=torch.float64) for k in keys}
    sum_sq = {k: torch.zeros((), device=device, dtype=torch.float64) for k in keys}
    count = 0
    alpha0_checked = False

    for batch in loader:
        batch_x, batch_y, *_ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if batch_x.ndim == 3 and batch_x.shape[-1] == 1:
            batch_x = batch_x.squeeze(-1)
        if batch_y.ndim == 3 and batch_y.shape[-1] == 1:
            batch_y = batch_y.squeeze(-1)
        y_true = batch_y[:, -args.pred_len:]

        q_base, q_mem = _memory_forward_quantiles(model_base, model_mem, batch_x)
        assert q_base.shape == q_mem.shape, f"shape mismatch: base={q_base.shape}, mem={q_mem.shape}"

        # Sanity: alpha=0 must exactly match base without touching q_mem (avoid 0*NaN propagation).
        if not alpha0_checked:
            q0 = q_base
            assert q0.data_ptr() == q_base.data_ptr(), "alpha=0 should directly use q_base tensor"
            finite = torch.isfinite(q_base)
            if finite.any():
                max_diff0 = (q0[finite] - q_base[finite]).abs().max().item()
                assert max_diff0 < 1e-7, f"alpha=0 sanity failed: max_diff={max_diff0}"
            alpha0_checked = True

        for a in alpha_list:
            q_fused = _memory_fuse_quantiles(q_base, q_mem, float(a))
            for pq, spec in zip(point_q_list, extract_specs):
                key = (a, pq)
                pred = _memory_point_pred_from_quantiles(q_fused, spec)
                err = pred - y_true
                sum_abs[key] += err.abs().sum().double()
                sum_sq[key] += (err ** 2).sum().double()

        count += y_true.numel()

    assert alpha0_checked, "alpha=0 sanity check did not run (empty loader?)"

    metrics = {a: {} for a in alpha_list}
    denom = float(max(count, 1))
    for a in alpha_list:
        for pq in point_q_list:
            key = (a, pq)
            mae = (sum_abs[key] / denom).item()
            mse = (sum_sq[key] / denom).item()
            metrics[a][pq] = (float(mse), float(mae))
    return metrics


def _memory_eval_grid_2segment(
    model_base,
    model_mem,
    loader,
    quantiles,
    device,
    alpha_list,
    point_q_list,
    point_method: str,
    alpha_split: int,
):
    q_levels = torch.tensor(quantiles, dtype=torch.float32, device=device)

    alpha_list = [float(a) for a in alpha_list]
    point_q_list = [float(pq) for pq in point_q_list]
    cfg_list = [(a_s, a_l) for a_s in alpha_list for a_l in alpha_list]

    extract_specs = _build_point_extractors(q_levels, point_q_list, point_method)

    keys = [(a_s, a_l, pq) for (a_s, a_l) in cfg_list for pq in point_q_list]
    sum_abs = {k: torch.zeros((), device=device, dtype=torch.float64) for k in keys}
    sum_sq = {k: torch.zeros((), device=device, dtype=torch.float64) for k in keys}
    count = 0
    alpha0_checked = False

    for batch in loader:
        batch_x, batch_y, *_ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if batch_x.ndim == 3 and batch_x.shape[-1] == 1:
            batch_x = batch_x.squeeze(-1)
        if batch_y.ndim == 3 and batch_y.shape[-1] == 1:
            batch_y = batch_y.squeeze(-1)
        y_true = batch_y[:, -args.pred_len:]

        q_base, q_mem = _memory_forward_quantiles(model_base, model_mem, batch_x)
        assert q_base.shape == q_mem.shape, f"shape mismatch: base={q_base.shape}, mem={q_mem.shape}"

        if not alpha0_checked:
            q0 = q_base
            assert q0.data_ptr() == q_base.data_ptr(), "alpha=0 should directly use q_base tensor"
            finite = torch.isfinite(q_base)
            if finite.any():
                max_diff0 = (q0[finite] - q_base[finite]).abs().max().item()
                assert max_diff0 < 1e-7, f"alpha=0 sanity failed: max_diff={max_diff0}"
            alpha0_checked = True

        for (a_s, a_l) in cfg_list:
            q_fused = _memory_fuse_quantiles_2segment(q_base, q_mem, float(a_s), float(a_l), int(alpha_split))
            for pq, spec in zip(point_q_list, extract_specs):
                key = (a_s, a_l, pq)
                pred = _memory_point_pred_from_quantiles(q_fused, spec)
                err = pred - y_true
                sum_abs[key] += err.abs().sum().double()
                sum_sq[key] += (err ** 2).sum().double()

        count += y_true.numel()

    assert alpha0_checked, "alpha=0 sanity check did not run (empty loader?)"

    metrics = {cfg: {} for cfg in cfg_list}
    denom = float(max(count, 1))
    for (a_s, a_l) in cfg_list:
        for pq in point_q_list:
            key = (a_s, a_l, pq)
            mae = (sum_abs[key] / denom).item()
            mse = (sum_sq[key] / denom).item()
            metrics[(a_s, a_l)][pq] = (float(mse), float(mae))
    return metrics


def _estimate_bias_from_val(
    model_base,
    model_mem,
    loader,
    quantiles,
    device,
    alpha_mode: str,
    alpha: float,
    alpha_short: float,
    alpha_long: float,
    alpha_split: int,
    point_q: float,
    point_method: str,
    bias_mode: str,
    max_windows: int,
):
    if bias_mode not in {"global", "horizon", "horizon_shrink_smooth"}:
        raise ValueError(f"Unknown bias_correct mode: {bias_mode}")

    q_levels = torch.tensor(quantiles, dtype=torch.float32, device=device)
    spec = _build_point_extractors(q_levels, [float(point_q)], point_method)[0]

    used = 0
    chunks = []
    for batch in loader:
        batch_x, batch_y, *_ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if batch_x.ndim == 3 and batch_x.shape[-1] == 1:
            batch_x = batch_x.squeeze(-1)
        if batch_y.ndim == 3 and batch_y.shape[-1] == 1:
            batch_y = batch_y.squeeze(-1)
        y_true = batch_y[:, -args.pred_len:]

        q_base, q_mem = _memory_forward_quantiles(model_base, model_mem, batch_x)
        if alpha_mode == "scalar":
            q_fused = _memory_fuse_quantiles(q_base, q_mem, float(alpha))
        elif alpha_mode == "2segment":
            q_fused = _memory_fuse_quantiles_2segment(
                q_base, q_mem, float(alpha_short), float(alpha_long), int(alpha_split)
            )
        else:
            raise ValueError(f"Unknown alpha_mode: {alpha_mode}")
        pred = _memory_point_pred_from_quantiles(q_fused, spec)

        resid = (y_true - pred).detach().cpu()
        if max_windows and max_windows > 0:
            remain = max_windows - used
            if remain <= 0:
                break
            if resid.shape[0] > remain:
                resid = resid[:remain]
        used += resid.shape[0]

        # Bias correction is defined on point forecasts; for horizon-wise median we require (N,T).
        if bias_mode in {"horizon", "horizon_shrink_smooth"} and resid.ndim != 2:
            raise ValueError(
                f"bias_correct=horizon requires univariate point forecasts (got resid shape {tuple(resid.shape)}). "
                "Use split-features mode or choose --bias_correct global."
            )

        chunks.append(resid)
        if max_windows and max_windows > 0 and used >= max_windows:
            break

    if not chunks:
        raise ValueError("Empty val loader while estimating bias")

    resid_all = torch.cat(chunks, dim=0)
    if bias_mode == "global":
        bias = resid_all.reshape(-1).median().item()
        return torch.tensor(bias, dtype=torch.float32, device=device), used

    # horizon-wise (T,)
    bias_vec = resid_all.median(dim=0).values.to(dtype=torch.float32)
    if bias_mode == "horizon":
        return bias_vec.to(device=device), used

    # horizon_shrink_smooth: smooth over horizons + shrink toward zero (to reduce overfitting).
    window = int(getattr(args, "bias_correct_smooth_window", 1))
    if window <= 1:
        smooth = bias_vec
    else:
        if window % 2 == 0:
            window += 1
            print(f"[bias_correct] bias_correct_smooth_window must be odd; using {window}")
        pad = window // 2
        x = bias_vec.reshape(1, 1, -1)
        x = F.pad(x, (pad, pad), mode="replicate")
        k = torch.ones((1, 1, window), dtype=x.dtype) / float(window)
        smooth = F.conv1d(x, k).reshape(-1)

    shrink_lambda = float(getattr(args, "bias_correct_shrink_lambda", 0.0))
    n = float(resid_all.shape[0])
    if shrink_lambda <= 0.0:
        shrink = 1.0
    else:
        shrink = n / (n + shrink_lambda)
    bias_out = smooth * float(shrink)
    return bias_out.to(device=device), used


def _estimate_bias_from_val_multi_alpha_scalar(
    model_base,
    model_mem,
    loader,
    quantiles,
    device,
    alpha_list,
    point_q: float,
    point_method: str,
    bias_mode: str,
    max_windows: int,
):
    alpha_list = [float(a) for a in alpha_list]
    if bias_mode not in {"global", "horizon", "horizon_shrink_smooth"}:
        raise ValueError(f"Unknown bias_correct mode: {bias_mode}")

    q_levels = torch.tensor(quantiles, dtype=torch.float32, device=device)
    spec = _build_point_extractors(q_levels, [float(point_q)], point_method)[0]

    used = 0
    chunks_by_alpha = {float(a): [] for a in alpha_list}
    for batch in loader:
        batch_x, batch_y, *_ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if batch_x.ndim == 3 and batch_x.shape[-1] == 1:
            batch_x = batch_x.squeeze(-1)
        if batch_y.ndim == 3 and batch_y.shape[-1] == 1:
            batch_y = batch_y.squeeze(-1)
        y_true = batch_y[:, -args.pred_len:]

        q_base, q_mem = _memory_forward_quantiles(model_base, model_mem, batch_x)

        if max_windows and max_windows > 0:
            remain = max_windows - used
            if remain <= 0:
                break
            if int(y_true.shape[0]) > int(remain):
                y_true = y_true[:remain]
                q_base = q_base[:remain]
                q_mem = q_mem[:remain]

        used += int(y_true.shape[0])

        for a in alpha_list:
            q_fused = _memory_fuse_quantiles(q_base, q_mem, float(a))
            pred = _memory_point_pred_from_quantiles(q_fused, spec)
            resid = (y_true - pred).detach().cpu()

            if bias_mode in {"horizon", "horizon_shrink_smooth"} and resid.ndim != 2:
                raise ValueError(
                    f"bias_correct=horizon requires univariate point forecasts (got resid shape {tuple(resid.shape)}). "
                    "Use split-features mode or choose --bias_correct global."
                )

            chunks_by_alpha[float(a)].append(resid)

        if max_windows and max_windows > 0 and used >= max_windows:
            break

    if used <= 0:
        raise ValueError("Empty val loader while estimating bias")

    def _postprocess_horizon_bias(bias_vec: torch.Tensor) -> torch.Tensor:
        if bias_mode == "horizon":
            return bias_vec

        # horizon_shrink_smooth: smooth over horizons + shrink toward zero (to reduce overfitting).
        window = int(getattr(args, "bias_correct_smooth_window", 1))
        if window <= 1:
            smooth = bias_vec
        else:
            if window % 2 == 0:
                window += 1
                print(f"[bias_correct] bias_correct_smooth_window must be odd; using {window}")
            pad = window // 2
            x = bias_vec.reshape(1, 1, -1)
            x = F.pad(x, (pad, pad), mode="replicate")
            k = torch.ones((1, 1, window), dtype=x.dtype) / float(window)
            smooth = F.conv1d(x, k).reshape(-1)

        shrink_lambda = float(getattr(args, "bias_correct_shrink_lambda", 0.0))
        n = float(used)
        if shrink_lambda <= 0.0:
            shrink = 1.0
        else:
            shrink = n / (n + shrink_lambda)
        return smooth * float(shrink)

    biases = {}
    for a in alpha_list:
        chunks = chunks_by_alpha[float(a)]
        if not chunks:
            raise ValueError(f"Empty val chunks for alpha={a}")
        resid_all = torch.cat(chunks, dim=0)
        if bias_mode == "global":
            bias = resid_all.reshape(-1).median().item()
            biases[float(a)] = torch.tensor(bias, dtype=torch.float32, device=device)
            continue
        bias_vec = resid_all.median(dim=0).values.to(dtype=torch.float32)
        biases[float(a)] = _postprocess_horizon_bias(bias_vec).to(device=device)

    return biases, used


def _memory_eval_grid_bias_corrected_scalar(
    model_base,
    model_mem,
    loader,
    quantiles,
    device,
    alpha_list,
    point_q: float,
    point_method: str,
    biases_by_alpha,
):
    alpha_list = [float(a) for a in alpha_list]
    q_levels = torch.tensor(quantiles, dtype=torch.float32, device=device)
    spec = _build_point_extractors(q_levels, [float(point_q)], point_method)[0]

    sum_abs = {float(a): torch.zeros((), device=device, dtype=torch.float64) for a in alpha_list}
    sum_sq = {float(a): torch.zeros((), device=device, dtype=torch.float64) for a in alpha_list}
    count = 0
    for batch in loader:
        batch_x, batch_y, *_ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if batch_x.ndim == 3 and batch_x.shape[-1] == 1:
            batch_x = batch_x.squeeze(-1)
        if batch_y.ndim == 3 and batch_y.shape[-1] == 1:
            batch_y = batch_y.squeeze(-1)
        y_true = batch_y[:, -args.pred_len:]

        q_base, q_mem = _memory_forward_quantiles(model_base, model_mem, batch_x)
        for a in alpha_list:
            bias = biases_by_alpha.get(float(a), None)
            if bias is None:
                raise ValueError(f"Missing bias for alpha={a}")
            q_fused = _memory_fuse_quantiles(q_base, q_mem, float(a))
            pred = _memory_point_pred_from_quantiles(q_fused, spec)
            if isinstance(bias, torch.Tensor):
                pred = pred + bias
            else:
                pred = pred + float(bias)
            err = pred - y_true
            sum_abs[float(a)] += err.abs().sum().double()
            sum_sq[float(a)] += (err ** 2).sum().double()
        count += y_true.numel()

    denom = float(max(count, 1))
    metrics = {}
    for a in alpha_list:
        mae = (sum_abs[float(a)] / denom).item()
        mse = (sum_sq[float(a)] / denom).item()
        metrics[float(a)] = (float(mse), float(mae))
    return metrics


def _memory_eval_one(
    model_base,
    model_mem,
    loader,
    quantiles,
    device,
    alpha_mode: str,
    alpha: float,
    alpha_short: float,
    alpha_long: float,
    alpha_split: int,
    point_q: float,
    point_method: str,
    bias=None,
):
    q_levels = torch.tensor(quantiles, dtype=torch.float32, device=device)
    spec = _build_point_extractors(q_levels, [float(point_q)], point_method)[0]

    sum_abs = torch.zeros((), device=device, dtype=torch.float64)
    sum_sq = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    for batch in loader:
        batch_x, batch_y, *_ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if batch_x.ndim == 3 and batch_x.shape[-1] == 1:
            batch_x = batch_x.squeeze(-1)
        if batch_y.ndim == 3 and batch_y.shape[-1] == 1:
            batch_y = batch_y.squeeze(-1)
        y_true = batch_y[:, -args.pred_len:]

        q_base, q_mem = _memory_forward_quantiles(model_base, model_mem, batch_x)
        if alpha_mode == "scalar":
            q_fused = _memory_fuse_quantiles(q_base, q_mem, float(alpha))
        elif alpha_mode == "2segment":
            q_fused = _memory_fuse_quantiles_2segment(
                q_base, q_mem, float(alpha_short), float(alpha_long), int(alpha_split)
            )
        else:
            raise ValueError(f"Unknown alpha_mode: {alpha_mode}")
        pred = _memory_point_pred_from_quantiles(q_fused, spec)
        if bias is not None:
            pred = pred + bias
        err = pred - y_true
        sum_abs += err.abs().sum().double()
        sum_sq += (err ** 2).sum().double()
        count += err.numel()

    denom = float(max(count, 1))
    mae = (sum_abs / denom).item()
    mse = (sum_sq / denom).item()
    return float(mse), float(mae)


run_mode = args.run_mode
if run_mode == 'auto':
    run_mode = 'rag' if 'retrieve' in args.model_id else 'base'

already_evaluated = False

# Optional report for memory alpha search (written to result file).
alpha_search_report = None  # dict with keys: alpha_list, best_alpha, val_metrics, test_metrics
# Optional report for memory speed benchmark (written to result file).
speed_report = None  # dict with keys: split, base_ms_per_iter, mem_ms_per_iter, ...


if run_mode == 'memory_quantile':
    print("=== NO RETRIEVAL: memory_quantile mode ===")
    assert 'faiss' not in sys.modules, "faiss imported in memory_quantile mode"
    assert 'retrieve' not in sys.modules, "retrieve module imported in memory_quantile mode"
    if "_retrieve_" in os.path.basename(args.data_path):
        raise ValueError("memory_quantile mode forbids using *_retrieve_*.csv; pass the raw dataset CSV instead.")

    if args.memory_ckpt is None:
        raise ValueError("--memory_ckpt is required for memory_quantile mode")
    if args.model not in {'ChronosBolt', 'ChronosBoltRetrieve'}:
        raise ValueError("memory_quantile currently supports ChronosBolt backbone only")

    # Load memory ckpt early to decide dataloader feature mode (split vs multivariate).
    memory_ckpt = torch.load(args.memory_ckpt, map_location=map_location)
    memory_type = str(memory_ckpt.get("memory_type", memory_ckpt.get("memdec_type", "context")))

    # Force non-retrieval dataset loader
    args_loader = copy.deepcopy(args)
    args_loader.model_id = args_loader.model_id.replace('retrieve', 'memory')
    if args_loader.data.endswith('_retrieve'):
        args_loader.data = args_loader.data.replace('_retrieve', '')
    # Multivariate memory expects (B,L,C) windows (no per-feature split).
    args_loader.split_features = (memory_type != "context_multivar")

    val_data, val_loader = data_provider(args_loader, 'val')
    test_data, test_loader = data_provider(args_loader, 'test')
    assert 'faiss' not in sys.modules, "faiss imported while building non-retrieval dataloaders"
    assert 'retrieve' not in sys.modules, "retrieve imported while building non-retrieval dataloaders"

    # Load frozen base model (Chronos-Bolt).
    # Some checkpoints include an optional `autogluon_model.pth` weight override; if missing, we fall back to
    # the weights loaded by from_pretrained.
    model_base = ChronosBoltPipeline.from_pretrained(args.pretrained_model_path)
    autogluon_state_path = os.path.join(args.pretrained_model_path, "autogluon_model.pth")
    if os.path.isfile(autogluon_state_path):
        model_base.model.load_state_dict(torch.load(autogluon_state_path, map_location=map_location))
    else:
        print(f"[warn] autogluon_model.pth not found at {autogluon_state_path}; using HF weights from {args.pretrained_model_path}")
    model_base.model.to(torch.device(device_address))
    assert 'faiss' not in sys.modules, "faiss imported while loading base model"
    assert 'retrieve' not in sys.modules, "retrieve imported while loading base model"

    quantiles = model_base.quantiles

    # Load memory
    mem_quantiles = memory_ckpt.get("quantiles", None)
    if mem_quantiles is None:
        raise ValueError("memory_ckpt missing 'quantiles'")
    if list(map(float, mem_quantiles)) != list(map(float, quantiles)):
        raise ValueError(f"Quantiles mismatch: base={quantiles} memory={mem_quantiles}")

    if memory_type == "context":
        mem_cfg = MemoryTSQuantileConfig(**memory_ckpt["model_config"])
        model_mem = MemoryTSQuantile(mem_cfg)
        model_mem.load_state_dict(memory_ckpt["state_dict"], strict=True)
    elif memory_type == "context_multivar":
        mem_cfg = MemoryTSQuantileMultivarConfig(**memory_ckpt["model_config"])
        model_mem = MemoryTSQuantileMultivar(mem_cfg)
        model_mem.load_state_dict(memory_ckpt["state_dict"], strict=True)
    elif memory_type == "base_hidden_delta":
        # Hidden-based memory currently assumes pred_len == base model horizon (typically 64)
        base_pred_len = int(model_base.model.config.chronos_config["prediction_length"])
        if int(args.pred_len) != base_pred_len:
            raise ValueError(
                f"memory_type=base_hidden_delta requires pred_len == base prediction_length "
                f"(pred_len={args.pred_len}, base={base_pred_len})"
            )
        mem_cfg = MemoryTSHiddenDeltaConfig(**memory_ckpt["model_config"])
        model_mem = MemoryTSHiddenDelta(mem_cfg)
        model_mem.load_state_dict(memory_ckpt["state_dict"], strict=True)
    else:
        raise ValueError(f"Unknown memory_type in ckpt: {memory_type}")
    model_mem.to(torch.device(device_address))
    model_mem.eval()
    assert 'faiss' not in sys.modules, "faiss imported while loading memory model"
    assert 'retrieve' not in sys.modules, "retrieve imported while loading memory model"

    # Optional speed benchmark (base vs memory forward). Runs before alpha/grid evaluation.
    if int(getattr(args, "report_speed", 0)) == 1:
        speed_loader = test_loader if str(getattr(args, "speed_split", "test")) == "test" else val_loader
        speed_report = _benchmark_memory_speed(
            model_base,
            model_mem,
            speed_loader,
            torch.device(device_address),
            warmup_iters=int(getattr(args, "speed_warmup_iters", 5)),
            num_iters=int(getattr(args, "speed_num_iters", 20)),
        )
        if speed_report is not None:
            print(
                "[speed] split={} iters={} warmup={} batch_size={} windows={} base_ms/iter={:.3f} mem_ms/iter={:.3f} total_ms/iter={:.3f} base_ms/win={:.4f} mem_ms/win={:.4f} total_ms/win={:.4f}".format(
                    speed_report["split"],
                    speed_report["num_iters"],
                    speed_report["warmup_iters"],
                    speed_report["batch_size"],
                    speed_report["windows"],
                    speed_report["base_ms_per_iter"],
                    speed_report["mem_ms_per_iter"],
                    speed_report["total_ms_per_iter"],
                    speed_report["base_ms_per_window"],
                    speed_report["mem_ms_per_window"],
                    speed_report["total_ms_per_window"],
                )
            )

    # Alpha search on val, then test
    if args.alpha_search:
        alpha_list = [float(x) for x in args.alpha_search.split(",") if x.strip() != ""]
    else:
        alpha_list = [float(args.alpha)]

    alpha_list_all = alpha_list

    if args.point_quantile_search:
        point_q_list = _parse_csv_floats(args.point_quantile_search)
        point_method = str(args.point_quantile_method)
        if point_method == "nearest":
            print("[point_quantile_search] point_quantile_method=nearest is degenerate; switching to linear interpolation.")
            point_method = "linear"
    else:
        point_q_list = [float(args.point_quantile)]
        point_method = str(args.point_quantile_method)

    alpha_mode = str(getattr(args, "alpha_mode", "scalar"))
    alpha_split = int(getattr(args, "alpha_split", 16))
    metric = str(args.alpha_search_metric)
    search_enabled = bool(args.alpha_search or args.point_quantile_search)
    select_split = str(getattr(args, "alpha_search_select_split", "val"))
    bias_split = str(getattr(args, "bias_correct_split", "val"))
    if select_split == "test":
        print("[warn] alpha_search_select_split=test uses test labels for selection (leaky).")
    if bias_split == "test":
        print("[warn] bias_correct_split=test uses test labels for bias estimation (leaky).")
    if select_split == "test" and bias_split != "test" and str(args.bias_correct) != "none":
        print("[warn] alpha_search_select_split=test but bias_correct_split!=test; bias still estimated on val (not full oracle).")

    if alpha_mode == "scalar":
        val_grid = _memory_eval_grid(
            model_base,
            model_mem,
            val_loader,
            quantiles,
            torch.device(device_address),
            alpha_list_all,
            point_q_list,
            point_method,
        )
        test_grid = _memory_eval_grid(
            model_base,
            model_mem,
            test_loader,
            quantiles,
            torch.device(device_address),
            alpha_list_all,
            point_q_list,
            point_method,
        )
        select_grid_raw = test_grid if select_split == "test" else val_grid
        bias_select_mode = str(getattr(args, "bias_correct_select_mode", "raw"))
        if str(args.bias_correct) == "none":
            bias_select_mode = "raw"
        if bias_select_mode != "raw" and len(point_q_list) != 1:
            raise ValueError(
                "bias_correct_select_mode != raw currently requires point_quantile_search to be disabled "
                "(use a single --point_quantile)."
            )

        def _metric_key(mse_a: float, mae_a: float):
            if metric == "mse_then_mae":
                return (float(mse_a), float(mae_a))
            if metric == "mae_then_mse":
                return (float(mae_a), float(mse_a))
            if metric == "mse":
                return (float(mse_a),)
            if metric == "mae":
                return (float(mae_a),)
            if metric in {"sum", "combo"}:
                return (float(mse_a) + float(args.alpha_search_mae_weight) * float(mae_a),)
            if metric == "mae_guard":
                # For cross-mode comparisons: MAE-first, then MSE.
                return (float(mae_a), float(mse_a))
            raise ValueError(f"Unknown alpha_search_metric: {metric}")

        def _select_best(select_grid):
            if not search_enabled:
                return float(alpha_list_all[0]), float(point_q_list[0])
            candidates = [(float(a), float(pq)) for a in alpha_list_all for pq in point_q_list]
            if metric == "mae_guard":
                guard = float(args.alpha_search_mse_guard_ratio)
                if guard < 0.0:
                    raise ValueError(f"alpha_search_mse_guard_ratio must be >=0, got {guard}")
                best_mse_only = min(float(select_grid[a][pq][0]) for (a, pq) in candidates)
                thresh = best_mse_only * (1.0 + guard)
                allowed = [(a, pq) for (a, pq) in candidates if float(select_grid[a][pq][0]) <= thresh]
                if not allowed:
                    allowed = candidates
                best_a, best_p = min(
                    allowed,
                    key=lambda cfg: (
                        float(select_grid[cfg[0]][cfg[1]][1]),
                        float(select_grid[cfg[0]][cfg[1]][0]),
                    ),
                )
                return float(best_a), float(best_p)
            return min(candidates, key=lambda cfg: _metric_key(*select_grid[cfg[0]][cfg[1]]))

        best_alpha_raw = best_pq_raw = None
        best_alpha_bias = best_pq_bias = None
        bias_selected = False
        biases_by_alpha = None
        bias_used = None
        test_metrics_bc = None

        best_alpha_raw, best_pq_raw = _select_best(select_grid_raw)
        sel_mse, sel_mae = select_grid_raw[best_alpha_raw][best_pq_raw]
        print(
            f"[alpha_point_search] select_split={select_split} metric={metric} best_alpha={best_alpha_raw} best_point_q={best_pq_raw} "
            f"select_mse={sel_mse:.6f} select_mae={sel_mae:.6f} (bias_select_mode=raw)"
        )

        best_alpha = float(best_alpha_raw)
        best_pq = float(best_pq_raw)

        if str(args.bias_correct) != "none" and bias_select_mode in {"bias", "auto"}:
            bias_loader = test_loader if bias_split == "test" else val_loader
            pq_fixed = float(point_q_list[0])
            biases_by_alpha, bias_used = _estimate_bias_from_val_multi_alpha_scalar(
                model_base,
                model_mem,
                bias_loader,
                quantiles,
                torch.device(device_address),
                alpha_list_all,
                point_q=pq_fixed,
                point_method=str(point_method),
                bias_mode=str(args.bias_correct),
                max_windows=int(args.bias_correct_max_windows),
            )
            select_loader = test_loader if select_split == "test" else val_loader
            select_metrics_bc = _memory_eval_grid_bias_corrected_scalar(
                model_base,
                model_mem,
                select_loader,
                quantiles,
                torch.device(device_address),
                alpha_list_all,
                point_q=pq_fixed,
                point_method=str(point_method),
                biases_by_alpha=biases_by_alpha,
            )
            select_grid_bias = {float(a): {pq_fixed: select_metrics_bc[float(a)]} for a in alpha_list_all}
            best_alpha_bias, best_pq_bias = _select_best(select_grid_bias)
            sel_mse_b, sel_mae_b = select_grid_bias[best_alpha_bias][best_pq_bias]
            print(
                f"[alpha_point_search] select_split={select_split} metric={metric} best_alpha={best_alpha_bias} best_point_q={best_pq_bias} "
                f"select_mse={sel_mse_b:.6f} select_mae={sel_mae_b:.6f} (bias_select_mode=bias)"
            )
            if bias_select_mode == "bias":
                best_alpha = float(best_alpha_bias)
                best_pq = float(best_pq_bias)
                bias_selected = True
            else:
                raw_key = _metric_key(*select_grid_raw[best_alpha_raw][best_pq_raw])
                bias_key = _metric_key(*select_grid_bias[best_alpha_bias][best_pq_bias])
                if bias_key < raw_key:
                    best_alpha = float(best_alpha_bias)
                    best_pq = float(best_pq_bias)
                    bias_selected = True
                print(
                    f"[alpha_point_search] bias_select_mode=auto chose={'bias' if bias_selected else 'raw'} "
                    f"best_alpha={best_alpha} best_point_q={best_pq}"
                )
        raw_mse, raw_mae = test_grid[best_alpha][best_pq]
        mse, mae = raw_mse, raw_mae
        bias = None
        if str(args.bias_correct) != "none":
            report_grid = bool(int(getattr(args, "bias_correct_report_grid", 0) or 0))
            bias_loader = test_loader if bias_split == "test" else val_loader
            apply_bias = True
            if bias_select_mode == "auto":
                apply_bias = bool(bias_selected)
            if biases_by_alpha is None and (report_grid or apply_bias):
                biases_by_alpha, bias_used = _estimate_bias_from_val_multi_alpha_scalar(
                    model_base,
                    model_mem,
                    bias_loader,
                    quantiles,
                    torch.device(device_address),
                    alpha_list_all,
                    point_q=float(best_pq),
                    point_method=str(point_method),
                    bias_mode=str(args.bias_correct),
                    max_windows=int(args.bias_correct_max_windows),
                )
            if report_grid and biases_by_alpha is not None:
                test_metrics_bc = _memory_eval_grid_bias_corrected_scalar(
                    model_base,
                    model_mem,
                    test_loader,
                    quantiles,
                    torch.device(device_address),
                    alpha_list_all,
                    point_q=float(best_pq),
                    point_method=str(point_method),
                    biases_by_alpha=biases_by_alpha,
                )
            if apply_bias:
                if report_grid and test_metrics_bc is not None:
                    bias = biases_by_alpha.get(float(best_alpha), None) if biases_by_alpha is not None else None
                    mse, mae = test_metrics_bc[float(best_alpha)]
                else:
                    if biases_by_alpha is None:
                        bias, bias_used = _estimate_bias_from_val(
                            model_base,
                            model_mem,
                            bias_loader,
                            quantiles,
                            torch.device(device_address),
                            alpha_mode="scalar",
                            alpha=float(best_alpha),
                            alpha_short=0.0,
                            alpha_long=0.0,
                            alpha_split=int(alpha_split),
                            point_q=float(best_pq),
                            point_method=str(point_method),
                            bias_mode=str(args.bias_correct),
                            max_windows=int(args.bias_correct_max_windows),
                        )
                    else:
                        bias = biases_by_alpha.get(float(best_alpha), None)
                    mse, mae = _memory_eval_one(
                        model_base,
                        model_mem,
                        test_loader,
                        quantiles,
                        torch.device(device_address),
                        alpha_mode="scalar",
                        alpha=float(best_alpha),
                        alpha_short=0.0,
                        alpha_long=0.0,
                        alpha_split=int(alpha_split),
                        point_q=float(best_pq),
                        point_method=str(point_method),
                        bias=bias,
                    )

        mses.append(round(mse, 5))
        maes.append(round(mae, 5))
        already_evaluated = True

        # Save an explicit report (keep legacy per-alpha blocks by collapsing at the selected point quantile).
        val_metrics = {
            float(a): (float(val_grid[float(a)][best_pq][0]), float(val_grid[float(a)][best_pq][1])) for a in alpha_list_all
        }
        test_metrics = {
            float(a): (float(test_grid[float(a)][best_pq][0]), float(test_grid[float(a)][best_pq][1])) for a in alpha_list_all
        }
        alpha_search_report = {
            "alpha_mode": "scalar",
            "alpha_split": None,
            "alpha_list": list(alpha_list_all),
            "best_alpha": float(best_alpha),
            "best_alpha_raw": None if best_alpha_raw is None else float(best_alpha_raw),
            "best_alpha_bias": None if best_alpha_bias is None else float(best_alpha_bias),
            "best_alpha_short": None,
            "best_alpha_long": None,
            "alpha_search_metric": str(metric),
            "alpha_search_mae_weight": float(args.alpha_search_mae_weight),
            "alpha_search_mse_guard_ratio": float(getattr(args, "alpha_search_mse_guard_ratio", 0.0)),
            "alpha_search_select_split": str(select_split),
            "val_metrics": dict(val_metrics),
            "test_metrics": dict(test_metrics),
            "point_quantile_method": str(point_method),
            "point_quantile_candidates": [float(pq) for pq in point_q_list],
            "best_point_quantile": float(best_pq),
            "bias_correct_split": str(bias_split),
            "bias_correct_select_mode": str(bias_select_mode),
            "bias_correct_selected": None
            if str(args.bias_correct) == "none"
            else ("bias" if bool(bias_selected) else "raw"),
            "val_grid_metrics": {
                float(a): {
                    float(pq): (float(val_grid[float(a)][float(pq)][0]), float(val_grid[float(a)][float(pq)][1]))
                    for pq in point_q_list
                }
                for a in alpha_list_all
            },
            "test_grid_metrics": {
                float(a): {
                    float(pq): (float(test_grid[float(a)][float(pq)][0]), float(test_grid[float(a)][float(pq)][1]))
                    for pq in point_q_list
                }
                for a in alpha_list_all
            },
            "bias_correct": str(args.bias_correct),
            "bias_correct_max_windows": int(args.bias_correct_max_windows),
            "bias_correct_report_grid": int(getattr(args, "bias_correct_report_grid", 0) or 0),
            "bias_used_windows": None if bias_used is None else int(bias_used),
            "best_test_raw": (float(raw_mse), float(raw_mae)),
            "best_test_bias_corrected": None if bias is None else (float(mse), float(mae)),
        }
        if str(args.bias_correct) != "none" and bool(int(getattr(args, "bias_correct_report_grid", 0) or 0)):
            alpha_search_report["test_metrics_bias_corrected"] = {
                float(a): (float(test_metrics_bc[float(a)][0]), float(test_metrics_bc[float(a)][1])) for a in alpha_list_all
            }

    elif alpha_mode == "2segment":
        val_grid_2 = _memory_eval_grid_2segment(
            model_base,
            model_mem,
            val_loader,
            quantiles,
            torch.device(device_address),
            alpha_list_all,
            point_q_list,
            point_method,
            alpha_split=int(alpha_split),
        )
        test_grid_2 = _memory_eval_grid_2segment(
            model_base,
            model_mem,
            test_loader,
            quantiles,
            torch.device(device_address),
            alpha_list_all,
            point_q_list,
            point_method,
            alpha_split=int(alpha_split),
        )
        select_grid_2 = test_grid_2 if select_split == "test" else val_grid_2
        if search_enabled:
            def _cfg_score(cfg):
                a_s, a_l, pq = cfg
                mse_a, mae_a = select_grid_2[(a_s, a_l)][pq]
                if metric == "mse_then_mae":
                    return (mse_a, mae_a)
                if metric == "mae_then_mse":
                    return (mae_a, mse_a)
                if metric == "mse":
                    return (mse_a,)
                if metric == "mae":
                    return (mae_a,)
                if metric in {"sum", "combo"}:
                    return (mse_a + float(args.alpha_search_mae_weight) * mae_a,)
                if metric == "mae_guard":
                    raise RuntimeError("mae_guard handled outside _cfg_score")
                raise ValueError(f"Unknown alpha_search_metric: {metric}")

            candidates = [
                (float(a_s), float(a_l), float(pq))
                for a_s in alpha_list_all
                for a_l in alpha_list_all
                for pq in point_q_list
            ]
            if metric == "mae_guard":
                guard = float(args.alpha_search_mse_guard_ratio)
                if guard < 0.0:
                    raise ValueError(f"alpha_search_mse_guard_ratio must be >=0, got {guard}")
                best_mse_only = min(float(select_grid_2[(a_s, a_l)][pq][0]) for (a_s, a_l, pq) in candidates)
                thresh = best_mse_only * (1.0 + guard)
                allowed = [
                    (a_s, a_l, pq)
                    for (a_s, a_l, pq) in candidates
                    if float(select_grid_2[(a_s, a_l)][pq][0]) <= thresh
                ]
                if not allowed:
                    allowed = candidates
                best_alpha_short, best_alpha_long, best_pq = min(
                    allowed,
                    key=lambda cfg: (
                        float(select_grid_2[(cfg[0], cfg[1])][cfg[2]][1]),
                        float(select_grid_2[(cfg[0], cfg[1])][cfg[2]][0]),
                    ),
                )
                sel_mse, sel_mae = select_grid_2[(best_alpha_short, best_alpha_long)][best_pq]
                print(
                    f"[alpha2seg_point_search] select_split={select_split} metric={metric} guard_ratio={guard} "
                    f"best_alpha_short={best_alpha_short} best_alpha_long={best_alpha_long} alpha_split={int(alpha_split)} "
                    f"best_point_q={best_pq} select_mse={sel_mse:.6f} select_mae={sel_mae:.6f} "
                    f"(mse_best={best_mse_only:.6f} mse_thresh={thresh:.6f})"
                )
            else:
                best_alpha_short, best_alpha_long, best_pq = min(candidates, key=_cfg_score)
                sel_mse, sel_mae = select_grid_2[(best_alpha_short, best_alpha_long)][best_pq]
                print(
                    f"[alpha2seg_point_search] select_split={select_split} metric={metric} best_alpha_short={best_alpha_short} best_alpha_long={best_alpha_long} "
                    f"alpha_split={int(alpha_split)} best_point_q={best_pq} "
                    f"select_mse={sel_mse:.6f} select_mae={sel_mae:.6f}"
                )
        else:
            a0 = float(alpha_list_all[0])
            best_alpha_short = a0
            best_alpha_long = a0
            best_pq = float(point_q_list[0])
        raw_mse, raw_mae = test_grid_2[(best_alpha_short, best_alpha_long)][best_pq]
        mse, mae = raw_mse, raw_mae
        bias = None
        bias_used = None
        if str(args.bias_correct) != "none":
            bias_loader = test_loader if bias_split == "test" else val_loader
            bias, bias_used = _estimate_bias_from_val(
                model_base,
                model_mem,
                bias_loader,
                quantiles,
                torch.device(device_address),
                alpha_mode="2segment",
                alpha=0.0,
                alpha_short=float(best_alpha_short),
                alpha_long=float(best_alpha_long),
                alpha_split=int(alpha_split),
                point_q=float(best_pq),
                point_method=str(point_method),
                bias_mode=str(args.bias_correct),
                max_windows=int(args.bias_correct_max_windows),
            )
            mse, mae = _memory_eval_one(
                model_base,
                model_mem,
                test_loader,
                quantiles,
                torch.device(device_address),
                alpha_mode="2segment",
                alpha=0.0,
                alpha_short=float(best_alpha_short),
                alpha_long=float(best_alpha_long),
                alpha_split=int(alpha_split),
                point_q=float(best_pq),
                point_method=str(point_method),
                bias=bias,
            )

        mses.append(round(mse, 5))
        maes.append(round(mae, 5))
        already_evaluated = True

        def _pair_key(a_s: float, a_l: float) -> str:
            return f"{float(a_s)},{float(a_l)}"

        cfg_list = [(float(a_s), float(a_l)) for a_s in alpha_list_all for a_l in alpha_list_all]
        val_pair_metrics = {
            _pair_key(a_s, a_l): (
                float(val_grid_2[(a_s, a_l)][best_pq][0]),
                float(val_grid_2[(a_s, a_l)][best_pq][1]),
            )
            for (a_s, a_l) in cfg_list
        }
        test_pair_metrics = {
            _pair_key(a_s, a_l): (
                float(test_grid_2[(a_s, a_l)][best_pq][0]),
                float(test_grid_2[(a_s, a_l)][best_pq][1]),
            )
            for (a_s, a_l) in cfg_list
        }
        alpha_search_report = {
            "alpha_mode": "2segment",
            "alpha_split": int(alpha_split),
            "alpha_list": list(alpha_list_all),
            "best_alpha": None,
            "best_alpha_short": float(best_alpha_short),
            "best_alpha_long": float(best_alpha_long),
            "alpha_search_metric": str(metric),
            "alpha_search_mae_weight": float(args.alpha_search_mae_weight),
            "alpha_search_mse_guard_ratio": float(getattr(args, "alpha_search_mse_guard_ratio", 0.0)),
            "alpha_search_select_split": str(select_split),
            "val_metrics": dict(val_pair_metrics),
            "test_metrics": dict(test_pair_metrics),
            "point_quantile_method": str(point_method),
            "point_quantile_candidates": [float(pq) for pq in point_q_list],
            "best_point_quantile": float(best_pq),
            "bias_correct_split": str(bias_split),
            "val_grid_metrics": {
                _pair_key(a_s, a_l): {
                    float(pq): (float(val_grid_2[(a_s, a_l)][float(pq)][0]), float(val_grid_2[(a_s, a_l)][float(pq)][1]))
                    for pq in point_q_list
                }
                for (a_s, a_l) in cfg_list
            },
            "test_grid_metrics": {
                _pair_key(a_s, a_l): {
                    float(pq): (float(test_grid_2[(a_s, a_l)][float(pq)][0]), float(test_grid_2[(a_s, a_l)][float(pq)][1]))
                    for pq in point_q_list
                }
                for (a_s, a_l) in cfg_list
            },
            "bias_correct": str(args.bias_correct),
            "bias_correct_max_windows": int(args.bias_correct_max_windows),
            "bias_used_windows": None if bias_used is None else int(bias_used),
            "best_test_raw": (float(raw_mse), float(raw_mae)),
            "best_test_bias_corrected": None if bias is None else (float(mse), float(mae)),
        }

    else:
        raise ValueError(f"Unknown alpha_mode: {alpha_mode}")

elif run_mode == 'rag' and 'retrieve' in args.model_id:
    retrieval_database_names = '_'.join(args.metadata['database_name'])
    retrieved_data_path = os.path.join(args.root_path, f'{ori_data_path.split(".")[0]}_retrieve_{retrieval_database_names}_{args.metadata["lookback_length"]}_{args.mode}_{args.embedding_tuning}.csv')
    if os.path.exists(retrieved_data_path):
        print(f'----------retrieval for {args.model_id} has done!!----------')
    else:
        print(f'----------retrieving for {args.model_id} ...----------')
        # Lazy import to keep memory mode runnable without faiss installed.
        from chronos import ChronosPipeline
        from sklearn.preprocessing import StandardScaler
        from retrieve import do_retrieve, load_database

        if 'chronos' in args.embedding_model_type:
            if args.embedding_tuning == None:
                model_path = "amazon/chronos-t5-base"
            else: 
                model_path = f"../tuning_results/{args.metadata_database_name}_{str(args.seq_len)}_chronos_{args.embedding_tuning}"
                if not os.path.exists(model_path):
                    exit('embedding model path does not exist!!')
            embedding_model = ChronosPipeline.from_pretrained(
                model_path,
                device_map=device_address,
                torch_dtype=torch.bfloat16,
            )
        else:
            print('embedding model type error!!')
            exit()
        top_k = args.top_k if args.top_k > 20 else 20
        do_retrieve(ori_data_path.split('.')[0], args.retrieval_database_dir, args.root_path, args.metadata, args.mode, top_k, args.seq_len, args.pred_len, fix_seed, args.dimension, embedding_model, args.save, args.embedding_tuning)
    print('retrieved_data_path = {}'.format(retrieved_data_path))
    args.data_path = retrieved_data_path.split('/')[-1]

    # load retrieved raw data, it will be used to reconstruct the retrieved data
    from sklearn.preprocessing import StandardScaler
    from retrieve import load_database
    retriever_rawdata = []
    if args.mode == 'only_self' or args.mode == 'only_self_train':
        # database: {var1: {}, var2: {}, ...}
        # retriever_rawdata: [var1_raw_data, var2_raw_data, ...]
        database = load_database(os.path.join(args.retrieval_database_dir, f'{args.metadata["database_name"][0]}_{args.metadata["frequency"]}_{args.metadata["lookback_length"]}.pkl'))
        for variable in database.keys():
            retriever_rawdata.append(database[variable]['raw_data'])
    elif args.mode == 'all_vars':
        for database_name in args.metadata['database_name']:
            database = load_database(os.path.join(args.retrieval_database_dir, f'{args.metadata["database_name"][0]}_{args.metadata["frequency"]}_{args.metadata["lookback_length"]}.pkl'))
            for variable in database.keys():
                retriever_rawdata.append(database[variable]['raw_data'])

    # scale transform for retrieved data
    scaler = StandardScaler()
    retriever_rawdata = np.array(retriever_rawdata).T
    scaler.fit(retriever_rawdata)
    retriever_rawdata = scaler.transform(retriever_rawdata)         #(n_samples, n_features)
    retriever_rawdata = retriever_rawdata.T
    test_data, test_loader = data_provider(args, 'test', retriever_rawdata=retriever_rawdata)

else:
    test_data, test_loader = data_provider(args, 'test')

if not already_evaluated:
    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))
    device = torch.device(device_address)

    time_now = time.time()

    if args.model == 'ChronosBolt':
        model = ChronosBoltPipeline.from_pretrained(args.pretrained_model_path)
        autogluon_state_path = os.path.join(args.pretrained_model_path, "autogluon_model.pth")
        if os.path.isfile(autogluon_state_path):
            model.model.load_state_dict(torch.load(autogluon_state_path, map_location=map_location))
        else:
            print(f"[warn] autogluon_model.pth not found at {autogluon_state_path}; using HF weights from {args.pretrained_model_path}")
        model.model.to(device)
    elif args.model == 'ChronosBoltRetrieve' and run_mode == 'rag':
        config = AutoConfig.from_pretrained(args.pretrained_model_path)
        model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(args.pretrained_model_path, config=config, augment=args.augment_mode)
        state_dict = torch.load(best_model_path, map_location=map_location)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        model.to(device)
    else:
        print('model error')
        exit()

    print("------------------------------------")

    if run_mode == 'rag' and 'retrieve' in args.model_id:
        mse, mae = test_retrieve(model, test_data, test_loader, args, device)
    else:
        mse, mae = test(model, test_data, test_loader, args, device)
    mses.append(round(mse,5))
    maes.append(round(mae,5))

if len(maes)==0 : exit()
maes = np.array(maes)
mses = np.array(mses)
print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))
    
log_dir = 'results/forecast_evaluation'
os.makedirs(log_dir, exist_ok=True)

# save_file_name can be either:
#   - a plain filename (backward-compatible): "memory_quantile_eval_foo.txt" (or older "memdec_quantile_eval_foo.txt")
#   - a path under results/forecast_evaluation: "amazon_run/foo.txt"
#   - an absolute path: "/abs/path/foo.txt"
if os.path.isabs(log_fine_name):
    file_path = log_fine_name
else:
    norm_name = os.path.normpath(log_fine_name)
    norm_log_dir = os.path.normpath(log_dir)
    if norm_name == norm_log_dir or norm_name.startswith(norm_log_dir + os.sep):
        file_path = norm_name
    else:
        file_path = os.path.join(log_dir, log_fine_name)
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'a') as f : 
    f.write("{}\n".format(args.model_id))
    f.write("mse:{:.4f}, std:{:.4f} ---- mae:{:.4f}, std:{:.4f}\n".format(np.mean(mses), np.std(mses) , np.mean(maes), np.std(maes)))
    if speed_report is not None:
        f.write("speed_split:{}\n".format(speed_report.get("split", "")))
        f.write("speed_device:{}\n".format(speed_report.get("device", "")))
        f.write("speed_warmup_iters:{}\n".format(speed_report.get("warmup_iters", "")))
        f.write("speed_num_iters:{}\n".format(speed_report.get("num_iters", "")))
        f.write("speed_batch_size:{}\n".format(speed_report.get("batch_size", "")))
        f.write("speed_windows:{}\n".format(speed_report.get("windows", "")))
        f.write("speed_base_ms_per_iter:{:.6f}\n".format(float(speed_report.get("base_ms_per_iter", 0.0))))
        f.write("speed_mem_ms_per_iter:{:.6f}\n".format(float(speed_report.get("mem_ms_per_iter", 0.0))))
        f.write("speed_total_ms_per_iter:{:.6f}\n".format(float(speed_report.get("total_ms_per_iter", 0.0))))
        f.write("speed_base_ms_per_window:{:.9f}\n".format(float(speed_report.get("base_ms_per_window", 0.0))))
        f.write("speed_mem_ms_per_window:{:.9f}\n".format(float(speed_report.get("mem_ms_per_window", 0.0))))
        f.write("speed_total_ms_per_window:{:.9f}\n".format(float(speed_report.get("total_ms_per_window", 0.0))))
    if alpha_search_report is not None:
        alpha_mode = str(alpha_search_report.get("alpha_mode", "scalar"))
        alpha_split = alpha_search_report.get("alpha_split", None)
        alpha_list = list(alpha_search_report["alpha_list"])
        best_alpha = alpha_search_report.get("best_alpha", None)
        best_alpha_short = alpha_search_report.get("best_alpha_short", None)
        best_alpha_long = alpha_search_report.get("best_alpha_long", None)
        metric = str(alpha_search_report.get("alpha_search_metric", "mse_then_mae"))
        mae_w = float(alpha_search_report.get("alpha_search_mae_weight", 1.0))
        select_split = str(alpha_search_report.get("alpha_search_select_split", "val"))
        bias_split = str(alpha_search_report.get("bias_correct_split", "val"))
        point_method = str(alpha_search_report.get("point_quantile_method", "nearest"))
        point_candidates = [float(x) for x in alpha_search_report.get("point_quantile_candidates", [0.5])]
        best_pq = float(alpha_search_report.get("best_point_quantile", 0.5))
        bias_correct = str(alpha_search_report.get("bias_correct", "none"))

        def _score(mse_v: float, mae_v: float):
            if metric == "mse_then_mae":
                return (mse_v, mae_v)
            if metric == "mae_then_mse":
                return (mae_v, mse_v)
            if metric == "mse":
                return mse_v
            if metric == "mae":
                return mae_v
            if metric in {"sum", "combo"}:
                return mse_v + mae_w * mae_v
            return None

        def _pair_key(a_s: float, a_l: float) -> str:
            return f"{float(a_s)},{float(a_l)}"

        f.write("alpha_candidates:{}\n".format(",".join([str(a) for a in alpha_list])))
        f.write("alpha_mode:{}\n".format(alpha_mode))
        f.write("alpha_search_select_split:{}\n".format(select_split))
        f.write("bias_correct_split:{}\n".format(bias_split))
        split_label = select_split
        if alpha_mode == "2segment":
            f.write("alpha_split:{}\n".format("" if alpha_split is None else int(alpha_split)))
            f.write("alpha_selected_short({}):{}\n".format(split_label, best_alpha_short))
            f.write("alpha_selected_long({}):{}\n".format(split_label, best_alpha_long))
        else:
            f.write("alpha_selected({}):{}\n".format(split_label, best_alpha))
        f.write("point_quantile_method:{}\n".format(point_method))
        f.write("point_quantile_candidates:{}\n".format(",".join([str(x) for x in point_candidates])))
        f.write("point_quantile_selected({}):{}\n".format(split_label, best_pq))
        f.write("bias_correct:{}\n".format(bias_correct))
        if bias_correct != "none":
            f.write("bias_correct_max_windows:{}\n".format(int(alpha_search_report.get("bias_correct_max_windows", 0))))
            f.write("bias_correct_report_grid:{}\n".format(int(alpha_search_report.get("bias_correct_report_grid", 0))))
            f.write("bias_used_windows:{}\n".format(alpha_search_report.get("bias_used_windows", "")))
            raw = alpha_search_report.get("best_test_raw", None)
            corr = alpha_search_report.get("best_test_bias_corrected", None)
            if raw is not None:
                f.write("best_test_raw:mse={},mae={}\n".format(raw[0], raw[1]))
            if corr is not None:
                f.write("best_test_bias_corrected:mse={},mae={}\n".format(corr[0], corr[1]))
        f.write("alpha_search_metric:{}\n".format(metric))
        if metric in {"sum", "combo"}:
            f.write("alpha_search_mae_weight:{}\n".format(mae_w))

        if alpha_mode == "2segment":
            f.write("val_alpha2seg_metrics:\n")
            for a_s in alpha_list:
                for a_l in alpha_list:
                    k = _pair_key(a_s, a_l)
                    mse_a, mae_a = alpha_search_report["val_metrics"][k]
                    score_a = _score(mse_a, mae_a)
                    f.write(
                        "  val alpha_short={} alpha_long={} mse={} mae={} score={}\n".format(
                            a_s, a_l, round(mse_a, 6), round(mae_a, 6), score_a
                        )
                    )
            f.write("test_alpha2seg_metrics:\n")
            for a_s in alpha_list:
                for a_l in alpha_list:
                    k = _pair_key(a_s, a_l)
                    mse_a, mae_a = alpha_search_report["test_metrics"][k]
                    score_a = _score(mse_a, mae_a)
                    f.write(
                        "  test alpha_short={} alpha_long={} mse={} mae={} score={}\n".format(
                            a_s, a_l, round(mse_a, 6), round(mae_a, 6), score_a
                        )
                    )
            f.write("val_alpha2seg_point_metrics:\n")
            for a_s in alpha_list:
                for a_l in alpha_list:
                    k = _pair_key(a_s, a_l)
                    for pq in point_candidates:
                        mse_a, mae_a = alpha_search_report["val_grid_metrics"][k][float(pq)]
                        score_a = _score(mse_a, mae_a)
                        f.write(
                            "  val alpha_short={} alpha_long={} point_q={} mse={} mae={} score={}\n".format(
                                a_s, a_l, pq, round(mse_a, 6), round(mae_a, 6), score_a
                            )
                        )
            f.write("test_alpha2seg_point_metrics:\n")
            for a_s in alpha_list:
                for a_l in alpha_list:
                    k = _pair_key(a_s, a_l)
                    for pq in point_candidates:
                        mse_a, mae_a = alpha_search_report["test_grid_metrics"][k][float(pq)]
                        score_a = _score(mse_a, mae_a)
                        f.write(
                            "  test alpha_short={} alpha_long={} point_q={} mse={} mae={} score={}\n".format(
                                a_s, a_l, pq, round(mse_a, 6), round(mae_a, 6), score_a
                            )
                        )
        else:
            f.write("val_alpha_metrics:\n")
            for a in alpha_list:
                mse_a, mae_a = alpha_search_report["val_metrics"][float(a)]
                score_a = _score(mse_a, mae_a)
                f.write(
                    "  val alpha={} mse={} mae={} score={}\n".format(
                        a, round(mse_a, 6), round(mae_a, 6), score_a
                    )
                )
            f.write("test_alpha_metrics:\n")
            for a in alpha_list:
                mse_a, mae_a = alpha_search_report["test_metrics"][float(a)]
                score_a = _score(mse_a, mae_a)
                f.write(
                    "  test alpha={} mse={} mae={} score={}\n".format(
                        a, round(mse_a, 6), round(mae_a, 6), score_a
                    )
                )
            f.write("val_alpha_point_metrics:\n")
            for a in alpha_list:
                for pq in point_candidates:
                    mse_a, mae_a = alpha_search_report["val_grid_metrics"][float(a)][float(pq)]
                    score_a = _score(mse_a, mae_a)
                    f.write(
                        "  val alpha={} point_q={} mse={} mae={} score={}\n".format(
                            a, pq, round(mse_a, 6), round(mae_a, 6), score_a
                        )
                    )
            f.write("test_alpha_point_metrics:\n")
            for a in alpha_list:
                for pq in point_candidates:
                    mse_a, mae_a = alpha_search_report["test_grid_metrics"][float(a)][float(pq)]
                    score_a = _score(mse_a, mae_a)
                    f.write(
                        "  test alpha={} point_q={} mse={} mae={} score={}\n".format(
                            a, pq, round(mse_a, 6), round(mae_a, 6), score_a
                        )
                    )
            if "test_metrics_bias_corrected" in alpha_search_report:
                f.write("test_alpha_bias_corrected_metrics:\n")
                for a in alpha_list:
                    mse_a, mae_a = alpha_search_report["test_metrics_bias_corrected"][float(a)]
                    score_a = _score(mse_a, mae_a)
                    f.write(
                        "  test_bc alpha={} point_q={} mse={} mae={} score={}\n".format(
                            a, best_pq, round(mse_a, 6), round(mae_a, 6), score_a
                        )
                    )
        
print(log_fine_name)
            
