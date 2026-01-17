from __future__ import annotations

from typing import List

import numpy as np
import torch


def weighted_quantile_torch(
    y: torch.Tensor, weights: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """
    Weighted quantile without Python loops.

    Args:
        y: (B, K, T) or (B, K, T, C)
        weights: (B, K), assumed >=0 (will be normalized internally)
        quantiles: (Q,) in [0,1]

    Returns:
        q: (B, Q, T) or (B, Q, T, C)
    """
    if y.ndim not in (3, 4):
        raise ValueError(f"y must be 3D or 4D, got {y.shape}")
    if weights.ndim != 2:
        raise ValueError(f"weights must be (B,K), got {weights.shape}")
    if quantiles.ndim != 1:
        raise ValueError(f"quantiles must be (Q,), got {quantiles.shape}")

    B = y.shape[0]
    K = y.shape[1]
    T = y.shape[2]
    C = 1 if y.ndim == 3 else y.shape[3]

    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

    y_sorted, idx = torch.sort(y, dim=1)

    if y.ndim == 3:
        w_expand = weights[:, :, None].expand(B, K, T)
    else:
        w_expand = weights[:, :, None, None].expand(B, K, T, C)

    w_sorted = torch.gather(w_expand, dim=1, index=idx)
    cdf = torch.cumsum(w_sorted, dim=1)

    # Flatten to (N, K) for compatibility with torch.searchsorted across versions.
    if y.ndim == 3:
        y_flat = y_sorted.permute(0, 2, 1).reshape(B * T, K)
        cdf_flat = cdf.permute(0, 2, 1).reshape(B * T, K)
        N = B * T
        q = quantiles.view(1, -1).expand(N, -1)
        pos = torch.searchsorted(cdf_flat, q, right=True).clamp(0, K - 1)
        out = torch.gather(y_flat, dim=1, index=pos)
        return out.reshape(B, T, -1).permute(0, 2, 1).contiguous()
    else:
        y_flat = y_sorted.permute(0, 2, 3, 1).reshape(B * T * C, K)
        cdf_flat = cdf.permute(0, 2, 3, 1).reshape(B * T * C, K)
        N = B * T * C
        q = quantiles.view(1, -1).expand(N, -1)
        pos = torch.searchsorted(cdf_flat, q, right=True).clamp(0, K - 1)
        out = torch.gather(y_flat, dim=1, index=pos)
        out = out.reshape(B, T, C, -1).permute(0, 3, 1, 2).contiguous()
        return out


def numpy_weighted_quantile_reference(
    y: np.ndarray, w: np.ndarray, quantiles: List[float]
) -> np.ndarray:
    """
    Reference implementation (slow) for sanity checks.
    y: (B,K,T)
    w: (B,K)
    """
    B, K, T = y.shape
    Q = len(quantiles)
    out = np.zeros((B, Q, T), dtype=np.float32)
    for b in range(B):
        w_b = w[b].astype(np.float64)
        w_b = w_b / (w_b.sum() + 1e-12)
        for t in range(T):
            order = np.argsort(y[b, :, t])
            y_s = y[b, order, t]
            w_s = w_b[order]
            cdf = np.cumsum(w_s)
            for qi, q in enumerate(quantiles):
                j = np.searchsorted(cdf, q, side="right")
                j = min(max(j, 0), K - 1)
                out[b, qi, t] = y_s[j]
    return out

