from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNorm(nn.Module):
    """
    Same InstanceNorm logic as ChronosBolt (models/ChronosBolt.py), but kept local to avoid heavy imports.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True)
            scale = torch.where(is_constant, torch.ones_like(scale), scale)
        else:
            loc, scale = loc_scale

        normalized = (x - loc) / scale
        is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True) if loc_scale is None else (scale == 1)
        normalized = torch.where(is_constant, torch.ones_like(normalized), normalized)
        return normalized, (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loc, scale = loc_scale
        is_constant = scale == 1
        return torch.where(is_constant, loc, x * scale + loc)


@dataclass
class MemoryTSQuantileConfig:
    context_len: int = 512
    pred_len: int = 64
    num_quantiles: int = 9
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    n_decoder_layers: int = 2
    patch_len: int = 16
    dropout: float = 0.1


class MemoryTSQuantile(nn.Module):
    """
    Lightweight parametric "memory decoder" for time series forecasting.

    Input:
        context: (B, L) or (B, L, C)
    Output:
        q: (B, Q, pred_len) or (B, Q, pred_len, C) if C>1 input is provided.
    """

    def __init__(self, config: MemoryTSQuantileConfig):
        super().__init__()
        self.config = config

        self.instance_norm = InstanceNorm()

        self.patch_len = int(config.patch_len)
        self.pred_len = int(config.pred_len)
        self.num_quantiles = int(config.num_quantiles)
        self.d_model = int(config.d_model)

        self.in_proj = nn.Linear(self.patch_len, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Learned positional embeddings for patch tokens and horizon queries
        self.max_patches = math.ceil(config.context_len / config.patch_len) + 4
        self.pos_embed = nn.Parameter(torch.zeros(self.max_patches, self.d_model))
        self.horizon_embed = nn.Parameter(torch.zeros(self.pred_len, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(config.n_heads),
            dim_feedforward=self.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(config.n_layers))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(config.n_heads),
            dim_feedforward=self.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(config.n_decoder_layers))

        self.out_proj = nn.Linear(self.d_model, self.num_quantiles)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.horizon_embed, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _pad_left_to_multiple(self, x: torch.Tensor, multiple: int) -> torch.Tensor:
        L = x.shape[-1]
        pad = (-L) % multiple
        if pad == 0:
            return x
        return F.pad(x, (pad, 0), value=0.0)

    def forward(self, context: torch.Tensor, pred_len: Optional[int] = None) -> torch.Tensor:
        if context.ndim == 3:
            # (B, L, C) -> treat each channel as an independent series
            B, L, C = context.shape
            x = context.permute(0, 2, 1).reshape(B * C, L)
            q = self._forward_univariate(x, pred_len=pred_len)
            out_len = q.shape[-1]
            q = q.reshape(B, C, self.num_quantiles, out_len).permute(0, 2, 3, 1).contiguous()
            return q
        if context.ndim != 2:
            raise ValueError(f"context must be (B,L) or (B,L,C), got {context.shape}")
        return self._forward_univariate(context, pred_len=pred_len)

    def _forward_univariate(self, x: torch.Tensor, pred_len: Optional[int] = None) -> torch.Tensor:
        out_len = self.pred_len if pred_len is None else int(pred_len)
        if out_len <= 0:
            raise ValueError(f"pred_len must be >0, got {out_len}")
        if out_len > self.pred_len:
            raise ValueError(f"pred_len={out_len} exceeds max_pred_len={self.pred_len}")

        x = x.to(dtype=torch.float32)
        x, loc_scale = self.instance_norm(x)
        x = self._pad_left_to_multiple(x, self.patch_len)

        B, L = x.shape
        n_patches = L // self.patch_len
        patches = x.view(B, n_patches, self.patch_len)

        h = self.in_proj(patches)
        if n_patches > self.pos_embed.shape[0]:
            raise ValueError(f"n_patches={n_patches} exceeds max_patches={self.pos_embed.shape[0]}")
        h = h + self.pos_embed[:n_patches].unsqueeze(0)
        h = self.dropout(h)

        memory = self.encoder(h)

        tgt = self.horizon_embed[:out_len].unsqueeze(0).expand(B, -1, -1)
        dec = self.decoder(tgt=tgt, memory=memory)
        dec = self.dropout(dec)

        q = self.out_proj(dec)  # (B, out_len, Q)
        q = q.permute(0, 2, 1).contiguous()  # (B, Q, out_len)

        # Inverse instance norm back to the same scale as input context.
        loc, scale = loc_scale
        q = self.instance_norm.inverse(q.reshape(B, -1), (loc, scale)).reshape(B, self.num_quantiles, out_len)
        return q

