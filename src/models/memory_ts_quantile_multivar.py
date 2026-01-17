from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _instance_norm_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ChronosBolt-style InstanceNorm stats over the last dimension.

    Args:
        x: (..., T)

    Returns:
        (loc, scale, is_constant) with shapes (..., 1)
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


@dataclass
class MemoryTSQuantileMultivarConfig:
    context_len: int = 512
    pred_len: int = 64
    num_quantiles: int = 9
    num_channels: int = 1
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    n_decoder_layers: int = 2
    patch_len: int = 16
    dropout: float = 0.1


class MemoryTSQuantileMultivar(nn.Module):
    """
    Multivariate Memory module (cross-channel) that consumes the full context window.

    Input:
        context: (B, L, C)
    Output:
        q: (B, Q, pred_len, C)
    """

    def __init__(self, config: MemoryTSQuantileMultivarConfig):
        super().__init__()
        self.config = config

        self.pred_len = int(config.pred_len)
        self.num_quantiles = int(config.num_quantiles)
        self.num_channels = int(config.num_channels)
        self.d_model = int(config.d_model)
        self.patch_len = int(config.patch_len)

        # Patch embedding over time with channel mixing.
        self.patch_embed = nn.Conv1d(
            in_channels=self.num_channels,
            out_channels=self.d_model,
            kernel_size=self.patch_len,
            stride=self.patch_len,
            bias=True,
        )
        self.dropout = nn.Dropout(config.dropout)

        self.max_patches = math.ceil(int(config.context_len) / self.patch_len) + 4
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

        self.out_proj = nn.Linear(self.d_model, self.num_channels * self.num_quantiles)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.horizon_embed, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _pad_left_to_multiple(self, x: torch.Tensor, multiple: int) -> torch.Tensor:
        L = x.shape[-1]
        pad = (-L) % multiple
        if pad == 0:
            return x
        return F.pad(x, (pad, 0), value=0.0)

    def forward(self, context: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError(f"context must be (B,L,C), got {context.shape}")
        out_len = self.pred_len if pred_len is None else int(pred_len)
        if out_len <= 0:
            raise ValueError(f"pred_len must be >0, got {out_len}")
        if out_len > self.pred_len:
            raise ValueError(f"pred_len={out_len} exceeds max_pred_len={self.pred_len}")
        x = context.to(dtype=torch.float32)

        # (B,L,C) -> (B,C,L) for Conv1d over time
        x = x.permute(0, 2, 1).contiguous()

        loc, scale, is_const = _instance_norm_stats(x)  # (B,C,1)
        x = _instance_norm_apply(x, loc, scale, is_const)
        x = self._pad_left_to_multiple(x, self.patch_len)

        tokens = self.patch_embed(x)  # (B, d_model, n_patches)
        tokens = tokens.transpose(1, 2).contiguous()  # (B, n_patches, d_model)

        n_patches = tokens.shape[1]
        if n_patches > self.pos_embed.shape[0]:
            raise ValueError(f"n_patches={n_patches} exceeds max_patches={self.pos_embed.shape[0]}")
        tokens = tokens + self.pos_embed[:n_patches].unsqueeze(0)
        tokens = self.dropout(tokens)

        memory = self.encoder(tokens)

        tgt = self.horizon_embed[:out_len].unsqueeze(0).expand(tokens.shape[0], -1, -1)
        dec = self.decoder(tgt=tgt, memory=memory)
        dec = self.dropout(dec)

        out = self.out_proj(dec)  # (B, out_len, C*Q)
        out = out.view(out.shape[0], out_len, self.num_channels, self.num_quantiles)
        q = out.permute(0, 3, 1, 2).contiguous()  # (B, Q, out_len, C)

        # Inverse instance norm back to dataset scale.
        loc_bc = loc.squeeze(-1).unsqueeze(1).unsqueeze(1)  # (B,1,1,C)
        scale_bc = scale.squeeze(-1).unsqueeze(1).unsqueeze(1)  # (B,1,1,C)
        q = _instance_norm_inverse(q, loc_bc, scale_bc)
        return q

