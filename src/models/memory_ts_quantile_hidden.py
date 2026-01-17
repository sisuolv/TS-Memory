from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


def instance_norm_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute InstanceNorm stats matching models/ChronosBolt.py semantics.

    Returns (loc, scale, is_constant) with reduction over the last dimension.
    """
    loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
    scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
    is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True)
    scale = torch.where(is_constant, torch.ones_like(scale), scale)
    return loc, scale, is_constant


@dataclass
class MemoryTSHiddenDeltaConfig:
    pred_len: int = 64
    num_quantiles: int = 9
    hidden_dim_in: int = 768
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    n_decoder_layers: int = 2
    dropout: float = 0.1
    max_tokens: int = 256
    hidden_source: str = "encoder"  # encoder|decoder (metadata only; selection handled by caller)


class MemoryTSHiddenDelta(nn.Module):
    """
    Memory head that consumes frozen ChronosBolt hidden states and predicts a *normalized* delta:
        delta_norm = (q_teacher - q_base) / scale
    It then outputs delta = delta_norm * scale (dataset scale).

    Input:
        hidden: (B, N, D_in) or (B, D_in)
        scale: (B, 1) or (B,)  (InstanceNorm scale from the *raw context* in dataset scale)
        is_constant: (B, 1) bool (optional)

    Output:
        delta: (B, Q, pred_len) in dataset scale
    """

    requires_base_hidden: bool = True
    outputs_delta: bool = True

    def __init__(self, config: MemoryTSHiddenDeltaConfig):
        super().__init__()
        self.config = config
        self.pred_len = int(config.pred_len)
        self.num_quantiles = int(config.num_quantiles)
        self.hidden_dim_in = int(config.hidden_dim_in)
        self.d_model = int(config.d_model)

        self.in_proj = nn.Linear(self.hidden_dim_in, self.d_model)
        self.dropout = nn.Dropout(float(config.dropout))

        self.pos_embed = nn.Parameter(torch.zeros(int(config.max_tokens), self.d_model))
        self.horizon_embed = nn.Parameter(torch.zeros(self.pred_len, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(config.n_heads),
            dim_feedforward=self.d_model * 4,
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(config.n_layers))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(config.n_heads),
            dim_feedforward=self.d_model * 4,
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(config.n_decoder_layers))

        self.out_proj = nn.Linear(self.d_model, self.num_quantiles)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.horizon_embed, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden: torch.Tensor, scale: torch.Tensor, is_constant: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)  # (B,1,D)
        if hidden.ndim != 3:
            raise ValueError(f"hidden must be (B,N,D) or (B,D), got {hidden.shape}")

        hidden = hidden.to(dtype=torch.float32)
        B, N, _D = hidden.shape

        h = self.in_proj(hidden)  # (B,N,d_model)
        if N > self.pos_embed.shape[0]:
            raise ValueError(f"N={N} exceeds max_tokens={self.pos_embed.shape[0]}")
        h = h + self.pos_embed[:N].unsqueeze(0)
        h = self.dropout(h)

        memory = self.encoder(h)

        tgt = self.horizon_embed.unsqueeze(0).expand(B, -1, -1)
        dec = self.decoder(tgt=tgt, memory=memory)
        dec = self.dropout(dec)

        delta_norm = self.out_proj(dec).permute(0, 2, 1).contiguous()  # (B,Q,pred_len)

        if scale.ndim == 1:
            scale = scale.view(B, 1)
        if scale.ndim != 2:
            raise ValueError(f"scale must be (B,) or (B,1), got {scale.shape}")
        scale = scale.to(dtype=delta_norm.dtype, device=delta_norm.device).view(B, 1, 1)
        delta = delta_norm * scale

        if is_constant is not None:
            if is_constant.ndim == 1:
                is_constant = is_constant.view(B, 1)
            is_constant = is_constant.to(device=delta.device).view(B, 1, 1)
            delta = torch.where(is_constant, torch.zeros_like(delta), delta)

        return delta

