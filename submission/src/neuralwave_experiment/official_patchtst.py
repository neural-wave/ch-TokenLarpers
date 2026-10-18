"""Official yuqinie98/PatchTST self-supervised encoder adapter.

The vendored backbone is the upstream PatchTSTEncoder with mask plumbing added
at its TSTEncoder/TSTEncoderLayer boundaries. This adapter keeps the official
projection, positional encoding, residual attention, and feed-forward layers,
while adding explicit calendar and observation embeddings required by the
NeuralWave single-channel input contract.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .encoder import EncoderConfig
from .vendor.patchtst.models.patchTST import PatchTSTEncoder


class OfficialPatchTSTEncoder(nn.Module):
    """Single-channel adapter around the attributed official PatchTSTEncoder."""

    backend = "official_patchtst"

    def __init__(self, config: EncoderConfig):
        super().__init__()
        if config.context_length <= 0 or config.patch_length <= 0:
            raise ValueError("context_length and patch_length must be positive")
        if config.context_length % config.patch_length:
            raise ValueError("context_length must be divisible by patch_length")
        if config.d_model <= 0 or config.n_heads <= 0 or config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if config.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        self.config = config
        self.patch_length = config.patch_length
        self.n_patches = config.context_length // config.patch_length
        self.backbone = PatchTSTEncoder(
            c_in=1,
            num_patch=self.n_patches,
            patch_len=self.patch_length,
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            shared_embedding=True,
            d_ff=4 * config.d_model,
            norm="LayerNorm",
            attn_dropout=0.0,
            dropout=config.dropout,
            act="gelu",
            res_attention=True,
            pre_norm=False,
            store_attn=False,
            pe="zeros",
            learn_pe=True,
        )
        # These are deliberately token-level additions, after value masking and
        # the official value/position projection. They cannot inspect hidden values.
        self.observation_proj = nn.Linear(self.patch_length, config.d_model)
        self.calendar_proj = nn.Linear(4 * self.patch_length, config.d_model)
        self.reconstruction_head = nn.Linear(config.d_model, self.patch_length)

    def _tokens(self, values: Tensor, observed: Tensor, calendar: Tensor):
        if values.ndim != 3 or values.shape[-1] != 1:
            raise ValueError("values must have shape [batch, context_length, 1]")
        batch, length, _ = values.shape
        if length != self.config.context_length:
            raise ValueError("values length does not match context_length")
        if observed.ndim == 2:
            observed = observed.unsqueeze(-1)
        if observed.shape != values.shape:
            raise ValueError("observed must have shape [batch, context_length, 1]")
        if calendar.ndim != 3 or calendar.shape != (batch, length, 4):
            raise ValueError("calendar must have shape [batch, context_length, 4]")
        p, q = self.patch_length, self.n_patches
        observed_bool = observed.to(dtype=torch.bool)[..., 0]
        mask = observed_bool.to(dtype=values.dtype).reshape(batch, q, p)
        masked_values = torch.where(observed_bool, values[..., 0], torch.zeros_like(values[..., 0]))
        patches = masked_values.reshape(batch, q, p)
        cal = calendar.reshape(batch, q, 4 * p).to(dtype=values.dtype)
        unavailable = mask.sum(dim=-1).eq(0)
        side = self.observation_proj(mask) + self.calendar_proj(cal)
        return patches.unsqueeze(2), side, unavailable

    def token_outputs(
        self,
        values: Tensor,
        observed: Tensor,
        calendar: Tensor,
        padding_mask: Optional[Tensor] = None,
    ):
        patches, side, unavailable = self._tokens(values, observed, calendar)
        if padding_mask is not None:
            if padding_mask.shape != unavailable.shape:
                raise ValueError("padding_mask must have shape [batch, n_patches]")
            unavailable = padding_mask.to(dtype=torch.bool)
        safe_mask = unavailable.clone()
        all_missing = safe_mask.all(dim=1)
        if bool(all_missing.any()):
            safe_mask[all_missing, 0] = False
        encoded = self.backbone.encode_tokens(
            patches, key_padding_mask=safe_mask, extra_tokens=side
        )
        # Official backbone emits [batch, vars, d_model, patches].
        return encoded[:, 0].transpose(1, 2), unavailable

    def forward(self, values: Tensor, observed: Tensor, calendar: Tensor) -> Tensor:
        tokens, unavailable = self.token_outputs(values, observed, calendar)
        weights = (~unavailable).to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


# Short compatibility name for callers referring to the backend rather than
# the adapter's concrete implementation.
OfficialPatchTST = OfficialPatchTSTEncoder
