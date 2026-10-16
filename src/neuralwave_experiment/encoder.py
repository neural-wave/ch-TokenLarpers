from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .windows import WindowBatch

@dataclass(frozen=True)
class EncoderConfig:
    context_length: int = 672
    patch_length: int = 16
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

class NumericPatchEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__(); self.config = config
        if config.context_length <= 0 or config.patch_length <= 0 or config.context_length % config.patch_length:
            raise ValueError("context_length must be divisible by patch_length")
        if config.d_model % config.n_heads: raise ValueError("d_model must be divisible by n_heads")
        self.n_patches = config.context_length // config.patch_length
        self.patch_length = config.patch_length
        self.value_proj = nn.Linear(config.patch_length, config.d_model)
        self.mask_proj = nn.Linear(config.patch_length, config.d_model)
        self.calendar_proj = nn.Linear(4 * config.patch_length, config.d_model)
        self.position = nn.Parameter(torch.randn(1, self.n_patches, config.d_model) * 0.02)
        self.reconstruction_head = nn.Linear(config.d_model, config.patch_length)
        layer = nn.TransformerEncoderLayer(config.d_model, config.n_heads, 4*config.d_model, config.dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.n_layers)
        # TransformerEncoder clones one layer; explicitly randomize each clone.
        for block in self.transformer.layers:
            for parameter in block.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.zeros_(parameter)
        self.norm = nn.LayerNorm(config.d_model)
    def _tokens(self, values, observed, calendar):
        B,L,C = values.shape
        if L != self.config.context_length or C != 1: raise ValueError("invalid input shape")
        p = self.patch_length; q = self.n_patches
        v = values[...,0].reshape(B,q,p); m = observed[...,0].float().reshape(B,q,p); c = calendar.reshape(B,q,p*4)
        tokens = self.value_proj(v * m) + self.mask_proj(m) + self.calendar_proj(c) + self.position
        unavailable = m.sum(-1).eq(0)
        return tokens, unavailable
    def token_outputs(self, values, observed, calendar, padding_mask=None):
        tokens, unavailable = self._tokens(values, observed, calendar)
        # Reconstruction may hide target patches but must retain them as query tokens.
        if padding_mask is not None:
            unavailable = padding_mask
        safe_mask = unavailable.clone()
        empty = safe_mask.all(dim=1)
        if bool(empty.any()):
            safe_mask[empty, 0] = False
        out = self.transformer(tokens, src_key_padding_mask=safe_mask)
        return self.norm(out), unavailable
    def forward(self, values, observed, calendar):
        out, unavailable = self.token_outputs(values, observed, calendar)
        weights = (~unavailable).float().unsqueeze(-1)
        return (out * weights).sum(1) / weights.sum(1).clamp_min(1.0)

def encode_windows(model: NumericPatchEncoder, batch: WindowBatch, batch_size: int = 32, device="cpu") -> np.ndarray:
    if batch_size <= 0: raise ValueError("batch_size must be positive")
    model = model.to(device); was_training = model.training; model.eval(); result=[]
    with torch.no_grad():
        for i in range(0, len(batch.values), batch_size):
            sl=slice(i,i+batch_size)
            v=torch.as_tensor(batch.values[sl],dtype=torch.float32,device=device)
            m=torch.as_tensor(batch.observed[sl],dtype=torch.bool,device=device)
            c=torch.as_tensor(batch.calendar[sl],dtype=torch.float32,device=device)
            result.append(model(v,m,c).detach().cpu().numpy().astype(np.float32))
    if was_training: model.train()
    return np.concatenate(result,axis=0) if result else np.empty((0, model.config.d_model),np.float32)
