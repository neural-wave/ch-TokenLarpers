"""Masked reconstruction pretraining for NumericPatchEncoder."""
from __future__ import annotations
import numpy as np
import torch
from .encoder import NumericPatchEncoder
from .windows import WindowBatch

def pretrain_encoder(model: NumericPatchEncoder, batch: WindowBatch, *, epochs=5, batch_size=32, learning_rate=0.001, mask_ratio=0.3, seed=0, device="cpu") -> list[float]:
    if epochs < 0 or batch_size <= 0 or not (0 < mask_ratio <= 1): raise ValueError("invalid pretraining settings")
    n = len(batch.values)
    if n == 0: raise ValueError("cannot pretrain on empty WindowBatch")
    p = model.patch_length; q = model.n_patches
    values = torch.as_tensor(batch.values, dtype=torch.float32, device=device)
    observed = torch.as_tensor(batch.observed, dtype=torch.bool, device=device)
    calendar = torch.as_tensor(batch.calendar, dtype=torch.float32, device=device)
    magnitude = torch.as_tensor(batch.magnitude, dtype=torch.float32, device=device)
    if magnitude.ndim != 2 or magnitude.shape[0] != n or magnitude.shape[1] < 2:
        raise ValueError("WindowBatch magnitude must contain center and scale")
    original_center = magnitude[:, 0]
    original_scale = magnitude[:, 1]
    if (not bool(torch.isfinite(original_center).all())
            or not bool(torch.isfinite(original_scale).all())
            or bool((original_scale <= 0).any())):
        raise ValueError("WindowBatch magnitude contains invalid normalization statistics")
    if epochs == 0:
        return []
    patch_obs = observed[...,0].reshape(n,q,p).any(-1)
    if not bool(patch_obs.any()): raise ValueError("no genuine observations available for reconstruction")
    # Targets and masks are generated once, deterministically, so each epoch has a clear objective.
    rng = np.random.default_rng(seed)
    selected = np.zeros((n,q), dtype=bool)
    for i in range(n):
        eligible = np.flatnonzero(patch_obs[i].detach().cpu().numpy())
        if len(eligible) <= 1:
            raise ValueError("mask selection would hide every genuine observation")
        k = max(1, int(np.ceil(len(eligible)*mask_ratio)))
        k = min(k, len(eligible) - 1)
        selected[i, rng.choice(eligible, size=k, replace=False)] = True
    target_patches = torch.as_tensor(selected, dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device); model.train(); history=[]
    for _ in range(epochs):
        order = rng.permutation(n); losses=[]
        for begin in range(0,n,batch_size):
            ix = torch.as_tensor(order[begin:begin+batch_size], dtype=torch.long, device=device)
            v, m, c = values[ix], observed[ix], calendar[ix]
            target = target_patches[ix]
            # Undo build_windows' affine transform before recomputing robust
            # statistics.  Hidden target values therefore cannot influence the
            # visible center or scale, including when visible values are flat.
            visible = m[..., 0].reshape(-1, q, p) & ~target.unsqueeze(-1)
            visible_flat = visible.reshape(visible.shape[0], -1)
            physical = (v[..., 0] * original_scale[ix].unsqueeze(-1)
                        + original_center[ix].unsqueeze(-1))
            raw = physical.reshape(visible.shape[0], -1)
            centers, scales = [], []
            for row, mask in zip(raw, visible_flat):
                genuine = row[mask]
                if genuine.numel() == 0:
                    raise ValueError("mask selection would hide every genuine observation")
                center = genuine.median()
                scale = (genuine - center).abs().median().clamp_min(0.001)
                centers.append(center)
                scales.append(scale)
            center = torch.stack(centers).unsqueeze(-1)
            scale = torch.stack(scales).unsqueeze(-1)
            normalized = (raw - center) / scale
            vm = normalized.reshape(-1,q,p).clone()
            mm = m.clone().reshape(-1,q,p)
            vm[~mm] = 0.0
            vm[target] = 0.0
            mm[target] = False
            masked_v, masked_m = vm.reshape(-1, q*p, 1), mm.reshape(-1,q*p,1)
            out, unavailable = model.token_outputs(masked_v, masked_m, c, padding_mask=~patch_obs[ix])
            pred = model.reconstruction_head(out).reshape(-1, q, p)
            target_values = normalized.reshape(-1,q,p)
            # Reconstruct every genuinely observed value in selected hidden patches.
            valid = target.unsqueeze(-1) & m[..., 0].reshape(-1,q,p)
            if not bool(valid.any()): continue
            loss = ((pred-target_values)**2)[valid].mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses: raise ValueError("no valid reconstruction targets")
        history.append(float(np.mean(losses)))
    return history

