"""Optional frozen encoder features and fold-fitted four-component projection.

This module deliberately imports torch only when a checkpoint is selected.  The
ordinary physical PV path therefore remains usable in environments without the
encoder runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EMBEDDING_FEATURES = tuple(f"embedding_pc{i}" for i in range(1, 5))


def _as_frame(raw: Any, ids: list[str] | None = None) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    a = np.asarray(raw, dtype=float)
    if a.ndim != 2:
        raise ValueError("raw embeddings must be a 2-D array or DataFrame")
    if ids is None:
        ids = [str(i) for i in range(len(a))]
    if len(ids) != len(a):
        raise ValueError("raw embedding IDs and rows differ")
    return pd.DataFrame(a, index=[str(x) for x in ids])


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



class ProjectionUnavailable(ValueError):
    """The optional embedding candidate cannot be fitted on this fold."""


class FrozenEncoder:
    """Loaded, immutable A-artifact encoder used by the hybrid option."""
    def __init__(self, model, directory: Path, metadata: dict, device: str):
        self.model = model
        self.directory = Path(directory)
        self.metadata = metadata
        self.device = device
        self.weights_path = self.directory / "encoder.pt"
        self.metadata_path = self.directory / "encoder.json"
        self.hashes = {
            "encoder.pt": _hash(self.weights_path),
            "encoder.json": _hash(self.metadata_path),
        }

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> "FrozenEncoder":
        # Keep torch and the neuralwave package out of baseline imports.
        from neuralwave_experiment.checkpoint import load_encoder_artifact

        directory = Path(directory)
        model, metadata = load_encoder_artifact(directory, device=device)
        return cls(model, directory, metadata, device)

    @property
    def config(self) -> dict:
        return dict(self.metadata.get("encoder_config", {}))

    @property
    def window_config(self) -> dict:
        return dict(self.metadata.get("window_config", {}))

    def reference(self) -> dict:
        return {
            "weights": "encoder/encoder.pt",
            "metadata": "encoder/encoder.json",
            "weights_sha256": self.hashes["encoder.pt"],
            "metadata_sha256": self.hashes["encoder.json"],
            "encoder_config": self.config,
            "window_config": self.window_config,
            "source": self.metadata.get("source", {}),
        }

    def raw_customer_features(self, table, ids: list[str]) -> pd.DataFrame:
        return raw_customer_features(table, ids, encoder=self)


def raw_customer_features(table, ids: list[str], *, encoder: FrozenEncoder | None = None) -> pd.DataFrame:
    """Aggregate chronological encoder window vectors as mean/std/last-first."""
    if encoder is None:
        raise ValueError("encoder is required for raw_customer_features")
    frame = getattr(table, "frame", None)
    if frame is None:
        raise ValueError("feature table does not retain parsed epoch rows")
    from neuralwave_experiment.windows import WindowConfig, build_windows
    import torch

    dim = int(encoder.config.get("d_model", 0))
    if dim <= 0:
        raise ValueError("encoder metadata does not specify a positive d_model")
    wcfg = encoder.window_config
    allowed = {"context_length", "stride", "min_observed_fraction", "max_windows_per_customer"}
    config = WindowConfig(**{k: wcfg[k] for k in allowed if k in wcfg})
    wanted = [str(c) for c in ids]
    cache = getattr(table, "_embedding_cache", None)
    if cache is None:
        cache = {}
        try:
            setattr(table, "_embedding_cache", cache)
        except AttributeError:
            pass
    cache_key = (encoder.hashes.get("encoder.pt", ""), tuple(wanted))
    if cache is not None and cache_key in cache:
        return cache[cache_key].copy()
    out = np.full((len(wanted), 3 * dim), np.nan, dtype=float)
    for row, cid in enumerate(wanted):
        one = frame[frame["customer_id"].astype(str) == cid]
        if one.empty:
            continue
        measurements = one.rename(columns={"epoch_s": "timestamp_epoch"}).copy()
        measurements["value_status"] = "observed"
        measurements = measurements[["customer_id", "timestamp_epoch", "energy_kwh", "value_status"]]
        batch = build_windows(measurements, config)
        if not batch.customer_ids:
            continue
        total = np.zeros(dim, dtype=np.float64)
        total_sq = np.zeros(dim, dtype=np.float64)
        first = last = None
        count = 0
        with torch.no_grad():
            for start in range(0, len(batch.values), 64):
                sl = slice(start, start + 64)
                values = torch.as_tensor(batch.values[sl], dtype=torch.float32, device=encoder.device)
                observed = torch.as_tensor(batch.observed[sl], dtype=torch.bool, device=encoder.device)
                calendar = torch.as_tensor(batch.calendar[sl], dtype=torch.float32, device=encoder.device)
                z = encoder.model(values, observed, calendar)
                if isinstance(z, (tuple, list)):
                    z = z[0]
                z = z.detach().to("cpu").numpy()
                if z.ndim != 2 or z.shape[1] != dim:
                    raise ValueError(f"encoder output must be [windows, {dim}], got {z.shape}")
                z = z.astype(np.float64, copy=False)
                total += z.sum(axis=0)
                total_sq += np.square(z).sum(axis=0)
                if first is None:
                    first = z[0].copy()
                last = z[-1].copy()
                count += len(z)
        mean = total / count
        std = np.sqrt(np.maximum(0.0, total_sq / count - np.square(mean)))
        out[row, :dim] = mean
        out[row, dim : 2 * dim] = std
        out[row, 2 * dim :] = last - first
    result = pd.DataFrame(out, index=wanted)
    if cache is not None:
        cache[cache_key] = result
    return result


def fit_projection(raw, train_ids: list[str], n_components: int = 4) -> dict:
    """Fit deterministic imputation/scaling/PCA on training customers only."""
    if n_components != 4:
        raise ProjectionUnavailable("the hybrid projection is exactly four components")
    rawf = _as_frame(raw)
    train = [str(c) for c in train_ids]
    missing = [c for c in train if c not in rawf.index]
    if missing:
        raise ValueError("projection training IDs are missing from raw embeddings")
    X = rawf.loc[train].to_numpy(float)
    if X.ndim != 2 or X.shape[0] < n_components:
        raise ProjectionUnavailable("projection training insufficient: fewer than four customers")
    finite_rows = np.isfinite(X).any(axis=1)
    if finite_rows.sum() < n_components:
        raise ProjectionUnavailable("projection training insufficient: fewer than four usable customers")
    with np.errstate(all="ignore"):
        imp = np.nanmedian(X, axis=0)
    imp = np.where(np.isfinite(imp), imp, 0.0)
    Xi = np.where(np.isfinite(X), X, imp)
    mean = Xi.mean(axis=0)
    scale = Xi.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (Xi - mean) / scale
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    if vt.shape[0] < n_components:
        raise ProjectionUnavailable("projection training insufficient: rank below four")
    components = vt[:n_components].copy()
    pivots = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(n_components), pivots])
    components *= np.where(signs == 0, 1.0, signs)[:, None]
    return {
        "version": 1,
        "n_components": 4,
        "impute": imp.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "components": components.tolist(),
        "training_count": int(len(train)),
        "usable_count": int(finite_rows.sum()),
    }


def transform_projection(raw, ids: list[str], projection: dict) -> pd.DataFrame:
    """Apply a persisted projection without fitting or inspecting labels."""
    if projection.get("n_components") != 4 or len(projection.get("components", [])) != 4:
        raise ValueError("projection must contain exactly four components")
    rawf = _as_frame(raw, ids)
    idx = [str(c) for c in ids]
    X = rawf.reindex(idx).to_numpy(float)
    imp = np.asarray(projection["impute"], float)
    mean = np.asarray(projection["mean"], float)
    scale = np.asarray(projection["scale"], float)
    comp = np.asarray(projection["components"], float)
    if X.shape[1] != len(imp) or comp.shape != (4, X.shape[1]):
        raise ValueError("raw embedding shape does not match projection")
    Xi = np.where(np.isfinite(X), X, imp)
    Z = (Xi - mean) / np.where(scale > 0, scale, 1.0)
    return pd.DataFrame(Z @ comp.T, index=idx, columns=list(EMBEDDING_FEATURES))


def projection_json(projection: dict) -> str:
    """Stable serialization helper used by artifact tests and model metadata."""
    return json.dumps(projection, sort_keys=True, separators=(",", ":"))
