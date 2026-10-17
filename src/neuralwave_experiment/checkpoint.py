"""Private resumable training checkpoints and compact frozen encoder artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .encoder import EncoderConfig
from .encoder_factory import build_encoder

ARTIFACT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "patchtst-unlabelled-pretrain-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode()); tensor = state[name].detach().cpu().contiguous()
        h.update(str(tensor.dtype).encode()); h.update(repr(tuple(tensor.shape)).encode()); h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _validate_encoder_config(config: Mapping[str, Any]) -> EncoderConfig:
    required = {"context_length", "patch_length", "d_model", "n_heads", "n_layers", "dropout"}
    if set(config) != required:
        raise ValueError("encoder_config must contain exactly the encoder architecture fields")
    try:
        return EncoderConfig(context_length=int(config["context_length"]), patch_length=int(config["patch_length"]),
                             d_model=int(config["d_model"]), n_heads=int(config["n_heads"]), n_layers=int(config["n_layers"]),
                             dropout=float(config["dropout"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid encoder_config") from exc


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def export_encoder_artifact(directory: Path, model: nn.Module, *, encoder_config: dict, window_config: dict, source: dict) -> Path:
    """Write exactly ``encoder.pt`` and ``encoder.json`` into a new directory."""
    directory = Path(directory)
    if directory.exists():
        if not directory.is_dir() or any(directory.iterdir()):
            raise FileExistsError("encoder artifact directory must be new and empty")
    if getattr(model, "backend", None) != "official_patchtst":
        raise ValueError("inference artifacts require the official_patchtst backend")
    required_source = {"population", "labels_used", "train_customer_count", "val_customer_count",
                       "train_exclusion_hash", "val_exclusion_hash"}
    if not required_source.issubset(source):
        raise ValueError("source must contain provenance population, labels_used and exclusion audits")
    if any("customer_id" in str(key).lower() or str(key).lower().endswith("_ids") for key in source):
        raise ValueError("source metadata must not contain customer IDs")
    if source.get("population") != "unlabelled_only" or source.get("labels_used") is not False:
        raise ValueError("encoder artifact provenance must be unlabelled_only and label-free")
    config_obj = _validate_encoder_config(encoder_config)
    if not isinstance(window_config, dict) or not window_config:
        raise ValueError("window_config is required")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    weights_path = directory / "encoder.pt"
    _atomic_torch_save(state, weights_path)
    metadata = {"schema_version": ARTIFACT_SCHEMA_VERSION, "backend": "official_patchtst",
                "encoder_config": {"context_length": config_obj.context_length, "patch_length": config_obj.patch_length,
                                   "d_model": config_obj.d_model, "n_heads": config_obj.n_heads,
                                   "n_layers": config_obj.n_layers, "dropout": config_obj.dropout},
                "window_config": window_config, "source": source,
                "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest()}
    meta_path = directory / "encoder.json"
    fd, tmp = tempfile.mkstemp(prefix=".encoder.", dir=directory); os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_bytes(_canonical(metadata)); os.replace(tmp_path, meta_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return directory


def load_encoder_artifact(directory: Path, *, device: str = "cpu") -> tuple[nn.Module, dict]:
    directory = Path(directory)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != {"encoder.pt", "encoder.json"}:
        raise ValueError("encoder artifact must contain exactly encoder.pt and encoder.json")
    try: metadata = json.loads((directory / "encoder.json").read_text())
    except Exception as exc: raise ValueError("encoder metadata is invalid") from exc
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION or metadata.get("backend") != "official_patchtst":
        raise ValueError("unsupported encoder artifact schema/backend")
    expected = metadata.get("weights_sha256")
    actual = hashlib.sha256((directory / "encoder.pt").read_bytes()).hexdigest()
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("encoder weights hash mismatch")
    try: config = _validate_encoder_config(metadata["encoder_config"])
    except (KeyError, TypeError) as exc: raise ValueError("invalid encoder metadata") from exc
    source = metadata.get("source")
    if not isinstance(source, dict) or source.get("population") != "unlabelled_only" or source.get("labels_used") is not False:
        raise ValueError("encoder provenance is not unlabelled-only")
    try:
        model = build_encoder(config, backend="official_patchtst")
        state = torch.load(directory / "encoder.pt", map_location=device, weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
            raise ValueError("encoder weights are not a tensor state dictionary")
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        if isinstance(exc, ValueError): raise
        raise ValueError("encoder weights cannot be loaded") from exc
    model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, metadata


def _safe_rng(value: Any) -> Any:
    if isinstance(value, np.ndarray): return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, dict): return {str(k): _safe_rng(v) for k, v in value.items()}
    if isinstance(value, tuple): return [_safe_rng(v) for v in value]
    if isinstance(value, list): return [_safe_rng(v) for v in value]
    return value


def _restore_safe(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__ndarray__", "dtype"}: return np.asarray(value["__ndarray__"], dtype=value["dtype"])
    if isinstance(value, dict): return {k: _restore_safe(v) for k, v in value.items()}
    if isinstance(value, list): return [_restore_safe(v) for v in value]
    return value


def capture_rng_state(mask_rng: Any | None = None) -> dict[str, Any]:
    py = random.getstate(); ns = np.random.get_state()
    state = {"python": [py[0], list(py[1]), py[2]],
             "numpy": [ns[0], ns[1].tolist(), int(ns[2]), int(ns[3]), float(ns[4])],
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda"] = torch.cuda.get_rng_state_all()
    if mask_rng is not None: state["mask_rng"] = _safe_rng(mask_rng.bit_generator.state)
    return state


def restore_rng_state(state: Mapping[str, Any], mask_rng: Any | None = None) -> None:
    py = state["python"]; random.setstate((py[0], tuple(py[1]), py[2]))
    ns = state["numpy"]; np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), int(ns[2]), int(ns[3]), float(ns[4])))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])
    if mask_rng is not None and "mask_rng" in state: mask_rng.bit_generator.state = _restore_safe(state["mask_rng"])


def save_pretrain_checkpoint(path: Path, state: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> None:
    payload = {"schema": CHECKPOINT_SCHEMA, "metadata": dict(metadata or {}), "state": dict(state)}
    _atomic_torch_save(payload, Path(path))


def load_pretrain_checkpoint(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer | None = None,
                             scaler: Any | None = None, expected_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try: payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc: raise ValueError("checkpoint is corrupt or unreadable") from exc
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA or not isinstance(payload.get("state"), dict):
        raise ValueError("unsupported or malformed pretraining checkpoint")
    metadata = payload.get("metadata", {})
    if expected_metadata:
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected: raise ValueError(f"checkpoint metadata mismatch: {key}")
    state = payload["state"]
    try: model.load_state_dict(state["model"], strict=True)
    except Exception as exc: raise ValueError("checkpoint model state mismatch") from exc
    if optimizer is not None:
        if "optimizer" not in state: raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None: scaler.load_state_dict(state["scaler"])
    return payload


def save_training_checkpoint(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
                             scaler: Any | None, sampler: Mapping[str, Any], progress: Mapping[str, Any],
                             config: Mapping[str, Any], source_identity: Mapping[str, Any]) -> None:
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict() if scaler is not None else None,
             "sampler": dict(sampler), "progress": dict(progress), "rng": capture_rng_state()}
    metadata = {"config": dict(config), "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(), **dict(source_identity)}
    save_pretrain_checkpoint(path, state, metadata)


__all__ = ["export_encoder_artifact", "load_encoder_artifact", "save_pretrain_checkpoint", "load_pretrain_checkpoint",
           "save_training_checkpoint", "capture_rng_state", "restore_rng_state"]
