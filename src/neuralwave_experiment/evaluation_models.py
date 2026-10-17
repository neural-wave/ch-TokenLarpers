"""Leakage-safe model arms for the evaluation runner.

The public surface is deliberately small: :data:`ARMS` and :func:`fit_predict`.
All fitting is done from the supplied training customers; evaluation windows are
only transformed and scored.
"""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ARMS = ("hourly_logistic", "patchtst_scratch", "patchtst_pretrained")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def _validate_inputs(train_windows: Mapping[str, Any], train_labels: pd.DataFrame,
                     eval_windows: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    if not isinstance(train_labels, pd.DataFrame) or not {"customer_id", "pv_installed"}.issubset(train_labels.columns):
        raise ValueError("train_labels must contain customer_id and pv_installed")
    labels = train_labels.assign(customer_id=train_labels.customer_id.astype(str))
    if labels.customer_id.duplicated().any():
        raise ValueError("train_labels must contain one row per training customer")
    train_ids = {str(x) for x in train_windows}
    eval_ids = {str(x) for x in eval_windows}
    if not train_ids:
        raise ValueError("train_windows cannot be empty")
    if train_ids & eval_ids:
        raise ValueError("training and evaluation customer IDs must be disjoint")
    label_ids = set(labels.customer_id)
    if label_ids != train_ids:
        missing = sorted(train_ids - label_ids)
        extra = sorted(label_ids - train_ids)
        raise ValueError(f"train_labels IDs must exactly match training IDs (missing={missing}, extra={extra})")
    for cid, batch in train_windows.items():
        actual = {str(x) for x in batch.customer_ids}
        if actual and actual != {str(cid)}:
            raise ValueError(f"training batch {cid!r} contains another customer ID")
        if len(batch.values) == 0:
            raise ValueError(f"training customer {cid!r} has no eligible windows")
    for cid, batch in eval_windows.items():
        actual = {str(x) for x in batch.customer_ids}
        if actual and actual != {str(cid)}:
            raise ValueError(f"evaluation batch {cid!r} contains another customer ID")
        if len(batch.values) == 0:
            raise ValueError(f"evaluation customer {cid!r} has no eligible windows")
    return train_ids, eval_ids

def _hourly_features(batch: Any) -> np.ndarray:
    """Return 24 hourly means followed by 24 hourly standard deviations."""
    values = np.asarray(batch.values, dtype=np.float64)[..., 0]
    observed = np.asarray(batch.observed, dtype=bool)[..., 0]
    n, length = values.shape
    result = np.zeros((n, 48), dtype=np.float64)
    for i in range(n):
        epochs = np.asarray(batch.start_epochs[i], dtype=np.int64) + np.arange(length, dtype=np.int64) * 900
        hours = pd.to_datetime(epochs, unit="s", utc=True).tz_convert("Europe/Zurich").hour.to_numpy()
        for hour in range(24):
            valid = observed[i] & (hours == hour)
            if valid.any():
                selected = values[i, valid]
                result[i, hour] = float(selected.mean())
                result[i, 24 + hour] = float(selected.std())
    return result


def _feature_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(features, dtype=np.float64).mean(axis=0)
    scale = np.asarray(features, dtype=np.float64).std(axis=0)
    return mean, np.maximum(scale, 1e-6)


def _labels(batch: Any, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from .task1_detector import assign_window_labels
    targets, valid = assign_window_labels(batch, labels)
    if not bool(valid.any()):
        raise ValueError("training windows contain no non-transition physical labels")
    return targets[valid].astype(np.float64), valid


def _artifact_dir(out_dir: Path, arm: str) -> Path:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save_common(directory: Path, arm: str, train_ids: set[str], config: Mapping[str, Any],
                 seed: int, history: Mapping[str, Any], scaler: Mapping[str, Any], lineage: Mapping[str, Any]) -> None:
    _save_json(directory / "config.json", {"arm": arm, "seed": int(seed), **dict(config)})
    _save_json(directory / "training_ids.json", sorted(train_ids))
    _save_json(directory / "history.json", history)
    _save_json(directory / "lineage.json", lineage)
    np.savez(directory / "scaler.npz", **{k: np.asarray(v) for k, v in scaler.items()})


def _fit_logistic(train_batch: Any, eval_windows: Mapping[str, Any], labels: pd.DataFrame,
                  train_ids: set[str], config: Mapping[str, Any], seed: int, directory: Path) -> dict[str, np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    train_features = _hourly_features(train_batch)
    target, valid = _labels(train_batch, labels)
    train_features = train_features[valid]
    if len(np.unique(target)) < 2:
        raise ValueError("hourly_logistic requires both physical training classes")
    feature_mean, feature_scale = _feature_scaler(train_features)
    x_train = (train_features - feature_mean) / feature_scale
    counts = np.bincount(target.astype(np.int64), minlength=2)
    class_weight = {0: len(target) / (2.0 * counts[0]), 1: len(target) / (2.0 * counts[1])}
    estimator = LogisticRegression(C=1.0, class_weight=class_weight, max_iter=1000,
                                   solver="lbfgs", random_state=int(seed))
    estimator.fit(x_train, target.astype(np.int64))
    history = {"train_loss": [], "class_counts": {str(i): int(counts[i]) for i in range(2)}}
    _save_common(directory, "hourly_logistic", train_ids, config, seed, history,
                 {"feature_mean": feature_mean, "feature_scale": feature_scale},
                 {"training_ids": sorted(train_ids), "pretraining": False,
                  "model": "hourly_profile_logistic", "normalization": "window_local"})
    with (directory / "model.pkl").open("wb") as handle:
        pickle.dump(estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    outputs: dict[str, np.ndarray] = {}
    for cid, original in eval_windows.items():
        features = (_hourly_features(original) - feature_mean) / feature_scale
        probabilities = estimator.predict_proba(features)[:, 1].astype(np.float64)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError(f"hourly_logistic produced invalid probabilities for {cid!r}")
        outputs[str(cid)] = probabilities
    return outputs


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _fit_patchtst(arm: str, train_batch: Any, eval_windows: Mapping[str, Any], labels: pd.DataFrame,
                  train_ids: set[str], config: Mapping[str, Any], seed: int, directory: Path) -> dict[str, np.ndarray]:
    import torch
    from .encoder import EncoderConfig
    from .pretraining import pretrain_encoder
    from .task1_detector import WindowClassifier
    from .encoder_factory import build_encoder

    _seed_all(int(config.get("model_seed", seed)))
    # Window construction already performs leakage-safe local normalization.
    # Do not fit another value scaler across customer folds.
    target, valid = _labels(train_batch, labels)
    if len(np.unique(target)) < 2:
        raise ValueError("PatchTST requires both physical training classes")
    enc_cfg = EncoderConfig(context_length=int(train_batch.values.shape[1]),
                            patch_length=int(config.get("patch_length", 16)),
                            d_model=int(config.get("d_model", 64)),
                            n_heads=int(config.get("n_heads", 4)),
                            n_layers=int(config.get("n_layers", 2)),
                            dropout=float(config.get("dropout", .2)))
    model = WindowClassifier(build_encoder(enc_cfg, backend="official_patchtst"), d_model=enc_cfg.d_model,
                             dropout=enc_cfg.dropout)
    device = str(config.get("device", "cpu"))
    model.to(device)
    pretrain_history: list[float] = []
    if arm == "patchtst_pretrained":
        pretrain_history = pretrain_encoder(model.encoder, train_batch, epochs=int(config.get("pretrain_epochs", 5)),
                                            batch_size=int(config.get("batch_size", 64)),
                                            learning_rate=float(config.get("learning_rate", .001)),
                                            mask_ratio=float(config.get("mask_ratio", .3)),
                                            seed=int(seed), device=device)
        # Pretraining consumes torch's RNG (notably through dropout); reset it
        # before the common supervised budget.
        _seed_all(int(config.get("model_seed", seed)))
    values = torch.as_tensor(train_batch.values[valid], dtype=torch.float32, device=device)
    observed = torch.as_tensor(train_batch.observed[valid], dtype=torch.bool, device=device)
    calendar = torch.as_tensor(train_batch.calendar[valid], dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    counts = np.bincount(target.astype(np.int64), minlength=2)
    pos_weight = torch.tensor(float(counts[0] / max(1, counts[1])), dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", .001)),
                                  weight_decay=float(config.get("weight_decay", .01)))
    # Reset this RNG after optional pretraining so both supervised arms use the
    # same shuffle order and the same supervised budget.
    shuffle_rng = np.random.default_rng(int(seed))
    batch_size = int(config.get("batch_size", 64))
    history: list[float] = []
    model.train()
    for _ in range(int(config.get("epochs", 6))):
        losses: list[float] = []
        for order in np.array_split(shuffle_rng.permutation(len(y)), max(1, int(np.ceil(len(y) / batch_size)))):
            ix = torch.as_tensor(order, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values[ix], observed[ix], calendar[ix]), y[ix])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("PatchTST training produced a non-finite loss")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    torch.save(model.state_dict(), directory / "model.pt")
    _save_common(directory, arm, train_ids, config, seed,
                 {"train_loss": history, "pretrain_loss": pretrain_history},
                 {},
                 {"training_ids": sorted(train_ids), "pretraining": arm == "patchtst_pretrained",
                  "backend": "official_patchtst", "normalization": "window_local",
                  "supervised_epochs": int(config.get("epochs", 6))})
    outputs: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for cid, batch in eval_windows.items():
            values_e = torch.as_tensor(batch.values, dtype=torch.float32, device=device)
            observed_e = torch.as_tensor(batch.observed, dtype=torch.bool, device=device)
            calendar_e = torch.as_tensor(batch.calendar, dtype=torch.float32, device=device)
            probabilities: list[np.ndarray] = []
            for begin in range(0, len(batch.values), batch_size):
                logits = model(values_e[begin:begin + batch_size], observed_e[begin:begin + batch_size],
                               calendar_e[begin:begin + batch_size])
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
            result = np.concatenate(probabilities).astype(np.float64)
            if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
                raise RuntimeError(f"{arm} produced invalid probabilities for {cid!r}")
            outputs[str(cid)] = result
    return outputs


def fit_predict(arm: str, train_windows: dict[str, Any], train_labels: pd.DataFrame,
                eval_windows: dict[str, Any], config: dict, seed: int, out_dir: Path) -> dict[str, np.ndarray]:
    """Fit one evaluation arm using training customers and score every eval window."""
    if arm not in ARMS:
        raise ValueError(f"unknown evaluation arm {arm!r}; expected one of {ARMS}")
    train_ids, _ = _validate_inputs(train_windows, train_labels, eval_windows)
    from .task1_detector import concat_batches
    train_batch = concat_batches([train_windows[cid] for cid in sorted(train_windows)])
    directory = _artifact_dir(Path(out_dir), arm)
    if arm == "hourly_logistic":
        outputs = _fit_logistic(train_batch, eval_windows, train_labels, train_ids, config, seed, directory)
    else:
        outputs = _fit_patchtst(arm, train_batch, eval_windows, train_labels, train_ids, config, seed, directory)
    if set(outputs) != {str(x) for x in eval_windows}:
        raise RuntimeError("model did not produce one probability array per evaluation customer")
    return outputs
