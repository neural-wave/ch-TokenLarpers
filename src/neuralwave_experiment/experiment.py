"""Offline outer-fold experiment runner.

This module is intentionally a plain CLI: all data and model artifacts are local,
fold boundaries are explicit, and no estimator or checkpoint is fetched remotely.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .features import aggregate_customer_features, date_candidates
from .heads import CapacityHead, DateHead, PresenceHead, fit_heads, training_date_labels
from .scoring import score_predictions, validate_predictions
from .windows import WindowConfig, build_windows
from .encoder import EncoderConfig, NumericPatchEncoder, encode_windows
from .pretraining import pretrain_encoder

def default_config() -> dict[str, Any]:
    return {"context_length": 672, "stride": 96, "min_observed_fraction": .75,
            "patch_length": 16, "d_model": 64, "n_heads": 4, "n_layers": 2,
            "dropout": .1, "pretrain_epochs": 5, "finetune_epochs": 3,
            "batch_size": 32, "learning_rate": .001, "finetune_learning_rate": .0001,
            "mask_ratio": .3, "seed": 0, "max_windows_per_customer": 128,
            "include_unlabelled": False}


def _load_config(path: str | Path | None) -> dict[str, Any]:
    cfg = default_config()
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("experiment config must be a JSON object")
        cfg.update(loaded)
    return cfg


def _seed(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed))
    try:
        import torch
        torch.manual_seed(int(seed)); torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _require_canonical(directory: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    required = ["run_manifest.json", "measurements.parquet", "labels.parquet", "splits.parquet"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"canonical directory missing required artifacts: {', '.join(missing)}")
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported canonical schema_version; expected 1")
    labels = pd.read_parquet(directory / "labels.parquet")
    splits = pd.read_parquet(directory / "splits.parquet")
    if labels.customer_id.astype(str).duplicated().any() or splits.customer_id.astype(str).duplicated().any():
        raise ValueError("labels and splits must contain one row per customer")
    labels.customer_id = labels.customer_id.astype(str); splits.customer_id = splits.customer_id.astype(str)
    if set(labels.customer_id) != set(splits.customer_id):
        raise ValueError("splits must cover every labelled customer exactly")
    extensions = pd.read_parquet(directory / "extensions.parquet") if (directory / "extensions.parquet").is_file() else pd.DataFrame()
    return directory / "measurements.parquet", labels, splits, extensions, manifest


def _read_measurements(path: Path, customer_ids: Iterable[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(path, filters=[("customer_id", "=", str(customer))])
              for customer in sorted(set(map(str, customer_ids)))]
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result["customer_id"] = result.customer_id.astype(str)
    return result




def _windows_by_customer(measurements: pd.DataFrame | Path, ids: Iterable[str], cfg: Mapping[str, Any]) -> dict[str, object]:
    result = {}
    window_cfg = WindowConfig(context_length=int(cfg["context_length"]), stride=int(cfg["stride"]),
                              min_observed_fraction=float(cfg["min_observed_fraction"]),
                              max_windows_per_customer=int(cfg["max_windows_per_customer"]))
    for customer in ids:
        frame = (_read_measurements(measurements, [customer]) if isinstance(measurements, Path)
                 else measurements.loc[measurements.customer_id.astype(str).eq(str(customer))])
        if frame.empty:
            raise ValueError(f"customer {customer!r} has no measurement rows")
        batch = build_windows(frame, window_cfg)
        if not batch.customer_ids:
            raise ValueError(f"customer {customer!r} has zero eligible windows")
        result[str(customer)] = batch
    return result

def _initial_batch(batch: object, extensions: pd.DataFrame) -> object:
    if extensions is None or extensions.empty or "extension_date" not in extensions:
        return batch
    cutoff = {}
    for customer, group in extensions.assign(customer_id=extensions.customer_id.astype(str)).groupby("customer_id"):
        cutoff[str(customer)] = pd.to_datetime(group.extension_date).dt.date.min()
    keep = np.array([str(c) not in cutoff or center < cutoff[str(c)] for c, center in zip(batch.customer_ids, batch.center_dates)])
    from dataclasses import replace
    indices = np.flatnonzero(keep)
    if len(indices) == 0:
        raise ValueError("extensions exclude every capacity window for a customer")
    return replace(batch, values=batch.values[indices], observed=batch.observed[indices],
                   calendar=batch.calendar[indices], magnitude=batch.magnitude[indices],
                   customer_ids=[batch.customer_ids[i] for i in indices],
                   center_dates=[batch.center_dates[i] for i in indices], start_epochs=batch.start_epochs[indices])


def _concat_batches(batches: list[object]) -> object:
    if not batches:
        raise ValueError("no windows available for representation fitting")
    from dataclasses import replace
    return replace(batches[0], values=np.concatenate([b.values for b in batches]), observed=np.concatenate([b.observed for b in batches]),
                   calendar=np.concatenate([b.calendar for b in batches]), magnitude=np.concatenate([b.magnitude for b in batches]),
                   customer_ids=sum((list(b.customer_ids) for b in batches), []), center_dates=sum((list(b.center_dates) for b in batches), []),
                   start_epochs=np.concatenate([b.start_epochs for b in batches]))


def _finetune(model: object, batch: object, labels: pd.DataFrame, epochs: int, lr: float, seed: int,
               device: str, batch_size: int = 32) -> list[float]:
    """Fine-tune only the final encoder block on known active-window targets."""
    import torch
    _seed(seed)
    label_map = labels.assign(customer_id=labels.customer_id.astype(str)).set_index("customer_id")
    valid, targets = [], []
    for index, (customer, center) in enumerate(zip(batch.customer_ids, batch.center_dates)):
        if str(customer) not in label_map.index:
            continue
        row = label_map.loc[str(customer)]
        if not bool(row.pv_installed):
            valid.append(index); targets.append(0.0); continue
        onset = row.get("pv_installation_date")
        if bool(row.get("date_target_valid", True)) is False or pd.isna(onset):
            continue
        valid.append(index); targets.append(float(pd.Timestamp(center).date() >= pd.Timestamp(onset).date()))
    if not valid:
        return []
    values = torch.from_numpy(np.asarray(batch.values)[valid]).to(device)
    observed = torch.from_numpy(np.asarray(batch.observed)[valid]).to(device)
    calendar = torch.from_numpy(np.asarray(batch.calendar)[valid]).to(device)
    target = torch.tensor(targets, dtype=torch.float32, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    params = list(model.transformer.layers[-1].parameters()) if hasattr(model, "transformer") else []
    for parameter in params:
        parameter.requires_grad_(True)
    dimension = int(getattr(model, "config").d_model)
    head = torch.nn.Linear(dimension, 1).to(device)
    optimizer = torch.optim.Adam(params + list(head.parameters()), lr=float(lr))
    history = []
    model.train(); head.train()
    rng = np.random.default_rng(seed)
    for _ in range(max(0, int(epochs))):
        losses = []
        order_all = rng.permutation(len(valid))
        for start in range(0, len(valid), max(1, int(batch_size))):
            order = order_all[start:start + max(1, int(batch_size))]
            ix = torch.as_tensor(order, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(model(values[ix], observed[ix], calendar[ix])).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[ix])
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    model.eval()
    return history


def _lineage_for_fold(synthetic_dir: Path | None, fold: int, train_ids: set[str], heldout_ids: set[str],
                      fold_map: Mapping[str, int], canonical_manifest: Mapping[str, Any],
                      canonical_manifest_identity: str | None = None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if synthetic_dir is None:
        return None, None
    folder = synthetic_dir / f"fold_{int(fold)}"
    if not folder.is_dir():
        raise ValueError(f"synthetic-dir must contain matching fold_{fold}/ directory; refusing unsafe reuse")
    manifest_path = folder / "run_manifest.json"
    required = [manifest_path, folder / "measurements.parquet", folder / "provenance.parquet", folder / "synthetic_labels.parquet"]
    if not all(path.is_file() for path in required):
        raise ValueError(f"synthetic fold {fold} is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported synthetic schema_version")
    if canonical_manifest_identity and manifest.get("source_manifest_identity") != canonical_manifest_identity:
        raise ValueError("synthetic source manifest identity mismatch")
    provenance = pd.read_parquet(folder / "provenance.parquet")
    syn_measurements = pd.read_parquet(folder / "measurements.parquet")
    syn_labels = pd.read_parquet(folder / "synthetic_labels.parquet")
    required_cols = {"customer_id", "source_customer_id", "source_fold"}
    if not required_cols.issubset(provenance.columns):
        raise ValueError("synthetic provenance missing source ancestry columns")
    if provenance.source_customer_id.isna().any() or provenance.source_customer_id.astype(str).isin({"", "nan", "None"}).any():
        raise ValueError("synthetic provenance has missing source_customer_id")
    provenance["source_customer_id"] = provenance.source_customer_id.astype(str)
    provenance["customer_id"] = provenance.customer_id.astype(str)
    syn_ids = set(provenance.customer_id)
    if syn_ids != set(syn_labels.customer_id.astype(str)):
        raise ValueError("synthetic labels/provenance customer IDs disagree")
    if not set(syn_measurements.customer_id.astype(str)).issubset(syn_ids):
        raise ValueError("synthetic measurements contain unprovenanced IDs")
    if set(provenance.source_customer_id) & heldout_ids:
        raise ValueError("synthetic ancestry includes held-out customer")
    if not set(provenance.source_customer_id).issubset(train_ids):
        raise ValueError("synthetic ancestry is not a training customer")
    expected_folds = provenance.source_customer_id.map(fold_map).astype("Int64")
    if not expected_folds.eq(provenance.source_fold.astype(int)).all():
        raise ValueError("synthetic provenance source_fold disagrees with canonical splits")
    if "pv_capacity_kwp" in syn_labels and syn_labels.pv_capacity_kwp.notna().any():
        raise ValueError("synthetic capacity labels cannot supervise capacity")
    return syn_measurements, syn_labels


def run_experiment(canonical_dir: str | Path, out_dir: str | Path, config: Mapping[str, Any] | None = None,
                   synthetic_dir: str | Path | None = None, seed: int | None = None, device: str = "cpu") -> dict[str, Any]:
    cfg = default_config(); cfg.update(dict(config or {})); cfg["seed"] = int(cfg["seed"] if seed is None else seed)
    _seed(cfg["seed"])
    canonical = Path(canonical_dir); out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be new and empty; refusing overwrite")
    out.mkdir(parents=True, exist_ok=True)
    measurement_path, labels, splits, extensions, manifest = _require_canonical(canonical)
    all_ids = labels.customer_id.astype(str).tolist()
    fold_map = dict(zip(splits.customer_id.astype(str), splits.fold.astype(int)))
    folds = sorted(set(fold_map.values()))
    unlabelled_ids: list[str] = []
    if bool(cfg.get("include_unlabelled", False)):
        import pyarrow.parquet as pq
        seen: set[str] = set()
        for part in pq.ParquetFile(measurement_path).iter_batches(columns=["customer_id"]):
            seen.update(map(str, part.column("customer_id").to_pylist()))
        unlabelled_ids = sorted(seen - set(all_ids))
    windows = _windows_by_customer(measurement_path, all_ids + unlabelled_ids, cfg)
    measurements = _read_measurements(measurement_path, all_ids)
    arms = {"baseline": [], "frozen": [], "finetuned": []}
    fold_metrics: dict[str, dict[str, Any]] = {}
    lineage_audit: list[dict[str, Any]] = []
    for fold in folds:
        head_records: dict[str, Any] = {}
        heldout = {c for c in all_ids if fold_map[c] == fold}
        train = set(all_ids) - heldout
        train_labels = labels[labels.customer_id.astype(str).isin(train)].copy()
        syn_measurements, syn_labels = _lineage_for_fold(
            Path(synthetic_dir) if synthetic_dir else None, fold, train, heldout, fold_map, manifest,
            hashlib.sha256((canonical / "run_manifest.json").read_bytes()).hexdigest())
        lineage_audit.append({"fold": int(fold), "training_customers": sorted(train), "heldout_customers": sorted(heldout), "synthetic": syn_measurements is not None})
        training_batches = [windows[c] for c in sorted(train)]
        if bool(cfg.get("include_unlabelled", False)):
            training_batches.extend(windows[c] for c in unlabelled_ids)
        if syn_measurements is not None and not syn_measurements.empty:
            syn_ids = sorted(syn_measurements.customer_id.astype(str).unique())
            syn_windows = _windows_by_customer(syn_measurements.assign(customer_id=syn_measurements.customer_id.astype(str)), syn_ids, cfg)
            training_batches.extend(syn_windows[c] for c in syn_ids)
        train_batch = _concat_batches(training_batches)
        eval_batch = _concat_batches([windows[c] for c in all_ids])
        # Baseline and embeddings each get exactly the same customer fold.
        base_features = aggregate_customer_features(eval_batch)
        base_lookup = {c: i for i, c in enumerate(base_features.customer_ids)}
        train_rows = [base_lookup[c] for c in sorted(train)]
        base_cap = aggregate_customer_features(_initial_batch(eval_batch, extensions))
        base_cap_lookup = {c: i for i, c in enumerate(base_cap.customer_ids)}
        base_candidates = date_candidates(measurements, all_ids)
        presence, date_head, cap_head, head_audit = fit_heads(
            base_features.matrix[train_rows], sorted(train), train_labels, measurements,
            capacity_features=base_cap.matrix[[base_cap_lookup[c] for c in sorted(train)]], candidates=base_candidates)
        head_records["baseline"] = {"presence": presence.to_dict(), "date": date_head.to_dict(),
                                    "capacity": cap_head.to_dict(), "audit": head_audit}
        base_pred = _predictions(base_features, all_ids, presence, date_head, cap_head, measurements,
                                 capacity_features=base_cap.matrix)
        arms["baseline"].append(base_pred[base_pred.customer_id.isin(heldout)])
        enc_cfg = EncoderConfig(context_length=int(cfg["context_length"]), patch_length=int(cfg["patch_length"]), d_model=int(cfg["d_model"]),
                                n_heads=int(cfg["n_heads"]), n_layers=int(cfg["n_layers"]), dropout=float(cfg["dropout"]))
        import torch
        model = NumericPatchEncoder(enc_cfg).to(device)
        pretrain_history = pretrain_encoder(model, train_batch, epochs=int(cfg["pretrain_epochs"]), batch_size=int(cfg["batch_size"]),
                                             learning_rate=float(cfg["learning_rate"]), mask_ratio=float(cfg["mask_ratio"]), seed=cfg["seed"] + fold, device=device)
        frozen_emb = encode_windows(model, eval_batch, batch_size=int(cfg["batch_size"]), device=device)
        frozen_features = aggregate_customer_features(eval_batch, frozen_emb)
        lookup = {c: i for i, c in enumerate(frozen_features.customer_ids)}
        rows = [lookup[c] for c in sorted(train)]
        frozen_cap = aggregate_customer_features(_initial_batch(eval_batch, extensions), frozen_emb)
        frozen_cap_lookup = {c: i for i, c in enumerate(frozen_cap.customer_ids)}
        frozen_candidates = date_candidates(measurements, all_ids, window_batch=eval_batch, window_embeddings=frozen_emb)
        presence, date_head, cap_head, head_audit = fit_heads(
            frozen_features.matrix[rows], sorted(train), train_labels, measurements,
            capacity_features=frozen_cap.matrix[[frozen_cap_lookup[c] for c in sorted(train)]],
            candidates=frozen_candidates)
        head_records["frozen"] = {"presence": presence.to_dict(), "date": date_head.to_dict(),
                                  "capacity": cap_head.to_dict(), "audit": head_audit}
        frozen_pred = _predictions(frozen_features, all_ids, presence, date_head, cap_head, measurements,
                                   capacity_features=frozen_cap.matrix, candidates=frozen_candidates)
        arms["frozen"].append(frozen_pred[frozen_pred.customer_id.isin(heldout)])
        fine_model = copy.deepcopy(model)
        fine_labels = train_labels
        if syn_labels is not None and not syn_labels.empty:
            fine_labels = pd.concat([train_labels, syn_labels], ignore_index=True, sort=False)
        fine_history = _finetune(fine_model, train_batch, fine_labels, int(cfg["finetune_epochs"]),
                                 float(cfg["finetune_learning_rate"]), cfg["seed"] + 1000 + fold, device,
                                 int(cfg["batch_size"]))
        fine_emb = encode_windows(fine_model, eval_batch, batch_size=int(cfg["batch_size"]), device=device)
        fine_features = aggregate_customer_features(eval_batch, fine_emb)
        lookup = {c: i for i, c in enumerate(fine_features.customer_ids)}; rows = [lookup[c] for c in sorted(train)]
        fine_cap = aggregate_customer_features(_initial_batch(eval_batch, extensions), fine_emb)
        fine_cap_lookup = {c: i for i, c in enumerate(fine_cap.customer_ids)}
        fine_candidates = date_candidates(measurements, all_ids, window_batch=eval_batch, window_embeddings=fine_emb)
        presence, date_head, cap_head, head_audit = fit_heads(
            fine_features.matrix[rows], sorted(train), train_labels, measurements,
            capacity_features=fine_cap.matrix[[fine_cap_lookup[c] for c in sorted(train)]],
            candidates=fine_candidates)
        head_records["finetuned"] = {"presence": presence.to_dict(), "date": date_head.to_dict(),
                                     "capacity": cap_head.to_dict(), "audit": head_audit}
        fine_pred = _predictions(fine_features, all_ids, presence, date_head, cap_head, measurements,
                                 capacity_features=fine_cap.matrix, candidates=fine_candidates)
        arms["finetuned"].append(fine_pred[fine_pred.customer_id.isin(heldout)])
        torch.save(model.state_dict(), out / f"encoder_fold_{fold}.pt")
        torch.save(fine_model.state_dict(), out / f"encoder_finetuned_fold_{fold}.pt")
        (out / f"heads_fold_{fold}.json").write_text(json.dumps({
            "arms": head_records, "encoder_config": asdict(enc_cfg)}, indent=2, default=str), encoding="utf-8")
        (out / f"history_fold_{fold}.json").write_text(json.dumps({"pretraining": pretrain_history, "finetune": fine_history}, indent=2), encoding="utf-8")
        for arm, pred in (("baseline", base_pred), ("frozen", frozen_pred), ("finetuned", fine_pred)):
            fold_metrics.setdefault(arm, {})[str(fold)] = score_predictions(labels[labels.customer_id.isin(heldout)], pred[pred.customer_id.isin(heldout)])
    pooled = {}
    for arm, parts in arms.items():
        combined = pd.concat(parts, ignore_index=True)
        pred = validate_predictions(combined, all_ids)
        # Each fold prediction should be the held-out row; enforce this rather than
        # silently choosing a duplicate if a fold implementation regresses.
        pred.to_csv(out / f"predictions_{arm}.csv", sep=";", index=False)
        pooled[arm] = score_predictions(labels, pred)
    result = {"fold_metrics": fold_metrics, "pooled_metrics": pooled, "lineage_audit": lineage_audit, "arms": list(arms)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (out / "experiment_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (out / "feature_definitions.json").write_text(json.dumps({"baseline": "customer mean/std/first-last of raw windows", "frozen": "same plus encoder embedding", "finetuned": "same plus active-window fine-tuned embedding"}, indent=2), encoding="utf-8")
    identity = hashlib.sha256((canonical / "run_manifest.json").read_bytes()).hexdigest()
    import torch, sklearn
    (out / "run_manifest.json").write_text(json.dumps({
        "schema_version": 1, "canonical_manifest_sha256": identity, "seed": cfg["seed"],
        "library": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__}}, indent=2), encoding="utf-8")
    return result


def _predictions(features: object, all_ids: list[str], presence: PresenceHead, date_head: DateHead,
                capacity: CapacityHead | None, measurements: pd.DataFrame,
                capacity_features: np.ndarray | None = None, candidates: list[object] | None = None) -> pd.DataFrame:
    lookup = {c: i for i, c in enumerate(features.customer_ids)}
    x = np.vstack([features.matrix[lookup[c]] for c in all_ids])
    cap_x = x if capacity_features is None else np.vstack([capacity_features[lookup[c]] for c in all_ids])
    probabilities = presence.probability(x); detected = probabilities >= .5
    candidates = date_candidates(measurements, all_ids) if candidates is None else candidates
    rows = []
    for i, customer in enumerate(all_ids):
        chosen = date_head.choose(candidates, measurements, customer) if detected[i] else None
        cap = float(capacity.predict(cap_x[i:i + 1])[0]) if detected[i] and capacity is not None else np.nan
        rows.append({"customer_id": customer, "pv_installed": bool(detected[i]), "presence_probability": float(probabilities[i]),
                     "predicted_installation_date": chosen, "predicted_capacity_kwp": cap})
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline time-series embedding experiment")
    parser.add_argument("--canonical-dir", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--config"); parser.add_argument("--synthetic-dir"); parser.add_argument("--seed", type=int); parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_experiment(args.canonical_dir, args.out, _load_config(args.config), args.synthetic_dir, args.seed, args.device)
    return 0


if __name__ == "__main__":
    main()
