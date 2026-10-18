"""Command line entry point for bounded-memory constrained synthetic augmentation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .synthesis import (
    SCHEMA_VERSION,
    _choose_onset,
    _json_hash,
    _observed_dates,
    _validate_canonical_measurements,
    _validate_labels,
    _validate_splits,
    generate_synthetic,
    load_config,
)
from .synthetic_validation import validate_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate constrained paired PV scenarios")
    parser.add_argument("--canonical-dir", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--held-out-fold", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--meter-mode", required=True, choices=["import_only", "signed_net"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path)
    return parser


def _write_chunk(writer: pq.ParquetWriter | None, path: Path, frame: pd.DataFrame) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer


def _read_source(path: Path, customer_id: str) -> pd.DataFrame:
    # PyArrow applies the predicate while scanning, so the full canonical table is
    # never materialized. One source is held in memory at a time.
    return pd.read_parquet(path, filters=[("customer_id", "=", customer_id)])


def _selected_ids(labels: pd.DataFrame, splits: pd.DataFrame, held_out_fold: int) -> tuple[list[str], dict[str, int]]:
    _validate_labels(labels)
    fold_map = _validate_splits(splits)
    label_ids = labels.customer_id.astype(str)
    selected: list[str] = []
    for customer, installed in zip(label_ids, labels.pv_installed.astype(bool)):
        if installed:
            continue
        if customer not in fold_map:
            raise ValueError(f"missing split assignment for eligible customer {customer}")
        if int(fold_map[customer]) != int(held_out_fold):
            selected.append(customer)
    return sorted(selected), fold_map


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    canonical = args.canonical_dir
    manifest_path = canonical / "run_manifest.json"
    measurements_path = canonical / "measurements.parquet"
    labels_path = canonical / "labels.parquet"
    if not manifest_path.exists() or not measurements_path.exists() or not labels_path.exists():
        raise ValueError("canonical directory is missing required versioned artifacts")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported canonical schema version")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    # Read small control tables first. Excluded positive, held-out, and unlabelled
    # customers are never read from measurements and cannot poison baseline checks.
    labels = pd.read_parquet(labels_path)
    splits = pd.read_parquet(args.splits)
    selected, fold_map = _selected_ids(labels, splits, args.held_out_fold)
    cfg = load_config(args.config)
    eligible: list[str] = []
    excluded: dict[str, int] = {"positive_label": int(labels.pv_installed.astype(bool).sum()),
                                "held_out_fold": int(sum(int(fold_map.get(c, -1)) == int(args.held_out_fold)
                                                         for c in labels.loc[~labels.pv_installed.astype(bool), "customer_id"].astype(str)))}
    daytime_values: list[float] = []
    # First pass computes eligibility and robust scale source-by-source.
    for customer in selected:
        source = _read_source(measurements_path, customer)
        if source.empty:
            excluded["missing_measurements"] = excluded.get("missing_measurements", 0) + 1
            continue
        _validate_canonical_measurements(source)
        days, day_sums = _observed_dates(source, cfg)
        onset = _choose_onset(days, args.seed, customer, cfg)
        if onset is None:
            excluded["insufficient_pre_post_coverage"] = excluded.get("insufficient_pre_post_coverage", 0) + 1
            continue
        positive_days = [value for value in day_sums.values() if value > 0]
        if not positive_days:
            excluded["zero_daytime_scale"] = excluded.get("zero_daytime_scale", 0) + 1
            continue
        eligible.append(customer)
        daytime_values.extend(positive_days)
    if not eligible:
        raise ValueError(f"no eligible non-PV training backgrounds; excluded={excluded}")
    robust_scale = float(pd.Series(daytime_values).median())
    if robust_scale <= 0:
        raise ValueError("eligible training backgrounds have zero daytime scale")

    labels_by_id = labels.set_index(labels.customer_id.astype(str))
    split_rows = splits[splits.customer_id.astype(str).isin(selected)]
    writers: dict[str, pq.ParquetWriter | None] = {name: None for name in ("measurements", "synthetic_labels", "provenance", "latent")}
    counts = {name: 0 for name in writers}
    scenario_counts: dict[str, int] = {}
    try:
        for customer in eligible:
            source = _read_source(measurements_path, customer)
            source_labels = labels_by_id.loc[[customer]].reset_index(drop=True)
            source_splits = split_rows[split_rows.customer_id.astype(str) == customer]
            source_cfg = dict(cfg)
            source_cfg["_robust_daytime_scale"] = robust_scale
            result = generate_synthetic(source, source_labels, source_splits, args.held_out_fold,
                                        args.meter_mode, seed=args.seed, config=source_cfg)
            validate_outputs(result.measurements, result.synthetic_labels, result.provenance,
                             result.latent, held_out_fold=args.held_out_fold, meter_mode=args.meter_mode)
            frames = {"measurements": result.measurements, "synthetic_labels": result.synthetic_labels,
                      "provenance": result.provenance, "latent": result.latent}
            for name, frame in frames.items():
                writers[name] = _write_chunk(writers[name], args.out / f"{name}.parquet", frame)
                counts[name] += len(frame)
            for scenario, count in result.provenance.scenario.value_counts().items():
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + int(count)
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()

    quality = {"schema_version": SCHEMA_VERSION, "generator_version": cfg.get("generator_version"),
               "meter_mode": args.meter_mode, "seed": int(args.seed),
               "eligible_sources": len(eligible), "generated_customers": counts["synthetic_labels"],
               "robust_daytime_scale_kwh": robust_scale, "excluded_sources": excluded,
               "scenario_counts": scenario_counts,
               "hard_gates": {"equations": True, "missingness_preserved": True, "nonnegative_pv": True}}
    (args.out / "quality_report.json").write_text(json.dumps(quality, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    run_manifest = {"schema_version": SCHEMA_VERSION, "generator_version": cfg.get("generator_version"),
                    "seed": int(args.seed), "config": cfg, "config_hash": _json_hash(cfg),
                    "source_manifest_identity": hashlib.sha256(manifest_bytes).hexdigest(),
                    "artifacts": ["measurements.parquet", "synthetic_labels.parquet", "provenance.parquet", "latent.parquet", "quality_report.json"],
                    "counts": counts}
    (args.out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    main()
