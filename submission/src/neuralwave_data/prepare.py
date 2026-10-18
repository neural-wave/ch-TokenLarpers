from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from .features import write_daily_features
from .io import load_manifest, source_hash, stage_measurements, write_canonical
from .labels import make_splits, parse_extensions, parse_labels, write_frame
from .schema import SCHEMA_VERSION, validate_schema_version

DEFAULT_CONFIG = {"daily_coverage_threshold": 0.9}


def _sources_by_role(sources: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [source for source in sources if source.get("role") == role]


def _read_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_CONFIG)
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("config is unreadable or invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    config = dict(DEFAULT_CONFIG)
    config.update(value)
    try:
        config["daily_coverage_threshold"] = float(config["daily_coverage_threshold"])
    except Exception as exc:
        raise ValueError("daily_coverage_threshold must be numeric") from exc
    if not 0 < config["daily_coverage_threshold"] <= 1:
        raise ValueError("daily_coverage_threshold must be in (0, 1]")
    return config


def prepare(manifest_path: str | Path, output_dir: str | Path, config_path: str | Path | None = None,
            folds: int | None = None, seed: int = 0) -> dict[str, Any]:
    manifest, manifest_file = load_manifest(manifest_path)
    config = _read_config(str(config_path) if config_path else None)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory is populated; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)
    measurement_sources = [s for s in manifest["sources"] if s["role"].endswith("_measurements")]
    if not measurement_sources:
        raise ValueError("manifest must contain at least one measurement source")
    # A private temporary SQLite database bounds memory while preserving arbitrary input order.
    with tempfile.TemporaryDirectory(prefix="neuralwave_stage-") as stage_dir:
        conn = sqlite3.connect(str(Path(stage_dir) / "raw.sqlite"))
        try:
            counts = stage_measurements(measurement_sources, conn)
            measurement_count = write_canonical(conn, output / "measurements.parquet", counts)
        finally:
            conn.close()
    daily_count = write_daily_features(output / "measurements.parquet", output / "daily_features.parquet", float(config["daily_coverage_threshold"]))
    labels = None
    label_sources = _sources_by_role(manifest["sources"], "development_labels")
    if len(label_sources) > 1:
        raise ValueError("multiple development_labels sources are unsupported")
    if label_sources:
        labels = parse_labels(label_sources[0])
        write_frame(labels, output / "labels.parquet", "labels")
    extensions_sources = _sources_by_role(manifest["sources"], "extension_labels")
    if len(extensions_sources) > 1:
        raise ValueError("multiple extension_labels sources are unsupported")
    if extensions_sources:
        extensions = parse_extensions(extensions_sources[0], labels)
        write_frame(extensions, output / "extensions.parquet", "extensions")
    actual_folds = None
    if folds is not None:
        if labels is None:
            raise ValueError("folds require a development_labels source")
        split_frame, actual_folds = make_splits(labels, int(folds), int(seed))
        write_frame(split_frame, output / "splits.parquet", "splits")
    source_records = []
    for source in manifest["sources"]:
        source_records.append({"role": source["role"], "path": source["path"], "sha256": source_hash(source["_path"]),
                               "delimiter": source["delimiter"], "decimal": source["decimal"], "encoding": source["encoding"]})
    software_versions = {"python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pyarrow.__version__}
    for package_name in ("numpy", "scikit-learn"):
        try:
            software_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            software_versions[package_name] = "unavailable"
    manifest_out = {"schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_roles": [s["role"] for s in manifest["sources"]], "sources": source_records,
                    "software_versions": software_versions,
                    "settings": {"epoch_unit": "seconds", "timezone": "Europe/Zurich", **config,
                                 "requested_folds": folds, "actual_folds": actual_folds, "seed": int(seed)},
                    "counts": {**counts, "canonical_measurement_rows": measurement_count, "daily_feature_rows": daily_count,
                               "label_rows": 0 if labels is None else len(labels)},
                    "artifacts": ["measurements.parquet", "daily_features.parquet"] + (["labels.parquet"] if labels is not None else [])
                    + (["extensions.parquet"] if extensions_sources else []) + (["splits.parquet"] if actual_folds is not None else [])}
    (output / "run_manifest.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True), encoding="utf-8")
    (output / "qa_report.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "counts": manifest_out["counts"]}, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare canonical NeuralWave measurements")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--folds", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        prepare(args.manifest, args.out, args.config, args.folds, args.seed)
    except Exception as exc:
        print(f"preprocessing failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
