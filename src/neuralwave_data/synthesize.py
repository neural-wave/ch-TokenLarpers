"""Command line entry point for constrained synthetic augmentation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .synthesis import SCHEMA_VERSION, generate_synthetic, load_config, write_result
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    canonical = args.canonical_dir
    manifest_path = canonical / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError("canonical directory is missing run_manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported canonical schema version")
    measurements = pd.read_parquet(canonical / "measurements.parquet")
    labels = pd.read_parquet(canonical / "labels.parquet")
    splits = pd.read_parquet(args.splits)
    cfg = load_config(args.config)
    result = generate_synthetic(measurements, labels, splits, args.held_out_fold,
                                args.meter_mode, seed=args.seed, config=cfg)
    # Validation is independent of generation arithmetic and is part of the CLI hard gate.
    validate_outputs(result.measurements, result.synthetic_labels, result.provenance,
                     result.latent, held_out_fold=args.held_out_fold, meter_mode=args.meter_mode)
    identity = hashlib.sha256(manifest_bytes).hexdigest()
    write_result(result, args.out, identity, args.seed, cfg)
    return 0


if __name__ == "__main__":
    main()
