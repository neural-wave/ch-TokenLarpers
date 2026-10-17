"""Run the identical experiment protocol with both encoder backends."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .experiment import _load_config, run_experiment


def run_comparison(canonical_dir: str | Path, out: str | Path, config: dict[str, Any] | None = None,
                   seed: int | None = None, device: str = "cpu", synthetic_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise ValueError("comparison output directory must be new and empty; refusing overwrite")
    root.mkdir(parents=True, exist_ok=True)
    base = dict(config or {})
    results: dict[str, Any] = {}
    for backend in ("custom", "official_patchtst"):
        run_dir = root / backend
        cfg = dict(base)
        cfg["encoder_backend"] = backend
        started = time.perf_counter()
        result = run_experiment(canonical_dir, run_dir, cfg, synthetic_dir=synthetic_dir, seed=seed, device=device)
        result["runtime_seconds_wall"] = time.perf_counter() - started
        results[backend] = result
    baseline_hashes = {
        backend: hashlib.sha256((root / backend / "predictions_baseline.csv").read_bytes()).hexdigest()
        for backend in results
    }
    baseline_equal = (results["custom"]["pooled_metrics"].get("baseline")
                      == results["official_patchtst"]["pooled_metrics"].get("baseline")
                      and baseline_hashes["custom"] == baseline_hashes["official_patchtst"])
    comparison = {
        "backends": results,
        "baseline_prediction_sha256": baseline_hashes,
        "protocol": {"canonical_dir": str(canonical_dir), "seed": seed,
                      "config": base, "device": device, "synthetic_dir": str(synthetic_dir) if synthetic_dir else None},
        "baseline_metrics_equal": baseline_equal,
    }
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    if not baseline_equal:
        raise RuntimeError("backend comparison baseline outputs differ; protocol is not matched")
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare custom and official PatchTST encoders")
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_comparison(args.canonical_dir, args.out, _load_config(args.config), args.seed, args.device, args.synthetic_dir)
    return 0


if __name__ == "__main__":
    main()
