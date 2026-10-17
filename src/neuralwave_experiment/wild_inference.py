"""Score unlabelled customers with a trained Task 1 window classifier.

A plausibility check on data outside the labelled development set: there is
no ground truth, so the output is a decision count and score ranges, not an
accuracy.  Customer identifiers are passed on the command line and the
per-customer table is written only to the private ``--out`` JSON; the
terminal shows aggregates unless ``--show-customers`` is given.

    python -m neuralwave_experiment.wild_inference \
        --unlabelled-csv /private/energy_consumption_unlabelled_dev.csv \
        --weights /private/task1_results/task1_model_fold_0.pt \
        --id-prefix POD --id-start 100 --id-count 20 \
        --out /private/task1_results/wild_batch.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from neuralwave_experiment.encoder_factory import build_encoder
from neuralwave_experiment.task1_detector import TASK1_ENCODER_CONFIG, WindowClassifier, aggregate_customer_score
from neuralwave_experiment.windows import WindowConfig, build_windows


def target_ids(args: argparse.Namespace) -> list[str]:
    if args.customer_ids:
        ids = [c.strip() for c in args.customer_ids.split(",") if c.strip()]
    elif args.id_count is not None:
        ids = [f"{args.id_prefix}{i}" for i in range(args.id_start, args.id_start + args.id_count)]
    else:
        raise SystemExit("give --customer-ids or --id-count (with --id-prefix/--id-start)")
    if not ids:
        raise SystemExit("no customer ids requested")
    return ids


def read_customers(csv_path: Path, ids: Sequence[str], prefix: str, chunksize: int, max_rows: int | None) -> pd.DataFrame:
    """Stream the unlabelled file and keep only the requested customers.

    Rows whose id does not start with ``prefix`` are dropped first (the
    organizer rule for unlabelled sources).  ``max_rows`` stops the scan early
    once that many matching rows are collected; the file is interleaved by
    customer, so an early stop can truncate histories.
    """
    wanted = set(ids)
    parts: list[pd.DataFrame] = []
    collected = 0
    for chunk in pd.read_csv(csv_path, sep=";", chunksize=chunksize, dtype={"customer_id": str}):
        chunk = chunk[chunk["customer_id"].str.startswith(prefix, na=False)]
        matched = chunk[chunk["customer_id"].isin(wanted)]
        if not matched.empty:
            parts.append(matched)
            collected += len(matched)
        if max_rows is not None and collected >= max_rows:
            break
    if not parts:
        return pd.DataFrame(columns=["customer_id", "timestamp_local", "timestamp_epoch", "energy_kwh", "value_status"])
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.rename(columns={"energy_consumption_kwh": "energy_kwh"})
    frame["value_status"] = "observed"
    frame["timestamp_epoch"] = pd.to_numeric(frame["timestamp_epoch"], errors="coerce")
    frame = frame.dropna(subset=["timestamp_epoch"])
    frame["timestamp_epoch"] = frame["timestamp_epoch"].astype("int64")
    frame = frame.drop_duplicates(subset=["customer_id", "timestamp_epoch"], keep="first")
    return frame


def load_classifier(weights: Path) -> WindowClassifier:
    backbone = build_encoder(TASK1_ENCODER_CONFIG, backend="official_patchtst")
    model = WindowClassifier(backbone, d_model=TASK1_ENCODER_CONFIG.d_model, dropout=TASK1_ENCODER_CONFIG.dropout)
    if not weights.exists():
        raise FileNotFoundError(f"model weights not found at {weights}")
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model


def score_customers(model: WindowClassifier, frame: pd.DataFrame, ids: Sequence[str],
                    win_cfg: WindowConfig, threshold: float) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with torch.no_grad():
        for cid in ids:
            sub = frame[frame["customer_id"] == cid]
            if len(sub) <= win_cfg.context_length:
                results.append({"customer_id": cid, "status": "insufficient_data", "n_rows": int(len(sub))})
                continue
            batch = build_windows(sub, win_cfg)
            if len(batch.customer_ids) == 0:
                results.append({"customer_id": cid, "status": "insufficient_data", "n_rows": int(len(sub))})
                continue
            logits = model(torch.tensor(batch.values, dtype=torch.float32),
                           torch.tensor(batch.observed, dtype=torch.bool),
                           torch.tensor(batch.calendar, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()
            score = aggregate_customer_score(probs, batch.center_dates)
            times = pd.to_datetime(sub["timestamp_local"], errors="coerce")
            midday = float(sub.loc[times.dt.hour.between(11, 14), "energy_kwh"].mean())
            night = float(sub.loc[times.dt.hour.between(1, 4), "energy_kwh"].mean())
            results.append({
                "customer_id": cid, "status": "scored",
                "decision": "PV INSTALLED" if score >= threshold else "NO PV",
                "score": float(score), "min_prob": float(np.min(probs)), "max_prob": float(np.max(probs)),
                "n_windows": int(len(probs)),
                "first_center_date": batch.center_dates[0].isoformat(),
                "last_center_date": batch.center_dates[-1].isoformat(),
                "midday_mean_kwh": midday, "night_mean_kwh": night,
                "mid_night_ratio": float(midday / max(1e-4, night)) if np.isfinite(night) else float("nan"),
            })
    return results


def aggregate(results: Sequence[dict[str, object]], threshold: float) -> dict[str, object]:
    scored = [r for r in results if r.get("status") == "scored"]
    pv = [float(r["score"]) for r in scored if r["decision"] == "PV INSTALLED"]
    no_pv = [float(r["score"]) for r in scored if r["decision"] == "NO PV"]

    def span(values: list[float]) -> list[float] | None:
        return [min(values), max(values)] if values else None

    return {
        "threshold": threshold, "requested": len(results), "scored": len(scored),
        "insufficient_data": len(results) - len(scored),
        "pv_installed": len(pv), "no_pv": len(no_pv),
        "score_range_pv_installed": span(pv), "score_range_no_pv": span(no_pv),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unlabelled-csv", required=True, help="semicolon CSV with the challenge columns")
    parser.add_argument("--weights", required=True, help="state dict saved by task1_detector")
    parser.add_argument("--out", required=True, help="private JSON with the per-customer table")
    parser.add_argument("--customer-ids", help="comma-separated ids")
    parser.add_argument("--id-prefix", default="POD")
    parser.add_argument("--id-start", type=int, default=0)
    parser.add_argument("--id-count", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--stride-days", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=150)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int, help="stop scanning after this many matching rows (may truncate histories)")
    parser.add_argument("--show-customers", action="store_true", help="also print the per-customer table to the terminal")
    args = parser.parse_args(argv)

    ids = target_ids(args)
    t0 = time.time()
    frame = read_customers(Path(args.unlabelled_csv), ids, args.id_prefix, args.chunksize, args.max_rows)
    print(f"collected {len(frame)} readings for {frame['customer_id'].nunique()} of {len(ids)} requested customers in {time.time() - t0:.1f}s")

    model = load_classifier(Path(args.weights))
    win_cfg = WindowConfig(context_length=TASK1_ENCODER_CONFIG.context_length, stride=96 * args.stride_days,
                           min_observed_fraction=0.75, max_windows_per_customer=args.max_windows)
    results = score_customers(model, frame, ids, win_cfg, args.threshold)
    summary = aggregate(results, args.threshold)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "customers": results}, indent=2))

    print(f"scored={summary['scored']} insufficient={summary['insufficient_data']} "
          f"pv_installed={summary['pv_installed']} no_pv={summary['no_pv']} threshold={args.threshold}")
    print(f"score_range_pv_installed={summary['score_range_pv_installed']} score_range_no_pv={summary['score_range_no_pv']}")
    if args.show_customers:
        for r in results:
            if r.get("status") != "scored":
                print(f"{r['customer_id']:<12} insufficient data ({r['n_rows']} rows)")
                continue
            print(f"{r['customer_id']:<12} {r['decision']:<13} score={r['score']:.4f} "
                  f"min={r['min_prob']:.4f} max={r['max_prob']:.4f} windows={r['n_windows']}")
    print(f"wrote per-customer table to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
