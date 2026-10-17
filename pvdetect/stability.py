"""Label-free stability of two models on the unlabelled customers (protocol v2, report stage).

    python -m pvdetect.stability MODEL_A MODEL_B UNLABELLED [--pool 300] [--chunksize N] [--id-prefix POD]

Builds the same bounded pool as pseudo-labelling (customers with ≥365 valid days, chosen by a salted hash
of the id; `pvdetect.unlabelled.build_pool`), predicts it with both models and prints aggregates only:
the predicted PV share per model, how many customers' decisions flip between them, and the mean absolute
change of the PV probability. No labels are read; this is a robustness check, not an accuracy estimate.
"""

from __future__ import annotations

import argparse

import numpy as np

from pvdetect.model import load_model
from pvdetect.pipeline import FeatureTable, predict
from pvdetect.unlabelled import POOL_MIN_VALID_DAYS, accumulate_file, build_pool


def stability(model_a: dict, model_b: dict, pool: dict) -> dict:
    """Aggregates of both models' predictions on the same pool customers."""
    ids = sorted(pool)
    table = FeatureTable(ids=ids, customers=pool)
    pa, _ = predict(model_a, table, ids)
    pb, _ = predict(model_b, table, ids)
    a, b = pa["pv_installed"].to_numpy(bool), pb["pv_installed"].to_numpy(bool)
    n = len(ids)
    return {
        "customers": n,
        "pv_share_a": float(a.mean()) if n else float("nan"),
        "pv_share_b": float(b.mean()) if n else float("nan"),
        "flips": int((a != b).sum()),
        "flip_share": float((a != b).mean()) if n else float("nan"),
        "off_to_on": int((~a & b).sum()),
        "on_to_off": int((a & ~b).sum()),
        "mean_abs_dprob": float(np.abs(pb["prob"].to_numpy(float) - pa["prob"].to_numpy(float)).mean()) if n else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_a")
    ap.add_argument("model_b")
    ap.add_argument("unlabelled", help="unlabelled timeseries CSV")
    ap.add_argument("--pool", type=int, default=300, help="maximum pool customers (as train.py --pseudo-pool)")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--id-prefix", default="POD", help='keep only ids with this prefix ("" keeps all)')
    args = ap.parse_args(argv)
    if args.pool < 1:
        ap.error("--pool must be at least 1")
    model_a, model_b = load_model(args.model_a), load_model(args.model_b)
    dailies, _ = accumulate_file(args.unlabelled, chunksize=args.chunksize, id_prefix=args.id_prefix)
    pool = build_pool(args.unlabelled, dailies, args.pool, chunksize=args.chunksize)
    del dailies
    print(f"stability: pool customers={len(pool)} max={args.pool} min_valid_days={POOL_MIN_VALID_DAYS} (unlabelled, no labels read)")
    if not pool:
        print("stability: empty pool; nothing to compare")
        return 1
    s = stability(model_a, model_b, pool)
    print(f"stability: predicted_pv_share A={s['pv_share_a']:.3f} B={s['pv_share_b']:.3f}")
    print(
        f"stability: flips={s['flips']} share={s['flip_share']:.3f} off->on={s['off_to_on']} on->off={s['on_to_off']} "
        f"mean_abs_dprob={s['mean_abs_dprob']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
