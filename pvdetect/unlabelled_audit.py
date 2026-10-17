"""Aggregate-only audit of the unlabelled file: streaming counts, dev-vs-unlabelled KS, peak RSS (Step 17).

Prints no customer ids and no per-customer values.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np
from scipy.stats import ks_2samp

from pvdetect.features import DAILY_LABEL_FREE, daily_label_free_features
from pvdetect.io import load_timeseries
from pvdetect.pipeline import build_feature_table
from pvdetect.unlabelled import accumulate_file


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def _feature_values(dailies) -> dict[str, np.ndarray]:
    rows = [daily_label_free_features(d) for d in dailies]
    return {f: np.array([r[f] for r in rows], float) for f in DAILY_LABEL_FREE}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unlabelled", required=True)
    ap.add_argument("--timeseries", required=True, help="dev timeseries (its customers are excluded)")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--id-prefix", default="POD")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()

    dev = build_feature_table(load_timeseries(args.timeseries))
    dev_vals = _feature_values(dev.dailies().values())
    dailies, rep = accumulate_file(
        args.unlabelled, chunksize=args.chunksize, id_prefix=args.id_prefix, exclude_ids=set(dev.ids)
    )
    unl_vals = _feature_values(dailies.values())
    print(
        f"unlabelled: customers={rep['customers']} rows_total={rep['rows_total']} "
        f"rows_duplicate={rep['rows_duplicate']} rows_excluded_null={rep['rows_excluded_null']} "
        f"rows_excluded_prefix={rep['rows_excluded_prefix']} rows_excluded_labelled={rep['rows_excluded_labelled']}"
    )
    for f in DAILY_LABEL_FREE:
        a, b = dev_vals[f], unl_vals[f]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        stat = ks_2samp(a, b).statistic if a.size and b.size else float("nan")
        print(f"ks: {f} stat={stat:.3f} n_dev={a.size} n_unl={b.size}")
    print(f"peak_rss_mb={peak_rss_mb():.1f} elapsed_s={time.perf_counter() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
