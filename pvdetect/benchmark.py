"""Synthetic benchmark suites (evidence tiers R/G) with a mandatory pilot and workload budget.

`--suite dev`: generator variant A, seeds 100+, free during development.
`--suite holdout`: generator variant B, seeds from benchmarks/HOLDOUT_SEED_BLOCK; phase gates only, logged.
Every full run is preceded by a pilot (base scenario, n=12, one repeat, dev seeds 900+) whose timing
projects the full run; runs over budget are refused unless `--force`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pvdetect.evaluation import REGIMES, nested_cv, outer_splits
from pvdetect.io import parse_bool
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.procedure import fit
from pvdetect.synthetic import SCENARIOS, generate_dataset

REPO = Path(__file__).resolve().parent.parent
PILOT_N = 12
PILOT_SEED = 900
DEV_SEED = 100
MEMORY_CAP_MB = 4000


@dataclass
class Timing:
    prep_s: float
    fit_s: float
    peak_rss_mb: float


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def _typed(labels: pd.DataFrame) -> pd.DataFrame:
    return labels.assign(
        pv_installed=parse_bool(labels["pv_installed"]),
        pv_installation_date=pd.to_datetime(labels["pv_installation_date"], errors="coerce"),
        pv_capacity_kwp=pd.to_numeric(labels["pv_capacity_kwp"], errors="coerce"),
    )


def _as_table(ts: pd.DataFrame):
    from pvdetect.io import LoadedTimeseries

    frame = pd.DataFrame(
        {
            "customer_id": ts["customer_id"].astype(str),
            "epoch_s": ts["timestamp_epoch"].astype(np.int64),
            "energy_kwh": ts["energy_consumption_kwh"].astype(float),
        }
    ).drop_duplicates(["customer_id", "epoch_s"])
    frame = frame.sort_values(["customer_id", "epoch_s"]).reset_index(drop=True)
    return build_feature_table(LoadedTimeseries(frame, sorted(frame["customer_id"].unique()), {}))


def run_scenario(scenario: str, n: int, repeats: int, seed: int, variant: str) -> tuple[dict, Timing]:
    t0 = time.perf_counter()
    ts, labels, _ = generate_dataset(n, seed, variant=variant, scenario=scenario)
    table = _as_table(ts)
    labels = _typed(labels)
    t1 = time.perf_counter()
    res = nested_cv(fit, predict, table, labels, outer_splits(labels, repeats=repeats), bootstrap=0)
    t2 = time.perf_counter()
    reg = {}
    for r in REGIMES:
        vals = [x["by_regime"][r]["D"] for x in res.regimes if x["by_regime"][r]["n"]]
        reg[r] = float(np.mean(vals)) if vals else np.nan
    row = {
        "scenario": scenario,
        "n": n,
        "repeats": repeats,
        "score": res.mean("score"),
        "f1": res.mean("f1"),
        "D": res.mean("D"),
        "P": res.mean("P"),
        "regime_D": " ".join(f"{k[:5]}={v:.2f}" for k, v in reg.items() if np.isfinite(v)),
    }
    return row, Timing(t1 - t0, t2 - t1, peak_rss_mb())


def project(pilot: Timing, n: int, repeats: int, n_scenarios: int) -> float:
    """Minutes: fitting ~quadratic in customers (inner CV per fit), preprocessing linear; per repeat."""
    f = n / PILOT_N
    return (pilot.fit_s * f**2 + pilot.prep_s * f) * repeats * n_scenarios / 60


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=("dev", "holdout"))
    ap.add_argument("--pilot", action="store_true", help="run only the pilot and print the projection")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--budget-min", type=float, default=30.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log")
    args = ap.parse_args(argv)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"unknown scenario: {sorted(unknown)[0]}")
    if not args.pilot and args.suite is None:
        ap.error("--suite is required unless --pilot")
    if args.suite == "holdout" and not args.log:
        ap.error("--suite holdout requires --log PATH")

    _, pilot = run_scenario("base", PILOT_N, 1, PILOT_SEED, "A")
    minutes = project(pilot, args.n, args.repeats, len(scenarios))
    print(
        f"pilot: prep_s={pilot.prep_s:.1f} fit_s={pilot.fit_s:.1f} peak_rss_mb={pilot.peak_rss_mb:.0f} "
        f"projected_min={minutes:.1f} for n={args.n} repeats={args.repeats} scenarios={len(scenarios)}"
    )
    if args.pilot:
        return 0
    over_time = minutes > args.budget_min
    over_mem = pilot.peak_rss_mb * 2 >= MEMORY_CAP_MB
    if (over_time or over_mem) and not args.force:
        print(
            f"refused: projected_min={minutes:.1f} (budget {args.budget_min}) peak_rss_mb={pilot.peak_rss_mb:.0f}; "
            "reduce --scenarios, --n or --repeats, or pass --force"
        )
        return 3

    if args.suite == "holdout":
        base_seed = int((REPO / "benchmarks" / "HOLDOUT_SEED_BLOCK").read_text().strip())
        variant = "B"
    else:
        base_seed, variant = DEV_SEED, "A"
    rows = []
    for j, sc in enumerate(scenarios):
        row, _ = run_scenario(sc, args.n, args.repeats, base_seed + j, variant)
        rows.append(row)
        print(
            f"{sc}: score={row['score']:.2f} f1={row['f1']:.3f} D={row['D']:.3f} P={row['P']:.3f} {row['regime_D']}",
            flush=True,
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.log:
        sha = _git_sha()
        today = dt.date.today().isoformat()
        with open(args.log, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(
                    f"|{today}|{sha}|{args.suite}|{r['scenario']}|{r['n']}|{r['repeats']}|{r['score']:.2f}|"
                    f"{r['f1']:.3f}|{r['D']:.3f}|{r['P']:.3f}|{r['regime_D']}|\n"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
