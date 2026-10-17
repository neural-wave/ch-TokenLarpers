"""Paired comparison of two out-of-fold files on the same outer splits, for the data holder only.

    python -m pvdetect.compare BASE_OOF CAND_OOF LABELS [--out-dir PRIVATE_DIR]

BASE_OOF and CAND_OOF are shared out-of-fold files from `train.py --cv --oof-out`; every
(repeat, customer_id) must sit in the same fold in both, else the comparison is refused.
Prints aggregate lines only: per repeat and pooled over repeats, the confusion matrix and the score
components of both; paired outcome changes; a paired customer bootstrap of the score difference.
`--out-dir` (outside the repository) receives `changes.csv`, the per-customer table; it is never printed.
The output is AIL-derived when run on real data: keep it with the team, never in an agent session.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pvdetect.diagnose import _fmt, _outside_repo, error_table, summarize
from pvdetect.io import _read_str_table, _to_date, _to_float, load_labels, parse_bool

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.90


def load_oof(path: str | Path) -> pd.DataFrame:
    """All repeats of a shared OOF file: customer_id, repeat, fold and the submitted outputs."""
    df, _ = _read_str_table(path)
    out = pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "repeat": df["repeat"].astype(int),
            "fold": df["fold"].astype(int),
            "pv_installed": parse_bool(df["pv_installed"]),
            "pv_installation_date": _to_date(df["pv_installation_date"]),
            "pv_capacity_kwp": _to_float(df["pv_capacity_kwp"]),
        }
    )
    if out.duplicated(["repeat", "customer_id"]).any():
        raise ValueError(f"{path}: duplicate (repeat, customer_id) rows")
    return out


def check_same_splits(base: pd.DataFrame, cand: pd.DataFrame) -> None:
    """Refuse unless every (repeat, customer_id) has the same fold in both files."""
    a = base.set_index(["repeat", "customer_id"])["fold"]
    b = cand.set_index(["repeat", "customer_id"])["fold"]
    if set(a.index) != set(b.index):
        raise ValueError("the two OOF files cover different (repeat, customer_id) pairs; not the same outer splits")
    differ = int((a != b.reindex(a.index)).sum())
    if differ:
        raise ValueError(f"{differ} (repeat, customer_id) pairs sit in different folds; not the same outer splits")


def _paired(tb: pd.DataFrame, tc: pd.DataFrame, t_pv: np.ndarray) -> dict:
    right_b = tb["outcome"].isin(["TP", "TN"]).to_numpy()
    right_c = tc["outcome"].isin(["TP", "TN"]).to_numpy()
    out = {}
    for name, grp in (("pv", t_pv), ("nonpv", ~t_pv)):
        out[name] = {
            "wrong_to_right": int((grp & ~right_b & right_c).sum()),
            "right_to_wrong": int((grp & right_b & ~right_c).sum()),
        }
    return out


def _components(tab: pd.DataFrame) -> np.ndarray:
    """Per-customer score parts (n × 5): TP, FP, FN indicators and D_i, P_i (0 unless TP)."""
    o = tab["outcome"].to_numpy()
    tp = o == "TP"
    return np.column_stack(
        [
            tp,
            o == "FP",
            o == "FN",
            np.where(tp, tab["D_i"].to_numpy(float), 0.0),
            np.where(tp, tab["P_i"].to_numpy(float), 0.0),
        ]
    ).astype(float)


def _weighted_scores(W: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """Official score of resampled customer sets (W: draws × n multiplicities); F1·(70+15D+15P) in closed form."""
    tp, fp, fn, sd, sp = (W @ comp).T
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tp > 0, 2 * (70 * tp + 15 * sd + 15 * sp) / denom, 0.0)


def paired_bootstrap(pairs: list[tuple[pd.DataFrame, pd.DataFrame]], B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED):
    """Score difference (candidate − base) averaged over repeats; customers resampled once per draw and
    the same resample applied to every repeat. Returns (point estimate, lo, hi) of the 90% interval."""
    ids = pairs[0][0].index
    n = len(ids)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)
    point, draws = [], np.zeros(B)
    for tb, tc in pairs:
        cb, cc = _components(tb.loc[ids]), _components(tc.loc[ids])
        one = np.ones((1, n))
        point.append(float(_weighted_scores(one, cc)[0] - _weighted_scores(one, cb)[0]))
        draws += _weighted_scores(W, cc) - _weighted_scores(W, cb)
    draws /= len(pairs)
    a = (1 - CI_LEVEL) / 2
    return float(np.mean(point)), float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "improvement"
    if hi < 0:
        return "worse"
    return "within_noise"


def _summary_line(label: str, s: dict) -> str:
    c = s["counts"]
    return (
        f"{label}: TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']} precision={_fmt(s['precision'])} "
        f"recall={_fmt(s['recall'])} f1={_fmt(s['f1'])} D={_fmt(s['D'])} P={_fmt(s['P'])} score={s['score']:.2f}"
    )


def _paired_line(label: str, p: dict) -> str:
    return f"{label}: " + " | ".join(
        f"{g} wrong->right={v['wrong_to_right']} right->wrong={v['right_to_wrong']}" for g, v in p.items()
    )


def compare(base: pd.DataFrame, cand: pd.DataFrame, truth: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """(printed aggregate lines, per-customer changes table)."""
    check_same_splits(base, cand)
    truth = truth[truth["customer_id"].isin(base["customer_id"])].reset_index(drop=True)
    t_pv = truth["pv_installed"].to_numpy(bool)
    lines = [f"compare: repeats={base['repeat'].nunique()} customers={len(truth)} (aggregates only)"]
    pairs, rows = [], []
    for r in sorted(base["repeat"].unique()):
        pb, pc = base[base["repeat"] == r], cand[cand["repeat"] == r]
        tb, tc = error_table(pb, truth), error_table(pc, truth)
        pairs.append((tb, tc))
        lines += [
            _summary_line(f"repeat {r} base", summarize(tb)),
            _summary_line(f"repeat {r} cand", summarize(tc)),
            _paired_line(f"repeat {r} paired", _paired(tb, tc, t_pv)),
        ]
        fold = pb.set_index("customer_id")["fold"].reindex(tb.index)
        rows.append(
            pd.DataFrame(
                {
                    "repeat": r,
                    "fold": fold.to_numpy(),
                    "true_pv": t_pv,
                    "base_pred_pv": tb["outcome"].isin(["TP", "FP"]).to_numpy(),
                    "cand_pred_pv": tc["outcome"].isin(["TP", "FP"]).to_numpy(),
                    "base_outcome": tb["outcome"].to_numpy(),
                    "cand_outcome": tc["outcome"].to_numpy(),
                    "base_date_err_days": tb["date_err_days"].to_numpy(),
                    "cand_date_err_days": tc["date_err_days"].to_numpy(),
                    "base_capacity_ape": tb["capacity_ape"].to_numpy(),
                    "cand_capacity_ape": tc["capacity_ape"].to_numpy(),
                },
                index=tb.index,
            )
        )
    pooled_b = pd.concat([tb for tb, _ in pairs])
    pooled_c = pd.concat([tc for _, tc in pairs])
    totals = {"pv": {}, "nonpv": {}}
    for tb, tc in pairs:
        for g, v in _paired(tb, tc, t_pv).items():
            for k, x in v.items():
                totals[g][k] = totals[g].get(k, 0) + x
    lines += [
        _summary_line("total base (pooled over repeats)", summarize(pooled_b)),
        _summary_line("total cand (pooled over repeats)", summarize(pooled_c)),
        _paired_line("total paired", totals),
    ]
    diff, lo, hi = paired_bootstrap(pairs)
    lines.append(
        f"bootstrap: score_diff={diff:+.2f} ci90=[{lo:+.2f},{hi:+.2f}] draws={BOOTSTRAP_B} "
        f"(customers resampled, same per repeat) verdict={verdict(lo, hi)}"
    )
    return lines, pd.concat(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("labels")
    ap.add_argument("--out-dir", type=Path, help="private folder for the per-customer changes table")
    args = ap.parse_args(argv)
    out_dir = _outside_repo(args.out_dir) if args.out_dir else None
    try:
        lines, changes = compare(load_oof(args.base), load_oof(args.cand), load_labels(args.labels))
    except ValueError as e:
        raise SystemExit(f"refusing: {e}") from None
    for line in lines:
        print(line)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        changes.to_csv(out_dir / "changes.csv", sep=";", index_label="customer_id", float_format="%.4f")
        print("per-customer table (changes.csv) written to the private --out-dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
