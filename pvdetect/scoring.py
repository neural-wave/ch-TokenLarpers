"""Official challenge scoring (PDF §4.2 main score, §4.3 extension bonus)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pvdetect.io import load_extension_labels, load_labels, parse_bool


def _norm_main(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"customer_id": df["customer_id"].astype(str)})
    pv = df["pv_installed"]
    out["pv_installed"] = pv.astype(bool) if pv.dtype == bool else parse_bool(pv.fillna("FALSE"))
    out["pv_installation_date"] = pd.to_datetime(df["pv_installation_date"], format="mixed", errors="coerce")
    out["pv_capacity_kwp"] = pd.to_numeric(df["pv_capacity_kwp"], errors="coerce")
    return out.set_index("customer_id")


def main_score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """F1 × (70 + 15·D + 15·P); D and P averaged over true positives (missing values score 0)."""
    t = _norm_main(truth)
    p = _norm_main(pred).reindex(t.index)
    p["pv_installed"] = p["pv_installed"].fillna(False).astype(bool)
    tp_mask = t["pv_installed"] & p["pv_installed"]
    tp = int(tp_mask.sum())
    fp = int((~t["pv_installed"] & p["pv_installed"]).sum())
    fn = int((t["pv_installed"] & ~p["pv_installed"]).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    if tp == 0:
        return {"f1": f1, "D": 0.0, "P": 0.0, "score": 0.0, "tp": tp, "fp": fp, "fn": fn}
    days = (p.loc[tp_mask, "pv_installation_date"] - t.loc[tp_mask, "pv_installation_date"]).dt.days.to_numpy(float)
    d_i = np.where(np.isfinite(days), np.maximum(0.0, 1 - np.abs(days) / 30), 0.0)
    pc = p.loc[tp_mask, "pv_capacity_kwp"].to_numpy(float)
    tc = t.loc[tp_mask, "pv_capacity_kwp"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(pc - tc) / tc
    ok = np.isfinite(ape) & (pc > 0) & (tc > 0)
    p_i = np.where(ok, np.maximum(0.0, 1 - ape / 0.5), 0.0)
    D, P = float(d_i.mean()), float(p_i.mean())
    return {"f1": f1, "D": D, "P": P, "score": f1 * (70 + 15 * D + 15 * P), "tp": tp, "fp": fp, "fn": fn}


def score_components(pred: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    """Per truth customer (truth order), the parts of `main_score` (n × 5): TP, FP, FN indicators and the
    date and capacity rewards D_i, P_i (0 unless TP). `weighted_scores` turns them into scores."""
    t = _norm_main(truth)
    p = _norm_main(pred).reindex(t.index)
    t_pv = t["pv_installed"].to_numpy(bool)
    p_pv = p["pv_installed"].fillna(False).astype(bool).to_numpy()
    tp = t_pv & p_pv
    days = (p["pv_installation_date"] - t["pv_installation_date"]).dt.days.to_numpy(float)
    d_i = np.where(np.isfinite(days), np.maximum(0.0, 1 - np.abs(days) / 30), 0.0)
    pc, tc = p["pv_capacity_kwp"].to_numpy(float), t["pv_capacity_kwp"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(pc - tc) / tc
        ok = np.isfinite(ape) & (pc > 0) & (tc > 0)
    p_i = np.where(ok, np.maximum(0.0, 1 - ape / 0.5), 0.0)
    return np.column_stack([tp, ~t_pv & p_pv, t_pv & ~p_pv, np.where(tp, d_i, 0.0), np.where(tp, p_i, 0.0)]).astype(float)


def weighted_scores(W: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """Main score of resampled customer sets (W: draws × n multiplicities); F1·(70+15D+15P) in closed form."""
    tp, fp, fn, sd, sp = (W @ comp).T
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tp > 0, 2 * (70 * tp + 15 * sd + 15 * sp) / denom, 0.0)


def bootstrap_weights(n: int, B: int, seed: int) -> np.ndarray:
    """B customer resamples of size n as multiplicities (draws × n), `default_rng(seed)`."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)


def extension_bonus(pred_ext: pd.DataFrame, truth_ext: pd.DataFrame) -> dict:
    """F1_ext × (7 + 1.5·D_ext + 1.5·P_ext) with one-to-one ±30-day matching per customer."""
    pe = pred_ext.assign(
        customer_id=pred_ext["customer_id"].astype(str),
        extension_date=pd.to_datetime(pred_ext["extension_date"], format="mixed", errors="coerce"),
        pv_capacity_after_extension_kwp=pd.to_numeric(pred_ext["pv_capacity_after_extension_kwp"], errors="coerce"),
    )
    te = truth_ext.assign(
        customer_id=truth_ext["customer_id"].astype(str),
        extension_date=pd.to_datetime(truth_ext["extension_date"], format="mixed", errors="coerce"),
        pv_capacity_after_extension_kwp=pd.to_numeric(truth_ext["pv_capacity_after_extension_kwp"], errors="coerce"),
    )
    tp = 0
    d_scores: list[float] = []
    p_scores: list[float] = []
    for cid in set(pe["customer_id"]) | set(te["customer_id"]):
        pr = pe[pe["customer_id"] == cid]
        tr = te[te["customer_id"] == cid]
        if pr.empty or tr.empty:
            continue
        pd_days = pr["extension_date"].to_numpy("datetime64[D]")
        td_days = tr["extension_date"].to_numpy("datetime64[D]")
        err = np.abs((pd_days[:, None] - td_days[None, :]).astype("timedelta64[D]").astype(float))
        valid = ~(np.isnat(pd_days)[:, None] | np.isnat(td_days)[None, :])
        cost = np.where(valid & (err <= 30), err, 1e6)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] >= 1e6:
                continue
            tp += 1
            d_scores.append(max(0.0, 1 - err[r, c] / 30))
            pc = pr["pv_capacity_after_extension_kwp"].iloc[r]
            tc = tr["pv_capacity_after_extension_kwp"].iloc[c]
            ok = np.isfinite(pc) and np.isfinite(tc) and pc > 0 and tc > 0
            p_scores.append(max(0.0, 1 - abs(pc - tc) / tc / 0.5) if ok else 0.0)
    fp = len(pe) - tp
    fn = len(te) - tp
    if tp == 0:
        return {"f1_ext": 0.0, "D_ext": 0.0, "P_ext": 0.0, "bonus": 0.0, "tp": 0, "fp": fp, "fn": fn}
    f1 = 2 * tp / (2 * tp + fp + fn)
    D, P = float(np.mean(d_scores)), float(np.mean(p_scores))
    return {"f1_ext": f1, "D_ext": D, "P_ext": P, "bonus": f1 * (7 + 1.5 * D + 1.5 * P), "tp": tp, "fp": fp, "fn": fn}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score predictions against truth with the official formulas.")
    ap.add_argument("pred")
    ap.add_argument("truth")
    ap.add_argument("--ext-pred")
    ap.add_argument("--ext-truth")
    args = ap.parse_args(argv)
    s = main_score(load_labels(args.pred), load_labels(args.truth))
    print(f"f1={s['f1']:.4f} D={s['D']:.4f} P={s['P']:.4f} score={s['score']:.2f}")
    if args.ext_pred and args.ext_truth:
        b = extension_bonus(load_extension_labels(args.ext_pred), load_extension_labels(args.ext_truth))
        print(f"f1_ext={b['f1_ext']:.4f} D_ext={b['D_ext']:.4f} P_ext={b['P_ext']:.4f} bonus={b['bonus']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
