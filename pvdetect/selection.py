"""Score-aware Tier A selection helpers used inside `Procedure.fit` (inner CV only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.05), 2)
KAPPAS = np.round(np.arange(0.80, 1.101, 0.05), 2)


def score_arrays(t_pv: np.ndarray, p_pv: np.ndarray, err_days: np.ndarray, ape: np.ndarray) -> float:
    """Official main score from aligned arrays (NaN error/APE scores 0 for a TP)."""
    tp = t_pv & p_pv
    n_tp = int(tp.sum())
    if n_tp == 0:
        return 0.0
    fp = int((~t_pv & p_pv).sum())
    fn = int((t_pv & ~p_pv).sum())
    f1 = 2 * n_tp / (2 * n_tp + fp + fn)
    e, a = err_days[tp], ape[tp]
    D = np.where(np.isfinite(e), np.maximum(0, 1 - np.abs(e) / 30), 0.0).mean()
    P = np.where(np.isfinite(a), np.maximum(0, 1 - a / 0.5), 0.0).mean()
    return float(f1 * (70 + 15 * D + 15 * P))


def ape_array(pred_kwp: np.ndarray, true_kwp: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(pred_kwp - true_kwp) / true_kwp
    return np.where((pred_kwp > 0) & (true_kwp > 0) & np.isfinite(ape), ape, np.nan)


def pick_threshold(scores: np.ndarray) -> tuple[float, float]:
    """Max score; ties -> smallest |t - 0.5|, then the higher t."""
    best = scores.max()
    tied = [t for t, s in zip(THRESHOLDS, scores) if s >= best - 1e-9]
    t = min(tied, key=lambda x: (abs(x - 0.5), -x))
    return float(t), float(best)


def threshold_scores(prob: np.ndarray, t_pv: np.ndarray, err_days: np.ndarray, ape: np.ndarray) -> np.ndarray:
    return np.array([score_arrays(t_pv, prob >= t, err_days, ape) for t in THRESHOLDS])


def choose_threshold(oof: pd.DataFrame, truth: pd.DataFrame) -> float:
    """oof columns: customer_id, prob, pv_installation_date, pv_capacity_kwp."""
    t = truth.set_index("customer_id").loc[oof["customer_id"]]
    err = (pd.to_datetime(oof["pv_installation_date"]).to_numpy() - t["pv_installation_date"].to_numpy())
    err_days = err.astype("timedelta64[D]").astype(float)
    err_days[np.isnat(err)] = np.nan
    ape = ape_array(oof["pv_capacity_kwp"].to_numpy(float), t["pv_capacity_kwp"].to_numpy(float))
    scores = threshold_scores(oof["prob"].to_numpy(float), t["pv_installed"].to_numpy(bool), err_days, ape)
    return pick_threshold(scores)[0]


def choose_kappa(pred_kwp: np.ndarray, true_kwp: np.ndarray) -> float:
    """κ maximizing mean P_i over the given TPs; ties -> closest to 1.0."""
    if len(pred_kwp) == 0:
        return 1.0
    p_scores = []
    for k in KAPPAS:
        ape = ape_array(k * pred_kwp, true_kwp)
        p_scores.append(np.where(np.isfinite(ape), np.maximum(0, 1 - ape / 0.5), 0.0).mean())
    p_scores = np.array(p_scores)
    best = p_scores.max()
    tied = [k for k, s in zip(KAPPAS, p_scores) if s >= best - 1e-9]
    return float(min(tied, key=lambda k: abs(k - 1.0)))


def bootstrap_prob_better(
    prob_a: np.ndarray,
    t_a: float,
    prob_b: np.ndarray,
    t_b: float,
    t_pv: np.ndarray,
    err_days: np.ndarray,
    ape: np.ndarray,
    B: int = 500,
    seed: int = 0,
) -> float:
    """Inner-only stability heuristic: share of paired customer resamples where B beats A."""
    rng = np.random.default_rng(seed)
    n = t_pv.size
    wins = 0
    for _ in range(B):
        i = rng.integers(0, n, n)
        sa = score_arrays(t_pv[i], prob_a[i] >= t_a, err_days[i], ape[i])
        sb = score_arrays(t_pv[i], prob_b[i] >= t_b, err_days[i], ape[i])
        wins += sb > sa
    return wins / B


def bootstrap_prob_better_outputs(
    t_pv: np.ndarray,
    pred_a: np.ndarray,
    err_a: np.ndarray,
    ape_a: np.ndarray,
    pred_b: np.ndarray,
    err_b: np.ndarray,
    ape_b: np.ndarray,
    B: int = 500,
    seed: int = 0,
) -> float:
    """Inner-only: share of paired customer resamples where submitted outputs B beat A.

    Each configuration brings its own final outputs (prediction, date error, capacity APE)."""
    rng = np.random.default_rng(seed)
    n = t_pv.size
    wins = 0
    for _ in range(B):
        i = rng.integers(0, n, n)
        sa = score_arrays(t_pv[i], pred_a[i], err_a[i], ape_a[i])
        sb = score_arrays(t_pv[i], pred_b[i], err_b[i], ape_b[i])
        wins += sb > sa
    return wins / B
