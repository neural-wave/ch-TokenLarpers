"""Weekly-evidence detection head (Step 2.5-E): clock-aligned weekly windows, a strongly regularized
linear head and per-customer persistence aggregation.

Windows are ISO weeks (Mon–Sun) of the fixed UTC+1 day grid with ≥ 5 valid days; missing readings stay
masked (the daily frame averages observed slots only, and a missing slot breaks a zero run). The
label-free part of a customer's windows may be cached in `table.cache["weekly"]`; `ds_med`/`ds_rel` use
the fit's climatology and are computed per fit. Labels and weights exist only inside a fit."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pvdetect.daily import EPS
from pvdetect.features import NOON_SLOT
from pvdetect.model import fit_logreg, linear_score, sigmoid, standardize_params
from pvdetect.season import Climatology

WEEK_FEATURES = (
    "ds_med",
    "ds_rel",
    "zero_frac",
    "zero_run_h",
    "eve_contrast",
    "night_rel",
    "mid_var",
    "amp",
    "woy_sin",
    "woy_cos",
    "coverage",
)
MIN_VALID_DAYS = 5  # valid days for a week to be kept
BAND_DAYS = 7  # weeks overlapping ± this many days around a labelled date are dropped
HEAD_C = 0.1
WINDOW = 4  # rolling-median width (kept weeks) of the customer score
ZERO_KWH = 0.005  # the slot zero rule of `daily_signal` / `features._zero_window_h`


@dataclass(frozen=True)
class Weeks:
    """One customer's kept weeks in time order. `lf` holds the label-free features (WEEK_FEATURES[2:]);
    `s`/`woy` are (weeks × 7) day values of the deseasonalized-ratio input, NaN on non-valid days."""

    start: np.ndarray  # datetime64[D] Mondays
    lf: np.ndarray
    s: np.ndarray
    woy: np.ndarray


def _empty() -> Weeks:
    n = len(WEEK_FEATURES) - 2
    return Weeks(np.array([], "datetime64[D]"), np.empty((0, n)), np.empty((0, 7)), np.ones((0, 7), np.int64))


def _noon_zero_run_h(M: np.ndarray) -> np.ndarray:
    """Per day: hours of the run of zero slots containing 12:30 UTC+1 (0 when that slot is not zero, NaN
    when it is missing). A missing slot is not a zero, so it ends the run."""
    z = M <= ZERO_KWH  # NaN compares False
    left = np.cumprod(z[:, NOON_SLOT::-1], axis=1).sum(axis=1)
    right = np.cumprod(z[:, NOON_SLOT:], axis=1).sum(axis=1)
    run = np.where(z[:, NOON_SLOT], (left + right - 1) / 4, 0.0)
    return np.where(np.isnan(M[:, NOON_SLOT]), np.nan, run)


def week_windows(cd) -> Weeks:
    """Label-free kept weeks of one customer (`CustomerData`; None or no valid day -> no weeks)."""
    if cd is None or cd.n_valid == 0:
        return _empty()
    daily = cd.daily
    valid = daily["valid"].to_numpy(bool)
    days = daily.index.to_numpy().astype("datetime64[D]")
    dow = daily.index.dayofweek.to_numpy()
    monday = days - dow.astype("timedelta64[D]")
    starts, week = np.unique(monday[valid], return_inverse=True)
    count = np.bincount(week, minlength=starts.size)
    keep = count >= MIN_VALID_DAYS
    if not keep.any():
        return _empty()
    pos = np.flatnonzero(valid)
    new_index = np.cumsum(keep) - 1
    in_kept = keep[week]
    rows, w, d = pos[in_kept], new_index[week[in_kept]], dow[pos[in_kept]]
    n = int(keep.sum())

    def grid(values: np.ndarray, fill=np.nan, dtype=float) -> np.ndarray:
        out = np.full((n, 7), fill, dtype)
        out[w, d] = values[rows]
        return out

    mid = daily["mid"].to_numpy(float)
    night = grid(daily["night"].to_numpy(float))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zero_frac = np.nanmean(grid(daily["zero"].to_numpy(float)), axis=1)
        zero_run = np.nanmedian(grid(_noon_zero_run_h(cd.M)), axis=1)
        eve_contrast = np.nanmedian(grid(daily["s_eve"].to_numpy(float)), axis=1)
        night_week = np.nanmedian(night, axis=1)
        night_ref = np.nanmedian(night)
        night_rel = np.log(np.maximum(night_week, 0) + EPS) - np.log(max(night_ref, 0) + EPS)
        mid_var = np.nanstd(grid(np.log(np.maximum(mid, 0) + EPS)), axis=1)
        amp = np.log1p(np.maximum(np.nanmean(grid(daily["total"].to_numpy(float)), axis=1), 0))
    start = starts[keep]
    woy = np.minimum(pd.DatetimeIndex(start).isocalendar().week.to_numpy().astype(int), 52)
    angle = 2 * np.pi * (woy - 1) / 52
    lf = np.column_stack(
        [zero_frac, zero_run, eve_contrast, night_rel, mid_var, amp, np.sin(angle), np.cos(angle), count[keep] / 7]
    )
    return Weeks(start, lf, grid(daily["s"].to_numpy(float)), grid(daily["woy"].to_numpy(), 1, np.int64))


def cached_weeks(table, cid: str) -> Weeks:
    cache = table.cache.setdefault("weekly", {})
    if cid not in cache:
        cache[cid] = week_windows(table.customers.get(cid))
    return cache[cid]


def week_matrix(weeks: Weeks, clim: Climatology) -> np.ndarray:
    """(weeks × WEEK_FEATURES) under one fit's climatology."""
    if weeks.start.size == 0:
        return np.empty((0, len(WEEK_FEATURES)))
    ds = weeks.s - clim.s[weeks.woy - 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ds_med = np.nanmedian(ds, axis=1)
        ds_rel = ds_med - np.nanmedian(ds)
    return np.column_stack([ds_med, ds_rel, weeks.lf])


def window_labels(start: np.ndarray, first_date, pv: bool, date) -> tuple[np.ndarray, np.ndarray]:
    """(kept mask, label) per week. Non-PV: all 0. PV dated before `first_date` (present at start): all 1.
    Else weeks ending ≥ 7 days before the date are 0, weeks starting ≥ 7 days after it are 1, the weeks
    in between are dropped. PV without a date: every week dropped."""
    n = start.size
    if not pv:
        return np.ones(n, bool), np.zeros(n, bool)
    if date is None or pd.isna(date):
        return np.zeros(n, bool), np.zeros(n, bool)
    d = np.datetime64(pd.Timestamp(date).date(), "D")
    if first_date is not None and d < np.datetime64(pd.Timestamp(first_date).date(), "D"):
        return np.ones(n, bool), np.ones(n, bool)
    band = np.timedelta64(BAND_DAYS, "D")
    neg = start + np.timedelta64(6, "D") <= d - band
    pos = start >= d + band
    return neg | pos, pos


def balanced_weights(customer: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each customer's windows share weight 1; with both classes present, the classes are then balanced
    on those weights (the weighted form of `class_weight="balanced"`: weight × total / (2 × class total))."""
    _, inv, counts = np.unique(customer, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    if not y.any() or y.all():
        return w
    total, pos = w.sum(), w[y].sum()
    return np.where(y, w * total / (2 * pos), w * total / (2 * (total - pos)))


def training_windows(table, lab: pd.DataFrame, ids: list[str], clim: Climatology):
    """(X, y, weight) of the labelled windows of `ids` under `clim`."""
    Xs, ys, owner = [], [], []
    for i, cid in enumerate(ids):
        weeks = cached_weeks(table, cid)
        cd = table.customers.get(cid)
        keep, y = window_labels(
            weeks.start,
            None if cd is None else cd.first_date,
            bool(lab.at[cid, "pv_installed"]),
            lab.at[cid, "pv_installation_date"],
        )
        if keep.any():
            Xs.append(week_matrix(weeks, clim)[keep])
            ys.append(y[keep])
            owner.append(np.full(int(keep.sum()), i))
    if not Xs:
        return np.empty((0, len(WEEK_FEATURES))), np.zeros(0, bool), np.zeros(0)
    y = np.concatenate(ys)
    return np.vstack(Xs), y, balanced_weights(np.concatenate(owner), y)


def fit_head(table, lab: pd.DataFrame, ids: list[str], clim: Climatology) -> dict:
    """The weekly head of one fit (plain JSON). With a single window class the head is constant."""
    X, y, w = training_windows(table, lab, ids, clim)
    if y.any() and not y.all():
        head = fit_logreg(X, y, C=HEAD_C, sample_weight=w, class_weight=None)
    else:
        head = standardize_params(X)
        head.update(coef=[0.0] * X.shape[1], intercept=0.0 if y.size == 0 else (50.0 if y.all() else -50.0))
    head.update(features=list(WEEK_FEATURES), aggregation={"window": WINDOW})
    return head


def aggregate(prob: np.ndarray, window: int = WINDOW) -> float:
    """Customer score from week probabilities in time order: the max of the rolling medians over `window`
    consecutive kept weeks; fewer weeks -> the median of those available; none -> 0.5. NaN-safe."""
    p = np.asarray(prob, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if p.size >= window:
            med = np.nanmedian(np.lib.stride_tricks.sliding_window_view(p, window), axis=1)
        else:
            med = np.array([np.nanmedian(p)]) if p.size else np.array([np.nan])
    return float(np.nanmax(med)) if np.isfinite(med).any() else 0.5


def week_probabilities(head: dict, weeks: Weeks, clim: Climatology) -> np.ndarray:
    X = week_matrix(weeks, clim)
    return sigmoid(linear_score(head, X)) if len(X) else np.empty(0)


def customer_scores(head: dict, table, ids: list[str], clim: Climatology) -> np.ndarray:
    """Per customer of `ids`: the aggregated weekly-head score under `clim`."""
    window = int(head["aggregation"]["window"])
    return np.array([aggregate(week_probabilities(head, cached_weeks(table, c), clim), window) for c in ids], float)
