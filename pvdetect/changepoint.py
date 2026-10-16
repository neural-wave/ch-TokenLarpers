"""Season-aware commissioning detection (`ruptures` candidates, recurrence test, SSE refinement) and date regimes."""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import ruptures as rpt

from pvdetect.season import Climatology

Regime = Literal["event", "present_at_start", "insufficient"]

MIN_VALID_DAYS = 56
WINDOW = 60
FOUND_EVIDENCE = 0.10
FOUND_SHARE = 0.5  # a one-time event keeps at least half of its step after removing the recurring part
BACKTRACK_DAYS = 150
MAX_CANDIDATES = 8  # post-install seasonal PV swings can outrank a weak winter onset; evidence ranks them
MIN_PERSIST = 0.5


@dataclass(frozen=True)
class Commissioning:
    """`date` is the detected date when `found`, else the first valid date; stats describe the chosen
    (or weak) candidate and are NaN when no candidate exists."""

    date: dt.date
    index: int
    step: float
    persist: float
    recur: float
    evidence: float
    gain: float
    yoy_step: float
    found: bool
    weak_date: dt.date | None
    tau: int | None  # contiguous-day index of the candidate used for the stats


def refine_breakpoint(
    y: np.ndarray, tau: int, half_window: int = 45, search: int = 10, min_seg: int = 14
) -> tuple[int, float]:
    """Two-segment SSE minimum of y[a:b] around tau; returns (t, relative SSE gain)."""
    n = y.size
    a, b = max(0, tau - half_window), min(n, tau + half_window)
    seg = y[a:b]
    sse_one = float(np.sum((seg - seg.mean()) ** 2)) if seg.size else 0.0
    lo, hi = max(a + min_seg, tau - search), min(b - min_seg, tau + search)
    if lo > hi:
        return tau, 0.0
    best_t, best_sse = tau, np.inf
    for t in range(lo, hi + 1):
        left, right = y[a:t], y[t:b]
        sse = float(np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))
        if sse < best_sse:
            best_t, best_sse = t, sse
    gain = (sse_one - best_sse) / sse_one if sse_one > 0 else 0.0
    return best_t, gain


def _med(x: np.ndarray, min_n: int) -> float:
    ok = np.isfinite(x)
    return float(np.median(x[ok])) if ok.sum() >= min_n else np.nan


def _window(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return a[max(0, lo) : max(0, min(a.size, hi))]


def _step(ds: np.ndarray, tau: int, min_n: int, inside: bool = False) -> float:
    """median(ds[tau-60, tau)) - median(ds[tau, tau+60)); `inside` requires both windows within the data."""
    if inside and (tau - WINDOW < 0 or tau + WINDOW > ds.size):
        return np.nan
    pre = _med(_window(ds, tau - WINDOW, tau), min_n)
    post = _med(_window(ds, tau, tau + WINDOW), min_n)
    return pre - post


def _persist(ds: np.ndarray, months: np.ndarray, tau: int) -> float:
    pre = _med(_window(ds, tau - 180, tau), 10)
    if not np.isfinite(pre):
        return np.nan
    post_ds, post_m = ds[tau:], months[tau:]
    hits, total = 0, 0
    for m in np.unique(post_m):
        vals = post_ds[post_m == m]
        med = _med(vals, 10)
        if np.isfinite(med):
            total += 1
            hits += med < pre - 0.05
    return hits / total if total else np.nan


def _candidates(ds: np.ndarray, first: int, last: int) -> list[int]:
    """Binseg breakpoints on 7-day block medians of the deseasonalized signal (contiguous-day indices)."""
    starts = np.arange(first, last + 1, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        blocks = np.array(
            [np.nanmedian(ds[s : s + 7]) if np.isfinite(ds[s : s + 7]).sum() >= 4 else np.nan for s in starts]
        )
    x = pd.Series(blocks).ffill().bfill().to_numpy()
    n_bkps = min(MAX_CANDIDATES, x.size // 6)
    if n_bkps < 1 or not np.isfinite(x).all():
        return []
    bkps = rpt.Binseg(model="l2", min_size=3, jump=1).fit(x.reshape(-1, 1)).predict(n_bkps=n_bkps)[:-1]
    return [int(starts[b]) for b in bkps]


def detect_commissioning(daily: pd.DataFrame, clim: Climatology) -> Commissioning | None:
    valid = daily["valid"].to_numpy(bool)
    if valid.sum() < MIN_VALID_DAYS:
        return None
    woy = daily["woy"].to_numpy()
    s_full = np.where(valid, daily["s"].to_numpy(float), np.nan)
    ds = s_full - clim.s[woy - 1]
    months = (daily.index.year * 12 + daily.index.month).to_numpy()
    dates = daily.index
    vidx = np.flatnonzero(valid)
    first, last = int(vidx[0]), int(vidx[-1])

    stats = []
    for tau in _candidates(ds, first, last):
        step = _step(ds, tau, 10)
        if not np.isfinite(step):
            continue
        persist = _persist(ds, months, tau)
        recurs = [_step(ds, tau + 364 * k, 20, inside=True) for k in (-3, -2, -1, 1, 2, 3)]
        recurs = [r for r in recurs if np.isfinite(r)]
        recur = max(recurs) if recurs else np.nan
        evidence = step - max(0.0, recur) if np.isfinite(recur) else step
        stats.append((tau, step, persist, recur, evidence))

    persistent = [c for c in stats if np.isfinite(c[2]) and c[2] >= MIN_PERSIST]
    best = max(persistent, key=lambda c: c[4]) if persistent else None
    found = best is not None and best[4] >= max(FOUND_EVIDENCE, FOUND_SHARE * best[1])
    chosen = best
    if chosen is None:
        positive = [c for c in stats if c[1] > 0]
        chosen = max(positive, key=lambda c: c[1]) if positive else None
    first_date = dates[first].date()
    if chosen is None:
        nan = float("nan")
        return Commissioning(first_date, first, nan, nan, nan, nan, 0.0, nan, False, None, None)

    tau, step, persist, recur, evidence = chosen
    ds_v = ds[vidx]

    def refine(at: int) -> tuple[int, float]:
        t, g = refine_breakpoint(ds_v, int(np.searchsorted(vidx, at)))
        return int(vidx[min(t, vidx.size - 1)]), g

    ref_idx, gain = refine(tau)
    date_idx = ref_idx
    if found:
        # A weak winter onset is often outranked by the larger spring step of the same install:
        # date the event at the earliest qualifying candidate shortly before the strongest one.
        onsets = [
            c
            for c in stats
            if tau - BACKTRACK_DAYS <= c[0] < tau
            and np.isfinite(c[2])
            and c[2] >= MIN_PERSIST
            and c[4] >= max(FOUND_EVIDENCE, FOUND_SHARE * c[1])
        ]
        if onsets:
            date_idx, _ = refine(min(onsets, key=lambda c: c[0])[0])
    ref_date = dates[date_idx].date()

    yoy = np.nan
    d = np.arange(ref_idx, min(ref_idx + 120, ds.size))
    d = d[d - 364 >= 0]
    if d.size:
        prev = d - 364
        ok = np.isfinite(s_full[d]) & np.isfinite(s_full[prev]) & (prev < ref_idx)
        if ok.sum() >= 20:
            yoy = float(np.median(s_full[d[ok]] - s_full[prev[ok]]))

    return Commissioning(
        date=ref_date if found else first_date,
        index=date_idx if found else first,
        step=float(step),
        persist=float(persist),
        recur=float(recur),
        evidence=float(evidence),
        gain=float(gain),
        yoy_step=yoy,
        found=found,
        weak_date=None if found else ref_date,
        tau=ref_idx,
    )


def assign_regime(cp: Commissioning | None, early_contrast: float, contrast_cut: float) -> Regime:
    if cp is not None and cp.found:
        return "event"
    if np.isfinite(early_contrast) and early_contrast <= contrast_cut:
        return "present_at_start"
    return "insufficient"


def shrunk_median(values: list[float], lo: float, hi: float) -> int:
    """median × n/(n+5), clipped and rounded; 0 for an empty list."""
    if not values:
        return 0
    n = len(values)
    return int(round(float(np.clip(np.median(values) * n / (n + 5), lo, hi))))
