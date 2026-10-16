"""Commissioning-date detection with `ruptures` candidates and SSE refinement."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ruptures as rpt


@dataclass(frozen=True)
class Commissioning:
    date: dt.date
    index: int
    drop: float
    gain: float
    found: bool


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


def detect_commissioning(daily: pd.DataFrame) -> Commissioning | None:
    """P1 baseline: Binseg on smoothed s, keep the largest persistent drop, refine on raw s."""
    v = daily[daily["valid"]]
    if len(v) < 56:
        return None
    s = v["s"].to_numpy(float)
    dates = v.index
    x = pd.Series(s).rolling(7, center=True, min_periods=4).median().ffill().bfill().to_numpy()
    n = s.size
    bkps = rpt.Binseg(model="l2", min_size=14, jump=1).fit(x.reshape(-1, 1)).predict(n_bkps=3)[:-1]
    best_tau, best_drop = None, -np.inf
    for tau in bkps:
        drop = float(np.nanmedian(s[max(0, tau - 60) : tau]) - np.nanmedian(s[tau : min(n, tau + 60)]))
        if drop > best_drop:
            best_tau, best_drop = tau, drop
    if best_tau is None or best_drop <= 0:
        return Commissioning(date=dates[0].date(), index=0, drop=0.0, gain=0.0, found=False)
    t, gain = refine_breakpoint(s, best_tau)
    return Commissioning(date=dates[t].date(), index=t, drop=best_drop, gain=gain, found=True)
