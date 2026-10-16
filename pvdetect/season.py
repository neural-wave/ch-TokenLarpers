"""Week-of-year climatology fitted on labelled training customers (always inside `fit`)."""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pvdetect.daily import EPS, WINTER

N_WEEKS = 52
LOG_COLUMNS = ("mid", "night", "eve", "total")
SIGNALS = ("s",) + tuple(f"log_{c}" for c in LOG_COLUMNS)


@dataclass(frozen=True)
class Climatology:
    """`curves[name][woy-1]` is the typical customer-centred value of signal `name` (s, log_mid, ...)."""

    curves: dict[str, np.ndarray]
    yield_shape: np.ndarray
    source: str = "labelled_nonpv"

    @property
    def s(self) -> np.ndarray:
        return self.curves["s"]

    def deseason(self, daily: pd.DataFrame, name: str) -> np.ndarray:
        return signal(daily, name) - self.curves[name][daily["woy"].to_numpy() - 1]

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "curves": {k: v.tolist() for k, v in self.curves.items()},
            "yield_shape": self.yield_shape.tolist(),
        }

    @classmethod
    def from_json(cls, d: dict) -> Climatology:
        return cls({k: np.asarray(v) for k, v in d["curves"].items()}, np.asarray(d["yield_shape"]), d["source"])


def signal(daily: pd.DataFrame, name: str) -> np.ndarray:
    if name == "s":
        return daily["s"].to_numpy(float)
    col = name.removeprefix("log_")
    return np.log(np.maximum(daily[col].to_numpy(float), 0) + EPS)


def log_mid(daily: pd.DataFrame) -> np.ndarray:
    return signal(daily, "log_mid")


def weekly_median(values: np.ndarray, woy: np.ndarray) -> np.ndarray:
    out = np.full(N_WEEKS, np.nan)
    ok = np.isfinite(values)
    if not ok.any():
        return out
    s = pd.Series(values[ok]).groupby(woy[ok]).median()
    out[s.index.to_numpy() - 1] = s.to_numpy()
    return out


def circular_smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    """NaN-aware circular rolling median of width k."""
    h = k // 2
    ext = np.r_[a[-h:], a, a[:h]]
    return np.array(
        [np.nanmedian(ext[i : i + k]) if np.isfinite(ext[i : i + k]).any() else np.nan for i in range(a.size)]
    )


def _combine(curves: list[np.ndarray], smooth: bool) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = np.nanmedian(np.vstack(curves), axis=0)
    if smooth:
        c = circular_smooth(c)
    return np.where(np.isfinite(c), c, 0.0)


def default_yield_shape() -> np.ndarray:
    w = np.arange(1, N_WEEKS + 1)
    y = 0.55 + 0.25 * np.cos(2 * np.pi * (w * 7 - 172) / 365)
    return y / y.max()


def zero_climatology() -> Climatology:
    return Climatology({k: np.zeros(N_WEEKS) for k in SIGNALS}, default_yield_shape(), "zeros")


def _yield_shape(dailies, pv_dates, log_mid_curve: np.ndarray, smooth: bool) -> np.ndarray:
    curves = []
    for cid, date in pv_dates.items():
        d = dailies.get(cid)
        if d is None:
            continue
        v = d[d["valid"]]
        ds_mid = log_mid(v) - log_mid_curve[v["woy"].to_numpy() - 1]
        cut = pd.Timestamp(date)
        post = v.index >= cut + pd.Timedelta(days=14)
        if post.sum() < 60:
            continue
        pre = v.index < cut
        if pre.sum() >= 14:
            pre_level = float(np.median(ds_mid[pre]))
        else:
            winter_post = post & v["month"].isin(WINTER).to_numpy()
            if winter_post.sum() < 14:
                continue
            pre_level = float(np.quantile(ds_mid[winter_post], 0.9))
        curve = weekly_median(pre_level - ds_mid[post], v["woy"].to_numpy()[post])
        mx = np.nanmax(curve) if np.isfinite(curve).any() else np.nan
        if np.isfinite(mx) and mx > 0:
            curves.append(curve / mx)
    if len(curves) < 5:
        return default_yield_shape()
    ys = np.clip(_combine(curves, smooth), 0.0, 1.0)
    return ys / ys.max() if ys.max() > 0 else default_yield_shape()


def fit_climatology(
    dailies: dict[str, pd.DataFrame],
    nonpv_ids: list[str],
    pv_dates: dict[str, dt.date],
    smooth: bool = True,
) -> Climatology:
    """Signal climatologies from labelled non-PV customers; yield shape from labelled PV customers."""
    per_signal: dict[str, list[np.ndarray]] = {k: [] for k in SIGNALS}
    for cid in nonpv_ids:
        d = dailies.get(cid)
        if d is None:
            continue
        v = d[d["valid"]]
        if len(v) < 56:
            continue
        woy = v["woy"].to_numpy()
        for name in SIGNALS:
            x = signal(v, name)
            per_signal[name].append(weekly_median(x - np.median(x), woy))
    if len(per_signal["s"]) < 3:
        curves = {k: np.zeros(N_WEEKS) for k in SIGNALS}
    else:
        curves = {k: _combine(c, smooth) for k, c in per_signal.items()}
    return Climatology(curves, _yield_shape(dailies, pv_dates, curves["log_mid"], smooth))
