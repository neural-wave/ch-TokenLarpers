"""Week-of-year climatology fitted on labelled training customers (always inside `fit`)."""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pvdetect.daily import EPS, WINTER

N_WEEKS = 52
LOG_COLUMNS = ("mid", "night", "eve", "total")
SIGNALS = ("s",) + tuple(f"log_{c}" for c in LOG_COLUMNS)
# Sunshine index (option `sun`): rolling-median windows (window, min days) per day type, contributors per date
# and the days after the labelled installation date before a PV customer contributes. Each date's value is the
# mean after dropping the lowest and highest contributor, so a saved value always mixes ≥3 customers.
SUN_WINDOWS = {False: ("15D", 5), True: ("29D", 4)}  # weekday, weekend
SUN_MIN_CONTRIB = 5
SUN_POST_DAYS = 14


def sun_residual(daily: pd.DataFrame) -> pd.Series:
    """Label-free daily midday anomaly over valid days: `mid` minus its centred rolling median within the same
    day type (weekday/weekend), divided by 1.4826·MAD. Seasonal and schedule changes slower than about two
    weeks cancel; day-to-day weather stays. Empty with fewer than 30 valid days or zero spread."""
    valid = daily["valid"].to_numpy(bool)
    if valid.sum() < 30:
        return pd.Series(dtype=float)
    v = daily.loc[valid, "mid"].astype(float)
    weekend = daily.loc[valid, "weekend"].to_numpy(bool)
    parts = []
    for day_type, (window, min_days) in SUN_WINDOWS.items():
        s = v[weekend == day_type]
        parts.append(s - s.rolling(window, center=True, min_periods=min_days).median())
    r = pd.concat(parts).sort_index().dropna()
    if r.empty:
        return r
    mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    return r / mad if np.isfinite(mad) and mad > 0 else pd.Series(dtype=float)


@dataclass(frozen=True)
class SunIndex:
    """Daily sunshine index of one fit: per date, the trimmed mean (lowest and highest dropped) of the
    `sun_residual` of that fit's labelled PV customers from 14 days after their installation date on (≥5 of
    them, else NaN). PV output lowers the midday load on sunny days, so the index is low on sunny days. `loo[c]`
    is the index without contributor c, used for c's own training feature; it exists only inside the fit and is
    never serialized."""

    start: pd.Timestamp
    values: np.ndarray
    loo: dict = field(default_factory=dict, compare=False, repr=False)

    def series(self, cid: str | None = None) -> pd.Series:
        vals = self.loo.get(cid, self.values)
        return pd.Series(vals, index=pd.date_range(self.start, periods=len(vals), freq="D"))

    def to_json(self) -> dict:
        return {
            "start": self.start.strftime("%Y-%m-%d"),
            "values": [float(x) if np.isfinite(x) else None for x in self.values],
        }

    @classmethod
    def from_json(cls, d: dict) -> SunIndex:
        return cls(pd.Timestamp(d["start"]), np.array([np.nan if x is None else x for x in d["values"]], float))


def _trimmed_rows(a: np.ndarray) -> np.ndarray:
    """Per row: mean of the finite values without the lowest and the highest; NaN below SUN_MIN_CONTRIB."""
    n = np.isfinite(a).sum(axis=1)
    out = np.full(a.shape[0], np.nan)
    ok = n >= SUN_MIN_CONTRIB
    if ok.any():
        b = a[ok]
        out[ok] = (np.nansum(b, axis=1) - np.nanmin(b, axis=1) - np.nanmax(b, axis=1)) / (n[ok] - 2)
    return out


def fit_sun_index(residuals: dict[str, pd.Series], pv_dates: dict[str, dt.date]) -> SunIndex | None:
    """`SunIndex` from the `sun_residual` of the labelled PV customers in `pv_dates`; None when no date has
    `SUN_MIN_CONTRIB` contributors."""
    cols = {}
    for cid, date in pv_dates.items():
        z = residuals.get(cid)
        if z is None or z.empty:
            continue
        z = z[z.index >= pd.Timestamp(date) + pd.Timedelta(days=SUN_POST_DAYS)]
        if len(z):
            cols[cid] = z
    if len(cols) < SUN_MIN_CONTRIB:
        return None
    Z = pd.DataFrame(cols)
    Z = Z.reindex(pd.date_range(Z.index.min(), Z.index.max(), freq="D"))
    A = Z.to_numpy(float)
    values = _trimmed_rows(A)
    if not np.isfinite(values).any():
        return None
    loo = {c: _trimmed_rows(np.delete(A, j, axis=1)) for j, c in enumerate(Z.columns)}
    return SunIndex(Z.index[0], values, loo)


@dataclass(frozen=True)
class Climatology:
    """`curves[name][woy-1]` is the typical customer-centred value of signal `name` (s, log_mid, ...).

    `evidence_scale[woy-1]` (optional) divides change-point step/recur/evidence at that week; `sun` (optional,
    option `sun`) is the fit's daily sunshine index."""

    curves: dict[str, np.ndarray]
    yield_shape: np.ndarray
    source: str = "labelled_nonpv"
    evidence_scale: np.ndarray | None = None
    sun: SunIndex | None = None

    @property
    def s(self) -> np.ndarray:
        return self.curves["s"]

    def deseason(self, daily: pd.DataFrame, name: str) -> np.ndarray:
        return signal(daily, name) - self.curves[name][daily["woy"].to_numpy() - 1]

    def to_json(self) -> dict:
        out = {
            "source": self.source,
            "curves": {k: v.tolist() for k, v in self.curves.items()},
            "yield_shape": self.yield_shape.tolist(),
        }
        if self.evidence_scale is not None:
            out["evidence_scale"] = np.asarray(self.evidence_scale).tolist()
        if self.sun is not None:
            out["sun"] = self.sun.to_json()
        return out

    @classmethod
    def from_json(cls, d: dict) -> Climatology:
        scale = d.get("evidence_scale")
        return cls(
            {k: np.asarray(v) for k, v in d["curves"].items()},
            np.asarray(d["yield_shape"]),
            d["source"],
            None if scale is None else np.asarray(scale, float),
            SunIndex.from_json(d["sun"]) if "sun" in d else None,
        )


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


def signal_curves(dailies: dict[str, pd.DataFrame], ids: list[str], smooth: bool = True) -> dict[str, np.ndarray]:
    """Combined customer-centred weekly curves of every signal over `ids` (≥56 valid days each)."""
    per_signal: dict[str, list[np.ndarray]] = {k: [] for k in SIGNALS}
    for cid in ids:
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
        return {k: np.zeros(N_WEEKS) for k in SIGNALS}
    return {k: _combine(c, smooth) for k, c in per_signal.items()}


def climatology_from_curves(
    curves: dict[str, np.ndarray],
    dailies: dict[str, pd.DataFrame],
    pv_dates: dict[str, dt.date],
    source: str,
    smooth: bool = True,
) -> Climatology:
    """Given signal curves + the yield shape fitted from labelled PV customers."""
    return Climatology(curves, _yield_shape(dailies, pv_dates, curves["log_mid"], smooth), source)


def fit_climatology(
    dailies: dict[str, pd.DataFrame],
    nonpv_ids: list[str],
    pv_dates: dict[str, dt.date],
    smooth: bool = True,
) -> Climatology:
    """Signal climatologies from labelled non-PV customers; yield shape from labelled PV customers."""
    return climatology_from_curves(
        signal_curves(dailies, nonpv_ids, smooth), dailies, pv_dates, "labelled_nonpv", smooth
    )
