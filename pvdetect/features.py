"""Per-customer features. Label-free parts may be cached; climatology-dependent parts are computed per fit."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pvdetect.changepoint import Commissioning, detect_commissioning
from pvdetect.daily import EPS, SUMMER, WINTER
from pvdetect.season import Climatology, log_mid

LABEL_FREE = (
    "seasonal_contrast",
    "early_contrast",
    "dip_center_hour",
    "summer_zero_frac",
    "zero_window_h",
    "log_mean_daily_kwh",
    "night_kw",
)
FITTED = (
    "cp_evidence",
    "cp_step",
    "cp_persist",
    "cp_recur",
    "cp_gain",
    "yoy_step",
    "d_mid",
    "d_night",
    "d_eve",
    "dtotal",
    "cloud_var",
    "summer_ds",
    "dip_kw",
    "log_kwp_phys",
)
FEATURE_NAMES = FITTED[:12] + LABEL_FREE + FITTED[12:]
EXPORT_FEATURES = ("neg_frac", "export_kw")
# E1 feature block: computed in every fit/predict, used by the classifier only when inner CV picks it.
PERSISTENCE_FEATURES = ("summer_suppressed_week_frac",)
PERSIST_WEEK_DAYS = 4  # valid summer days for an ISO week to count
PERSIST_MIN_WEEKS = 8
PERSIST_MIN_WINTER_DAYS = 30
PERSIST_DROP = 0.5  # weekly median ds this far below the winter median counts as suppressed
# Schedule feature block: computed in every fit/predict, used by the classifier only when inner CV picks it.
SCHEDULE_FEATURES = ("solar_shape_consistency",)
SCHEDULE_MONTHS = (6, 7, 8)
SCHEDULE_GAP_DAYS = 14  # post window starts this many days after the cut
SCHEDULE_MIN_DAYS = 20  # valid days per stratum (weekday / weekend) on each side of the cut
CAPACITY_FEATURES = ("dip_kw", "log_kwp_phys", "zero_window_h", "night_kw", "log_mean_daily_kwh")
EXPORT_CAPACITY_FEATURES = ("export_kw", "night_kw")
LOG1P_CAPACITY = ("dip_kw", "zero_window_h", "night_kw", "export_kw")
NOON_SLOT = 50  # 12:30 UTC+1


def feature_names(export_enabled: bool, persistence: bool = False) -> list[str]:
    return (
        list(FEATURE_NAMES)
        + (list(EXPORT_FEATURES) if export_enabled else [])
        + (list(PERSISTENCE_FEATURES) if persistence else [])
    )


def capacity_feature_names(export_enabled: bool) -> list[str]:
    return list(CAPACITY_FEATURES) + (["export_kw"] if export_enabled else [])


def last365(v: pd.DataFrame) -> pd.DataFrame:
    if v.empty:
        return v
    return v[v.index > v.index[-1] - pd.Timedelta(days=365)]


def first365(v: pd.DataFrame) -> pd.DataFrame:
    if v.empty:
        return v
    return v[v.index < v.index[0] + pd.Timedelta(days=365)]


def _med(x, min_n: int) -> float:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    return float(np.median(x[ok])) if ok.sum() >= min_n else np.nan


def _contrast(v: pd.DataFrame) -> float:
    su = _med(v.loc[v["month"].isin(SUMMER), "s"], 30)
    wi = _med(v.loc[v["month"].isin(WINTER), "s"], 30)
    return su - wi if np.isfinite(su) and np.isfinite(wi) else np.nan


def _dip_center_hour(dates: np.ndarray, M: np.ndarray, v: pd.DataFrame) -> float:
    w = last365(v)
    pos = np.searchsorted(dates, w.index.to_numpy().astype("datetime64[D]"))
    rows = M[pos]
    hourly = np.nansum(rows.reshape(len(rows), 24, 4), axis=2)
    tot = hourly.sum(axis=1)
    ok = tot > 0
    months = w["month"].to_numpy()
    summer = ok & np.isin(months, SUMMER)
    winter = ok & np.isin(months, WINTER)
    if summer.sum() < 30 or winter.sum() < 30:
        return np.nan
    m = (hourly[summer] / tot[summer, None]).mean(axis=0)
    wn = (hourly[winter] / tot[winter, None]).mean(axis=0)
    hours = np.arange(6, 19)
    diff = np.maximum(0.0, wn[hours] - m[hours])
    if diff.sum() <= 0:
        return np.nan
    return float(np.sum((hours + 0.5) * diff) / diff.sum())


def _zero_window_h(dates: np.ndarray, M: np.ndarray, v: pd.DataFrame) -> float:
    w = last365(v)
    su = w[w["month"].isin(SUMMER)]
    if su.empty:
        return np.nan
    top = su[su["zero"] >= su["zero"].quantile(0.75)]
    pos = np.searchsorted(dates, top.index.to_numpy().astype("datetime64[D]"))
    hours = []
    for row in M[pos]:
        z = row <= 0.005  # NaN compares False and breaks the run
        if not z[NOON_SLOT]:
            hours.append(0.0)
            continue
        lo = NOON_SLOT
        while lo > 0 and z[lo - 1]:
            lo -= 1
        hi = NOON_SLOT
        while hi < 95 and z[hi + 1]:
            hi += 1
        hours.append((hi - lo + 1) / 4)
    return float(np.mean(hours))


DAILY_LABEL_FREE = ("seasonal_contrast", "early_contrast", "summer_zero_frac", "log_mean_daily_kwh", "night_kw")


def daily_label_free_features(daily: pd.DataFrame) -> dict[str, float]:
    """The label-free features computable from the daily frame alone (also used for unlabelled data)."""
    out = dict.fromkeys(DAILY_LABEL_FREE, np.nan)
    v = daily[daily["valid"]]
    if v.empty:
        return out
    out["seasonal_contrast"] = _contrast(last365(v))
    out["early_contrast"] = _contrast(first365(v))
    su = last365(v)
    su = su[su["month"].isin(SUMMER)]
    out["summer_zero_frac"] = float(su["zero"].mean()) if len(su) else np.nan
    out["log_mean_daily_kwh"] = float(np.log1p(max(0.0, v["total"].mean())))
    out["night_kw"] = 4 * float(v["night"].median())
    return out


def label_free_features(cd) -> dict[str, float]:
    """Features that never depend on labels or fitted artifacts (cacheable across folds)."""
    out = dict.fromkeys(LABEL_FREE + EXPORT_FEATURES, np.nan)
    if cd is None or cd.n_valid == 0:
        return out
    v = cd.daily[cd.daily["valid"]]
    out.update(daily_label_free_features(cd.daily))
    out["dip_center_hour"] = _dip_center_hour(cd.dates, cd.M, v)
    out["zero_window_h"] = _zero_window_h(cd.dates, cd.M, v)
    out["neg_frac"] = cd.neg_frac
    out["export_kw"] = 4 * max(0.0, -cd.q001)
    return out


def _delta(v: pd.DataFrame, x: np.ndarray, cut: pd.Timestamp, days: int = 90) -> float:
    """Change of the median of a deseasonalized log signal across `cut` (90-day windows)."""
    idx = v.index
    pre = _med(x[(idx < cut) & (idx >= cut - pd.Timedelta(days=days))], 20)
    post = _med(x[(idx >= cut) & (idx < cut + pd.Timedelta(days=days))], 20)
    return post - pre if np.isfinite(pre) and np.isfinite(post) else np.nan


def _mad_diff(x: np.ndarray) -> float:
    d = np.diff(x)
    return float(np.median(np.abs(d - np.median(d))))


def _cloud_var(v: pd.DataFrame, cut: pd.Timestamp) -> float:
    su = v[v["month"].isin(SUMMER)]
    lm = log_mid(su)
    idx = su.index
    pre = lm[(idx < cut) & (idx >= cut - pd.Timedelta(days=180))]
    post = lm[(idx >= cut) & (idx < cut + pd.Timedelta(days=180))]
    if pre.size < 20 or post.size < 20:
        return np.nan
    return float(np.clip(np.log((_mad_diff(post) + 1e-6) / (_mad_diff(pre) + 1e-6)), -5.0, 5.0))


def _summer_suppressed_week_frac(w: pd.DataFrame, ds: np.ndarray) -> float:
    """E1: share of summer ISO weeks in `w` (≥4 valid days) whose median ds ≤ the winter median ds − 0.5."""
    months = w["month"].to_numpy()
    winter = _med(ds[np.isin(months, WINTER)], PERSIST_MIN_WINTER_DAYS)
    if not np.isfinite(winter):
        return np.nan
    summer = np.isin(months, SUMMER)
    iso = w.index[summer].isocalendar()
    week = iso["year"].to_numpy(np.int64) * 100 + iso["week"].to_numpy(np.int64)
    weekly = pd.Series(ds[summer]).groupby(week).agg(["median", "count"])
    weekly = weekly[weekly["count"] >= PERSIST_WEEK_DAYS]
    if len(weekly) < PERSIST_MIN_WEEKS:
        return np.nan
    return float((weekly["median"] <= winter - PERSIST_DROP).mean())


def _solar_shape_consistency(v: pd.DataFrame, cut: pd.Timestamp) -> float:
    """Change of the June–August midday trough depth across `cut`, the smaller of weekdays and weekends.

    Trough depth per valid day: (mid − mean(night, eve)) / the customer's median June–August total per slot.
    Pre: June–August days before `cut`; post: June–August days ≥ 14 days after it; ≥ 20 days per stratum and
    side, else NaN. Positive only when both day types acquire a deeper midday trough (PV, not a schedule)."""
    jja = v[v["month"].isin(SCHEDULE_MONTHS)]
    norm = _med(jja["total"].to_numpy(float) / 96, 1)
    if not (np.isfinite(norm) and norm > 0):
        return np.nan
    t = (jja["mid"].to_numpy(float) - (jja["night"].to_numpy(float) + jja["eve"].to_numpy(float)) / 2) / norm
    idx = jja.index
    pre = idx < cut
    post = idx >= cut + pd.Timedelta(days=SCHEDULE_GAP_DAYS)
    weekend = jja["weekend"].to_numpy(bool)
    deltas = []
    for stratum in (False, True):
        on = weekend == stratum
        a, b = _med(t[pre & on], SCHEDULE_MIN_DAYS), _med(t[post & on], SCHEDULE_MIN_DAYS)
        if not (np.isfinite(a) and np.isfinite(b)):
            return np.nan
        deltas.append(a - b)
    return float(min(deltas))


def fitted_features(
    cd, clim: Climatology, lf: dict[str, float], *, recur_strict: bool = False
) -> tuple[dict[str, float], Commissioning | None]:
    """Climatology-dependent features (computed inside every fit/predict with that fit's climatology);
    `recur_strict` is the fit's short-series detection option (`detect_commissioning`)."""
    out = dict.fromkeys(FITTED + PERSISTENCE_FEATURES + SCHEDULE_FEATURES, np.nan)
    if cd is None or cd.n_valid == 0:
        return out, None
    daily = cd.daily
    cp = detect_commissioning(daily, clim, recur_strict=recur_strict)
    v = daily[daily["valid"]]
    if cp is not None:
        has = cp.tau is not None
        for name, val in (
            ("cp_evidence", cp.evidence),
            ("cp_step", cp.step),
            ("cp_persist", cp.persist),
            ("cp_recur", cp.recur),
            ("cp_gain", cp.gain),
            ("yoy_step", cp.yoy_step),
        ):
            out[name] = val if has else (0.0 if name != "yoy_step" else np.nan)
        if has:
            cut = daily.index[cp.tau]
            out["d_mid"] = _delta(v, clim.deseason(v, "log_mid"), cut)
            out["d_night"] = _delta(v, clim.deseason(v, "log_night"), cut)
            out["d_eve"] = _delta(v, clim.deseason(v, "log_eve"), cut)
            out["dtotal"] = _delta(v, clim.deseason(v, "log_total"), cut)
            out["cloud_var"] = _cloud_var(v, cut)
            out["solar_shape_consistency"] = _solar_shape_consistency(v, cut)
    ds = v["s"].to_numpy(float) - clim.s[v["woy"].to_numpy() - 1]
    w365 = v.index > v.index[-1] - pd.Timedelta(days=365)
    summer = v["month"].isin(SUMMER).to_numpy()
    sd = _med(ds[w365 & summer], 30)
    out["summer_ds"] = sd if np.isfinite(sd) else _med(ds[summer], 30)
    out["summer_suppressed_week_frac"] = _summer_suppressed_week_frac(v[w365], ds[w365])

    w = last365(v)
    summer_all = v[v["month"].isin(SUMMER)]
    if cp is not None and cp.found:
        cut = pd.Timestamp(cp.date)
        pre = _med(summer_all.loc[summer_all.index < cut, "mid"], 10)
        post = summer_all.loc[summer_all.index >= cut, "mid"]
    else:
        pre, post = np.nan, w.loc[w["month"].isin(SUMMER), "mid"]
    if not np.isfinite(pre):
        pre = _med(w.loc[w["month"].isin(WINTER), "mid"], 1)
    if np.isfinite(pre) and post.notna().sum():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["dip_kw"] = 4 * max(0.0, pre - float(np.nanquantile(post, 0.10)))
    zw = lf.get("zero_window_h", np.nan)
    if np.isfinite(pre) and np.isfinite(zw) and zw > 0:
        zw = min(zw, 11.0)
        out["log_kwp_phys"] = float(np.log(4 * max(pre, 0) + EPS) - 1.5 * np.log(np.cos(np.pi / 4 * zw / 6.5)))
    elif np.isfinite(out["dip_kw"]):
        out["log_kwp_phys"] = float(np.log(out["dip_kw"] + 0.1))
    return out, cp
