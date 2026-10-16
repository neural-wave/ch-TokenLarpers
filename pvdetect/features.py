"""Per-customer physical features (P1 baseline set)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pvdetect.changepoint import Commissioning, detect_commissioning
from pvdetect.daily import SUMMER, WINTER

FEATURE_NAMES = (
    "summer_dip",
    "seasonal_contrast",
    "summer_zero_frac",
    "cp_drop",
    "cp_gain",
    "log_mean_daily_kwh",
    "dip_kw",
    "night_kw",
)
CAPACITY_FEATURES = ("dip_kw", "night_kw", "log_mean_daily_kwh")
LOG1P_CAPACITY = ("dip_kw", "night_kw")


def last365(v: pd.DataFrame) -> pd.DataFrame:
    if v.empty:
        return v
    return v[v.index > v.index[-1] - pd.Timedelta(days=365)]


def _med(x: pd.Series, min_n: int) -> float:
    return float(np.nanmedian(x)) if x.notna().sum() >= min_n else np.nan


def baseline_features(daily: pd.DataFrame) -> tuple[dict[str, float], Commissioning | None]:
    feats = dict.fromkeys(FEATURE_NAMES, np.nan)
    v = daily[daily["valid"]]
    if v.empty:
        return feats, None
    cp = detect_commissioning(daily)
    w = last365(v)
    summer_w = w[w["month"].isin(SUMMER)]
    winter_w = w[w["month"].isin(WINTER)]
    summer_all = v[v["month"].isin(SUMMER)]
    sd = _med(summer_w["s"], 30)
    if np.isnan(sd):
        sd = _med(summer_all["s"], 30)
    feats["summer_dip"] = sd
    wd = _med(winter_w["s"], 30)
    feats["seasonal_contrast"] = sd - wd if np.isfinite(sd) and np.isfinite(wd) else np.nan
    feats["summer_zero_frac"] = float(summer_w["zero"].mean()) if len(summer_w) else np.nan
    if cp is not None:
        feats["cp_drop"] = cp.drop if cp.found else 0.0
        feats["cp_gain"] = cp.gain if cp.found else 0.0
    feats["log_mean_daily_kwh"] = float(np.log1p(max(0.0, v["total"].mean())))
    feats["night_kw"] = 4 * float(v["night"].median())
    if cp is not None and cp.found:
        cut = pd.Timestamp(cp.date)
        pre_s = summer_all[summer_all.index < cut]["mid"]
        post = summer_all[summer_all.index >= cut]["mid"]
    else:
        pre_s, post = pd.Series(dtype=float), summer_w["mid"]
    pre = _med(pre_s, 10)
    if np.isnan(pre):
        pre = _med(winter_w["mid"], 1)
    if np.isfinite(pre) and len(post):
        feats["dip_kw"] = 4 * max(0.0, pre - float(np.nanquantile(post, 0.10)))
    return feats, cp
