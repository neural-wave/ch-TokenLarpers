"""Per-day physical signals from the UTC+1 day matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

MID = slice(40, 60)  # 10:00-14:59 UTC+1
NIGHT = slice(0, 20)  # 00:00-04:59
EVE = slice(76, 88)  # 19:00-21:59
EPS = 0.01
SUMMER = (4, 5, 6, 7, 8, 9)
WINTER = (11, 12, 1, 2)


def daily_signal(dates: np.ndarray, M: np.ndarray) -> pd.DataFrame:
    """Columns: mid, night, eve, total, zero, valid, month, woy, weekend, s, s_eve (index: dates)."""
    with np.errstate(invalid="ignore"):
        mid_blk, night_blk, eve_blk = M[:, MID], M[:, NIGHT], M[:, EVE]
        mid_n = np.sum(~np.isnan(mid_blk), axis=1)
        night_n = np.sum(~np.isnan(night_blk), axis=1)
        eve_n = np.sum(~np.isnan(eve_blk), axis=1)
        all_n = np.sum(~np.isnan(M), axis=1)
        mid = np.where(mid_n > 0, np.nansum(mid_blk, axis=1) / np.maximum(mid_n, 1), np.nan)
        night = np.where(night_n > 0, np.nansum(night_blk, axis=1) / np.maximum(night_n, 1), np.nan)
        eve = np.where(eve_n > 0, np.nansum(eve_blk, axis=1) / np.maximum(eve_n, 1), np.nan)
        total = np.where(all_n > 0, np.nansum(M, axis=1) * 96 / np.maximum(all_n, 1), np.nan)
        zero = np.where(mid_n > 0, np.sum(mid_blk <= 0.005, axis=1) / np.maximum(mid_n, 1), np.nan)
    idx = pd.DatetimeIndex(dates.astype("datetime64[ns]"))
    df = pd.DataFrame(
        {
            "mid": mid,
            "night": night,
            "eve": eve,
            "total": total,
            "zero": zero,
            "valid": (mid_n >= 16) & (night_n >= 16),
            "month": idx.month.to_numpy(),
            "woy": np.minimum(idx.isocalendar().week.to_numpy().astype(int), 52),
            "weekend": idx.dayofweek.to_numpy() >= 5,
        },
        index=idx,
    )
    df["s"] = np.log((np.maximum(df["mid"], 0) + EPS) / (np.maximum(df["night"], 0) + EPS))
    df["s_eve"] = np.log((np.maximum(df["mid"], 0) + EPS) / (np.maximum(df["eve"], 0) + EPS))
    return df
