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


def daily_from_sums(
    dates: np.ndarray,
    mid_sum: np.ndarray,
    mid_n: np.ndarray,
    night_sum: np.ndarray,
    night_n: np.ndarray,
    eve_sum: np.ndarray,
    eve_n: np.ndarray,
    tot_sum: np.ndarray,
    tot_n: np.ndarray,
    mid_zero: np.ndarray,
) -> pd.DataFrame:
    """Per-day block sums/counts -> the daily frame (shared by `daily_signal` and streaming)."""
    mid_n, night_n, eve_n, tot_n = (np.asarray(x, np.int64) for x in (mid_n, night_n, eve_n, tot_n))
    with np.errstate(invalid="ignore"):
        mid = np.where(mid_n > 0, mid_sum / np.maximum(mid_n, 1), np.nan)
        night = np.where(night_n > 0, night_sum / np.maximum(night_n, 1), np.nan)
        eve = np.where(eve_n > 0, eve_sum / np.maximum(eve_n, 1), np.nan)
        total = np.where(tot_n > 0, tot_sum * 96 / np.maximum(tot_n, 1), np.nan)
        zero = np.where(mid_n > 0, np.asarray(mid_zero, np.int64) / np.maximum(mid_n, 1), np.nan)
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


def daily_signal(dates: np.ndarray, M: np.ndarray) -> pd.DataFrame:
    """Columns: mid, night, eve, total, zero, valid, month, woy, weekend, s, s_eve (index: dates)."""
    with np.errstate(invalid="ignore"):
        mid_blk, night_blk, eve_blk = M[:, MID], M[:, NIGHT], M[:, EVE]
        return daily_from_sums(
            dates,
            np.nansum(mid_blk, axis=1),
            np.sum(~np.isnan(mid_blk), axis=1),
            np.nansum(night_blk, axis=1),
            np.sum(~np.isnan(night_blk), axis=1),
            np.nansum(eve_blk, axis=1),
            np.sum(~np.isnan(eve_blk), axis=1),
            np.nansum(M, axis=1),
            np.sum(~np.isnan(M), axis=1),
            np.sum(mid_blk <= 0.005, axis=1),
        )
