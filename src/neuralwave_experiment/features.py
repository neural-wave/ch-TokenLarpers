"""Leakage-aware feature extraction for the embedding experiment.

The functions in this module deliberately keep customer identifiers and labels out of
numeric arrays.  They operate on the canonical version-1 frames and on WindowBatch
instances produced by :mod:`neuralwave_experiment.windows`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class CustomerFeatureSet:
    customer_ids: list[str]
    matrix: np.ndarray
    definitions: list[str]


@dataclass
class DateCandidate:
    customer_id: str
    local_date: date
    features: np.ndarray
    covered: bool = True


def _finite(values: object) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _window_raw_features(batch: object) -> np.ndarray:
    """Return non-leaky per-window raw features.

    Magnitude is retained in its original units.  Calendar features are summarized
    as means, while mask and normalized shape are summarized only over observed
    points.  No customer ID is ever encoded.
    """
    values = np.asarray(batch.values, dtype=np.float32)
    observed = np.asarray(batch.observed, dtype=bool)
    calendar = np.asarray(batch.calendar, dtype=np.float32)
    if values.ndim != 3 or observed.shape != values.shape:
        raise ValueError("WindowBatch values/observed must have shape [N,L,1]")
    n = values.shape[0]
    out = np.zeros((n, 13), dtype=np.float32)
    for i in range(n):
        mask = observed[i, :, 0]
        vals = values[i, :, 0]
        finite = vals[mask & np.isfinite(vals)]
        out[i, 0] = float(mask.mean())
        if finite.size:
            out[i, 1] = float(finite.mean())
            out[i, 2] = float(finite.std())
            out[i, 3] = float(np.quantile(finite, .9) - np.quantile(finite, .1))
            # Daylight activity relative to the night clock bands.
            daylight = calendar[i, :, 0] > 0
            day = vals[mask & daylight]
            night = vals[mask & ~daylight]
            out[i, 4] = float(day.mean() - night.mean()) if day.size and night.size else 0.0
        if calendar.ndim == 3 and calendar.shape[-1] >= 4:
            out[i, 5:9] = np.nan_to_num(calendar[i].mean(axis=0)[:4], nan=0.0)
        mag = np.asarray(batch.magnitude[i], dtype=np.float32).reshape(-1)
        out[i, 9:13] = np.nan_to_num(mag[:4], nan=0.0)
    return out


def per_window_features(batch: object, embeddings: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Combine encoder embeddings with magnitude/calendar raw features."""
    raw = _window_raw_features(batch)
    names = ["coverage", "shape_mean", "shape_std", "shape_p90_p10", "day_night_shape_delta",
             "clock_sin_day", "clock_cos_day", "clock_sin_week", "clock_cos_week",
             "magnitude_center", "magnitude_robust_scale", "magnitude_mean_kwh", "magnitude_std_kwh"]
    if embeddings is None:
        return raw, names
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != raw.shape[0] or not np.isfinite(emb).all():
        raise ValueError("embeddings must be finite [number_of_windows, dimension]")
    return np.concatenate([raw, emb], axis=1), names + [f"embedding_{i}" for i in range(emb.shape[1])]


def aggregate_customer_features(batch: object, embeddings: np.ndarray | None = None) -> CustomerFeatureSet:
    """Aggregate windows using mean/std and first-versus-last changes."""
    matrix, names = per_window_features(batch, embeddings)
    ids = [str(x) for x in batch.customer_ids]
    starts = np.asarray(batch.start_epochs)
    order = np.argsort(starts, kind="stable")
    rows: list[np.ndarray] = []
    customer_ids: list[str] = []
    for customer in dict.fromkeys(ids):
        indices = np.asarray([i for i, value in enumerate(ids) if value == customer], dtype=int)
        indices = indices[np.argsort(starts[indices], kind="stable")]
        x = matrix[indices]
        mean = np.nan_to_num(x.mean(axis=0), nan=0.0)
        std = np.nan_to_num(x.std(axis=0), nan=0.0)
        delta = np.nan_to_num(x[-1] - x[0], nan=0.0) if len(x) > 1 else np.zeros(x.shape[1], dtype=np.float32)
        rows.append(np.concatenate([mean, std, delta]).astype(np.float32))
        customer_ids.append(customer)
    if not rows:
        return CustomerFeatureSet([], np.empty((0, matrix.shape[1] * 3), dtype=np.float32),
                                 names + [f"{n}_std" for n in names] + [f"{n}_change" for n in names])
    defs = names + [f"{n}_std" for n in names] + [f"{n}_change" for n in names]
    return CustomerFeatureSet(customer_ids, np.vstack(rows), defs)


def _daily_stats(measurements: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    required = {"customer_id", "energy_kwh", "value_status"}
    missing = required - set(measurements.columns)
    if missing:
        raise ValueError(f"measurements missing required columns: {sorted(missing)}")
    frame = measurements.loc[measurements.customer_id.astype(str).eq(str(customer_id))].copy()
    if frame.empty:
        return pd.DataFrame(columns=["local_date", "midday", "night", "observed"])
    if "local_date" not in frame:
        if "timestamp_local" in frame:
            frame["local_date"] = pd.to_datetime(frame.timestamp_local).dt.date
        elif "timestamp_epoch" in frame:
            frame["local_date"] = pd.to_datetime(frame.timestamp_epoch, unit="s", utc=True).dt.date
        else:
            raise ValueError("measurements require local_date or timestamp_local")
    frame["local_date"] = pd.to_datetime(frame.local_date).dt.date
    frame["observed"] = frame.value_status.eq("observed") & pd.to_numeric(frame.energy_kwh, errors="coerce").notna()
    frame["energy_kwh"] = pd.to_numeric(frame.energy_kwh, errors="coerce")
    slot = pd.to_numeric(frame.get("local_slot", pd.Series(np.nan, index=frame.index)), errors="coerce")
    hour = (slot // 4).fillna(pd.to_datetime(frame.get("timestamp_local"), errors="coerce").dt.hour if "timestamp_local" in frame else 12)
    frame["midday_mask"] = hour.between(10, 15)
    frame["night_mask"] = hour.between(0, 5)
    rows = []
    for d, g in frame.groupby("local_date", sort=True):
        obs = g[g.observed]
        mid = obs[obs["midday_mask"]]["energy_kwh"]
        night = obs[obs["night_mask"]]["energy_kwh"]
        rows.append({"local_date": d, "midday": float(mid.mean()) if len(mid) else np.nan,
                     "night": float(night.mean()) if len(night) else np.nan,
                     "observed": int(len(obs))})
    return pd.DataFrame(rows)


def date_candidates(measurements: pd.DataFrame, customer_ids: Iterable[str] | None = None,
                    min_observed_points: int = 4, window_batch: object | None = None,
                    window_embeddings: np.ndarray | None = None) -> list[DateCandidate]:
    """Build time-indexed candidates; optionally append embedding change features."""
    ids = [str(x) for x in customer_ids] if customer_ids is not None else sorted(measurements.customer_id.astype(str).unique())
    embedding_by_day: dict[tuple[str, date], list[np.ndarray]] = {}
    if window_batch is not None and window_embeddings is not None:
        emb = np.asarray(window_embeddings, dtype=np.float64)
        if len(emb) != len(window_batch.customer_ids):
            raise ValueError("window_embeddings and WindowBatch length differ")
        for customer, center, vector in zip(window_batch.customer_ids, window_batch.center_dates, emb):
            embedding_by_day.setdefault((str(customer), pd.Timestamp(center).date()), []).append(vector)
    dimension = int(np.asarray(window_embeddings).shape[1]) if window_embeddings is not None else 0
    result: list[DateCandidate] = []
    for customer in ids:
        daily = _daily_stats(measurements, customer)
        if daily.empty:
            continue
        for i, row in daily.iterrows():
            if int(row.observed) < min_observed_points:
                continue
            prev = daily.iloc[max(0, i - 3):i]; nxt = daily.iloc[i + 1:i + 4]
            before_mid = np.nanmedian(prev.midday) if len(prev) else row.midday
            after_mid = np.nanmedian(nxt.midday) if len(nxt) else row.midday
            before_night = np.nanmedian(prev.night) if len(prev) else row.night
            after_night = np.nanmedian(nxt.night) if len(nxt) else row.night
            x = [row.midday - row.night, after_mid - before_mid,
                 (after_mid - after_night) - (before_mid - before_night),
                 float(row.observed) / 96.0, float(i) / max(1, len(daily) - 1)]
            if dimension:
                def mean_emb(days: Iterable[date]) -> np.ndarray:
                    vectors = [v for day in days for v in embedding_by_day.get((customer, day), [])]
                    return np.mean(vectors, axis=0) if vectors else np.zeros(dimension)
                before = mean_emb(daily.iloc[max(0, i - 3):i].local_date)
                after = mean_emb(daily.iloc[i + 1:i + 4].local_date)
                x.extend(after - before)
            result.append(DateCandidate(customer, row.local_date, np.nan_to_num(np.asarray(x, dtype=np.float64))))
    return result


def candidate_change_point(measurements: pd.DataFrame, customer_id: str) -> date | None:
    """Declared raw-feature fallback when no train event candidates are usable."""
    daily = _daily_stats(measurements, str(customer_id))
    if len(daily) < 2:
        return None
    scores = []
    for i in range(len(daily)):
        left = daily.iloc[max(0, i - 7):i].midday
        right = daily.iloc[i + 1:i + 8].midday
        if not len(left) or not len(right):
            scores.append(-np.inf)
        else:
            scores.append(abs(float(np.nanmedian(right) - np.nanmedian(left))))
    idx = int(np.argmax(scores))
    return daily.iloc[idx].local_date if np.isfinite(scores[idx]) else None
