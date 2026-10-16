"""Regular, leakage-safe customer windows."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timezone
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class WindowConfig:
    context_length: int = 672
    stride: int = 96
    min_observed_fraction: float = 0.75
    max_windows_per_customer: int | None = None

@dataclass
class WindowBatch:
    values: np.ndarray
    observed: np.ndarray
    calendar: np.ndarray
    magnitude: np.ndarray
    customer_ids: list[str]
    center_dates: list[date]
    start_epochs: np.ndarray

def _calendar(epoch: int) -> np.ndarray:
    t = pd.Timestamp(epoch, unit="s", tz="UTC").tz_convert("Europe/Zurich")
    tod = (t.hour * 3600 + t.minute * 60 + t.second) / 86400.0
    wd = t.weekday() / 7.0
    return np.array([np.sin(2*np.pi*tod), np.cos(2*np.pi*tod), np.sin(2*np.pi*wd), np.cos(2*np.pi*wd)], dtype=np.float32)

def build_windows(measurements: pd.DataFrame, config: WindowConfig) -> WindowBatch:
    if config.context_length <= 0 or config.stride <= 0 or not (0 <= config.min_observed_fraction <= 1):
        raise ValueError("invalid WindowConfig")
    if config.max_windows_per_customer is not None and config.max_windows_per_customer <= 0:
        raise ValueError("max_windows_per_customer must be positive")
    required = {"customer_id", "timestamp_epoch", "energy_kwh", "value_status"}
    missing = required - set(measurements.columns)
    if missing: raise ValueError(f"measurements missing columns: {sorted(missing)}")
    if measurements["customer_id"].isna().any(): raise ValueError("customer_id cannot be null")
    L, step, stride = config.context_length, 900, config.stride * 900
    vals, masks, cals, mags, ids, dates, starts = [], [], [], [], [], [], []
    for customer, frame in measurements.groupby("customer_id", sort=True):
        raw_epochs = pd.to_numeric(frame["timestamp_epoch"], errors="coerce")
        if raw_epochs.isna().any() or (raw_epochs % 1 != 0).any():
            raise ValueError(f"invalid timestamp_epoch for customer {customer!r}")
        epochs = raw_epochs.astype("int64")
        if (epochs % step != 0).any():
            raise ValueError(f"timestamp_epoch is not on the 900-second grid for customer {customer!r}")
        if epochs.duplicated().any():
            raise ValueError(f"duplicate timestamp_epoch for customer {customer!r}")
        if epochs.empty: continue
        origin = int(epochs.min())
        end = int(epochs.max())
        grid = np.arange(origin, end + step, step, dtype=np.int64)
        lookup: dict[int, float] = {}
        for row in frame.itertuples(index=False):
            e = int(getattr(row, "timestamp_epoch"))
            status = str(getattr(row, "value_status"))
            x = getattr(row, "energy_kwh")
            try:
                valid_value = status.lower() == "observed" and bool(np.isfinite(x))
            except (TypeError, ValueError):
                valid_value = False
            if valid_value: lookup[e] = float(x)
        present = np.array([lookup.get(int(e), np.nan) for e in grid], dtype=np.float64)
        available = np.isfinite(present)
        if not available.any(): continue
        if len(grid) <= L:
            starts_local = [int(grid[0] - (L-len(grid))*step)]
        else:
            max_start = int(grid[-1] - (L-1)*step)
            starts_local = list(range(int(grid[0]), max_start + 1, stride))
            tail = max_start
            if starts_local[-1] != tail: starts_local.append(tail)
        if config.max_windows_per_customer is not None and len(starts_local) > config.max_windows_per_customer:
            limit = config.max_windows_per_customer
            chosen = np.linspace(0, len(starts_local) - 1, limit, dtype=np.int64)
            starts_local = [starts_local[int(i)] for i in np.unique(chosen)]
        for start in starts_local:
            idx0 = int(round((start - origin) / step)); idx1 = idx0 + L
            x = np.zeros(L, dtype=np.float32); m = np.zeros(L, dtype=bool)
            lo, hi = max(0, idx0), min(len(grid), idx1)
            if hi > lo:
                src = present[lo:hi]; good = np.isfinite(src)
                x[(lo-idx0):(hi-idx0)][good] = src[good].astype(np.float32)
                m[(lo-idx0):(hi-idx0)] = good
            count = int(m.sum())
            if count == 0 or count / L < config.min_observed_fraction: continue
            observed_values = x[m].astype(np.float64)
            center = float(np.median(observed_values)); mad = float(np.median(np.abs(observed_values-center)))
            scale = max(mad, 0.001)
            mean = float(np.mean(observed_values)); std = float(np.std(observed_values))
            xn = np.zeros(L, dtype=np.float32); xn[m] = ((observed_values-center)/scale).astype(np.float32)
            ep = start + (L//2)*step
            vals.append(xn[:,None]); masks.append(m[:,None]); cals.append(np.stack([_calendar(start+i*step) for i in range(L)]))
            mags.append([center, scale, mean, std]); ids.append(str(customer)); dates.append(pd.Timestamp(ep, unit="s", tz="UTC").tz_convert("Europe/Zurich").date()); starts.append(start)
    if not vals:
        return WindowBatch(np.empty((0,L,1),np.float32), np.empty((0,L,1),bool), np.empty((0,L,4),np.float32), np.empty((0,4),np.float32), [], [], np.empty(0,np.int64))
    return WindowBatch(np.stack(vals), np.stack(masks), np.stack(cals), np.asarray(mags,np.float32), ids, dates, np.asarray(starts,np.int64))
