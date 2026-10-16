from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import LOCAL_TZ

_TZ = ZoneInfo(LOCAL_TZ)
_BANDS = {"night": (0, 5), "morning": (6, 9), "midday": (10, 15), "evening": (18, 22)}


def _expected_day(date_value: date) -> pd.DataFrame:
    start = datetime.combine(date_value, time.min, tzinfo=_TZ)
    next_start = datetime.combine(date_value + timedelta(days=1), time.min, tzinfo=_TZ)
    epochs = range(int(start.timestamp()), int(next_start.timestamp()), 900)
    rows = []
    for epoch in epochs:
        local = datetime.fromtimestamp(epoch, tz=_TZ)
        rows.append((epoch, local.hour * 4 + local.minute // 15))
    return pd.DataFrame(rows, columns=["timestamp_epoch", "local_slot"])


def daily_features(measurements: pd.DataFrame, coverage_threshold: float = 0.9) -> pd.DataFrame:
    if not 0 < coverage_threshold <= 1:
        raise ValueError("coverage_threshold must be in (0, 1]")
    columns = ["customer_id", "local_date", "expected_intervals", "observed_intervals", "coverage_fraction",
               "is_boundary_day", "observed_energy_kwh_sum", "weekday", "month", "zero_fraction", "negative_fraction",
               "midday_night_ratio"]
    for band_name in _BANDS:
        columns += [f"{band_name}_observed_count", f"{band_name}_mean_kwh", f"{band_name}_median_kwh", f"{band_name}_coverage_fraction"]
    if measurements.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    frame = measurements.copy()
    frame["local_date"] = pd.to_datetime(frame["local_date"]).dt.date
    frame["observed"] = frame["value_status"].eq("observed") & frame["energy_kwh"].notna()
    rows: list[dict[str, object]] = []
    for customer, customer_frame in frame.groupby("customer_id", sort=True):
        first_date = min(customer_frame["local_date"])
        last_date = max(customer_frame["local_date"])
        current = first_date
        grouped = {d: g for d, g in customer_frame.groupby("local_date", sort=False)}
        while current <= last_date:
            expected = _expected_day(current)
            day = grouped.get(current, pd.DataFrame(columns=frame.columns))
            observed = day.loc[day["observed"]].copy() if not day.empty else day
            expected_count = len(expected)
            observed_count = len(observed)
            values = observed["energy_kwh"].astype(float) if observed_count else pd.Series(dtype=float)
            result: dict[str, object] = {"customer_id": str(customer), "local_date": current,
                "expected_intervals": expected_count, "observed_intervals": observed_count,
                "coverage_fraction": observed_count / expected_count if expected_count else 0.0,
                "is_boundary_day": current == first_date or current == last_date,
                "observed_energy_kwh_sum": float(values.sum()) if observed_count else None,
                "weekday": current.weekday(), "month": current.month,
                "zero_fraction": float((values == 0).mean()) if observed_count else None,
                "negative_fraction": float((values < 0).mean()) if observed_count else None,
                "midday_night_ratio": None}
            band_sums: dict[str, float | None] = {}
            band_good: dict[str, bool] = {}
            for band_name, (start_hour, end_hour) in _BANDS.items():
                expected_band = int(expected["local_slot"].between(start_hour * 4, end_hour * 4 - 1).sum())
                band_rows = day.loc[day["local_slot"].between(start_hour * 4, end_hour * 4 - 1) & day["observed"]] if not day.empty else day
                count = len(band_rows)
                band_values = band_rows["energy_kwh"].astype(float) if count else pd.Series(dtype=float)
                good = expected_band > 0 and count / expected_band >= coverage_threshold
                band_good[band_name] = good
                band_sums[band_name] = float(band_values.sum()) if count else None
                if good:
                    result.update({f"{band_name}_observed_count": count, f"{band_name}_mean_kwh": float(band_values.mean()),
                                   f"{band_name}_median_kwh": float(band_values.median()), f"{band_name}_coverage_fraction": count / expected_band})
                else:
                    result.update({f"{band_name}_observed_count": count, f"{band_name}_mean_kwh": None,
                                   f"{band_name}_median_kwh": None, f"{band_name}_coverage_fraction": count / expected_band if expected_band else 0.0})
            if band_good.get("midday") and band_good.get("night") and band_sums["night"] not in (None, 0):
                result["midday_night_ratio"] = band_sums["midday"] / band_sums["night"]
            rows.append(result)
            current += timedelta(days=1)
    return pd.DataFrame(rows, columns=columns)


def write_daily_features(measurement_path: str | Path, output: str | Path, coverage_threshold: float = 0.9) -> pd.DataFrame:
    measurements = pq.read_table(measurement_path).to_pandas()
    features = daily_features(measurements, coverage_threshold)
    pq.write_table(pa.Table.from_pandas(features, preserve_index=False), output)
    return features
