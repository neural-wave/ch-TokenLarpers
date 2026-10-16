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
_FEATURE_COLUMNS = ["customer_id", "local_date", "expected_intervals", "observed_intervals", "coverage_fraction",
    "is_boundary_day", "energy_observed_kwh", "weekday", "month", "zero_fraction", "negative_fraction", "midday_night_ratio"]
for _band in _BANDS:
    _FEATURE_COLUMNS += [f"{_band}_observed_count", f"{_band}_mean_kwh", f"{_band}_median_kwh", f"{_band}_coverage_fraction"]
FEATURE_ARROW_SCHEMA = pa.schema([
    pa.field("customer_id", pa.string(), nullable=False), pa.field("local_date", pa.date32(), nullable=False),
    pa.field("expected_intervals", pa.int64(), nullable=False), pa.field("observed_intervals", pa.int64(), nullable=False),
    pa.field("coverage_fraction", pa.float64(), nullable=False), pa.field("is_boundary_day", pa.bool_(), nullable=False),
    pa.field("energy_observed_kwh", pa.float64()), pa.field("weekday", pa.int64(), nullable=False), pa.field("month", pa.int64(), nullable=False),
    pa.field("zero_fraction", pa.float64()), pa.field("negative_fraction", pa.float64()), pa.field("midday_night_ratio", pa.float64()),
] + [field for band in _BANDS for field in (pa.field(f"{band}_observed_count", pa.int64(), nullable=False), pa.field(f"{band}_mean_kwh", pa.float64()), pa.field(f"{band}_median_kwh", pa.float64()), pa.field(f"{band}_coverage_fraction", pa.float64(), nullable=False))])


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
    if measurements.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in _FEATURE_COLUMNS})
    frame = measurements.copy()
    frame["local_date"] = pd.to_datetime(frame["local_date"]).dt.date
    frame["observed"] = frame["value_status"].eq("observed") & frame["energy_kwh"].notna()
    rows: list[dict[str, object]] = []
    for customer, customer_frame in frame.groupby("customer_id", sort=True):
        first_date = min(customer_frame["local_date"]); last_date = max(customer_frame["local_date"]); current = first_date
        grouped = {d: g for d, g in customer_frame.groupby("local_date", sort=False)}
        while current <= last_date:
            expected = _expected_day(current); day = grouped.get(current, pd.DataFrame(columns=frame.columns))
            observed = day.loc[day["observed"]].copy() if not day.empty else day
            expected_count = len(expected); observed_count = len(observed)
            values = observed["energy_kwh"].astype(float) if observed_count else pd.Series(dtype=float)
            day_coverage_good = expected_count > 0 and observed_count / expected_count >= coverage_threshold
            result: dict[str, object] = {"customer_id": str(customer), "local_date": current, "expected_intervals": expected_count,
                "observed_intervals": observed_count, "coverage_fraction": observed_count / expected_count if expected_count else 0.0,
                "is_boundary_day": current == first_date or current == last_date, "energy_observed_kwh": float(values.sum()) if observed_count else None,
                "weekday": current.weekday(), "month": current.month, "zero_fraction": float((values == 0).mean()) if observed_count and day_coverage_good else None,
                "negative_fraction": float((values < 0).mean()) if observed_count and day_coverage_good else None, "midday_night_ratio": None}
            band_sums: dict[str, float | None] = {}; band_good: dict[str, bool] = {}
            for band_name, (start_hour, end_hour) in _BANDS.items():
                expected_band = int(expected["local_slot"].between(start_hour * 4, end_hour * 4 - 1).sum())
                band_rows = day.loc[day["local_slot"].between(start_hour * 4, end_hour * 4 - 1) & day["observed"]] if not day.empty else day
                count = len(band_rows); band_values = band_rows["energy_kwh"].astype(float) if count else pd.Series(dtype=float)
                good = day_coverage_good and expected_band > 0 and count / expected_band >= coverage_threshold
                band_good[band_name] = good; band_sums[band_name] = float(band_values.sum()) if count else None
                result.update({f"{band_name}_observed_count": count, f"{band_name}_mean_kwh": float(band_values.mean()) if good else None,
                    f"{band_name}_median_kwh": float(band_values.median()) if good else None,
                    f"{band_name}_coverage_fraction": count / expected_band if expected_band else 0.0})
            if band_good.get("midday") and band_good.get("night") and band_sums["night"] not in (None, 0):
                result["midday_night_ratio"] = band_sums["midday"] / band_sums["night"]
            rows.append(result); current += timedelta(days=1)
    return pd.DataFrame(rows, columns=_FEATURE_COLUMNS)


def iter_daily_features(measurement_path: str | Path, coverage_threshold: float = 0.9):
    parquet = pq.ParquetFile(measurement_path); pending_customer = None; pending: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(batch_size=100_000):
        frame = batch.to_pandas()
        for customer, group in frame.groupby("customer_id", sort=False):
            if pending_customer is not None and customer != pending_customer:
                yield daily_features(pd.concat(pending, ignore_index=True), coverage_threshold)
                pending = []
            pending_customer = customer; pending.append(group)
    if pending:
        yield daily_features(pd.concat(pending, ignore_index=True), coverage_threshold)


def write_daily_features(measurement_path: str | Path, output: str | Path, coverage_threshold: float = 0.9) -> int:
    writer = None; total = 0
    for features in iter_daily_features(measurement_path, coverage_threshold):
        table = pa.Table.from_pandas(features, schema=FEATURE_ARROW_SCHEMA, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output, FEATURE_ARROW_SCHEMA)
        writer.write_table(table); total += len(features)
    if writer is None:
        pq.write_table(pa.Table.from_pandas(daily_features(pd.DataFrame(), coverage_threshold), schema=FEATURE_ARROW_SCHEMA, preserve_index=False), output)
    else:
        writer.close()
    return total
