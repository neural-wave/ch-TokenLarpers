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


def _expected_support(date_value: date) -> tuple[int, tuple[int, ...]]:
    """Return total and per-band interval support for one local calendar day."""
    start = datetime.combine(date_value, time.min, tzinfo=_TZ)
    next_start = datetime.combine(date_value + timedelta(days=1), time.min, tzinfo=_TZ)
    expected_count = (int(next_start.timestamp()) - int(start.timestamp())) // 900
    if expected_count == 96:
        return expected_count, tuple((end_hour - start_hour) * 4 for start_hour, end_hour in _BANDS.values())

    # DST days are the only days needing an interval index: normal days use
    # the fixed support above, while this bounded index captures skipped or
    # repeated local slots for each affected band exactly.
    slots = pd.date_range(pd.Timestamp(start), pd.Timestamp(next_start), freq="15min", inclusive="left")
    local_slots = slots.hour * 4 + slots.minute // 15
    supports = tuple(
        int(((local_slots >= start_hour * 4) & (local_slots < end_hour * 4)).sum())
        for start_hour, end_hour in _BANDS.values()
    )
    return expected_count, supports


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
    customer_results: list[pd.DataFrame] = []

    for customer, customer_frame in frame.groupby("customer_id", sort=True):
        first_date = customer_frame["local_date"].min()
        last_date = customer_frame["local_date"].max()
        dates = pd.date_range(first_date, last_date, freq="D").date
        support = [_expected_support(day) for day in dates]
        day = pd.DataFrame({
            "local_date": dates,
            "expected_intervals": [item[0] for item in support],
            "_expected_night": [item[1][0] for item in support],
            "_expected_morning": [item[1][1] for item in support],
            "_expected_midday": [item[1][2] for item in support],
            "_expected_evening": [item[1][3] for item in support],
        })

        observed = customer_frame.loc[
            customer_frame["observed"], ["local_date", "local_slot", "energy_kwh"]
        ].copy()
        if observed.empty:
            day["observed_intervals"] = 0
            day["energy_observed_kwh"] = None
            day["_zero_count"] = 0
            day["_negative_count"] = 0
            band_agg = pd.DataFrame(columns=["local_date", "_band", "observed_count", "mean_kwh", "median_kwh", "sum_kwh"])
        else:
            observed["energy_kwh"] = observed["energy_kwh"].astype(float)
            observed["_zero"] = observed["energy_kwh"].eq(0).astype("int64")
            observed["_negative"] = observed["energy_kwh"].lt(0).astype("int64")
            grouped = observed.groupby("local_date", sort=False)
            day_agg = grouped.agg(
                observed_intervals=("energy_kwh", "size"),
                energy_observed_kwh=("energy_kwh", "sum"),
                _zero_count=("_zero", "sum"),
                _negative_count=("_negative", "sum"),
            ).reset_index()
            day = day.merge(day_agg, on="local_date", how="left")
            day["observed_intervals"] = day["observed_intervals"].fillna(0).astype("int64")
            day["_zero_count"] = day["_zero_count"].fillna(0).astype("int64")
            day["_negative_count"] = day["_negative_count"].fillna(0).astype("int64")

            observed["_band"] = "outside"
            slots = observed["local_slot"]
            for band_name, (start_hour, end_hour) in _BANDS.items():
                mask = slots.between(start_hour * 4, end_hour * 4 - 1)
                observed.loc[mask, "_band"] = band_name
            band_observed = observed.loc[observed["_band"] != "outside"]
            band_agg = band_observed.groupby(["local_date", "_band"], sort=False)["energy_kwh"].agg(
                observed_count="size", mean_kwh="mean", median_kwh="median", sum_kwh="sum"
            ).reset_index()

        day["observed_intervals"] = day.get("observed_intervals", 0)
        day["observed_intervals"] = day["observed_intervals"].fillna(0).astype("int64")
        day["coverage_fraction"] = day["observed_intervals"] / day["expected_intervals"]
        day_good = day["coverage_fraction"] >= coverage_threshold
        day["energy_observed_kwh"] = day["energy_observed_kwh"].where(day["observed_intervals"].gt(0), None)
        day["zero_fraction"] = (day["_zero_count"] / day["observed_intervals"]).where(day_good)
        day["negative_fraction"] = (day["_negative_count"] / day["observed_intervals"]).where(day_good)

        band_good: dict[str, pd.Series] = {}
        band_sums: dict[str, pd.Series] = {}
        for band_name in _BANDS:
            expected_column = f"_expected_{band_name}"
            if band_agg.empty:
                values = pd.DataFrame(index=day.index)
            else:
                values = band_agg.loc[band_agg["_band"].eq(band_name)].set_index("local_date")
            counts = day["local_date"].map(values["observed_count"] if not values.empty else {}).fillna(0).astype("int64")
            means = day["local_date"].map(values["mean_kwh"] if not values.empty else {})
            medians = day["local_date"].map(values["median_kwh"] if not values.empty else {})
            sums = day["local_date"].map(values["sum_kwh"] if not values.empty else {})
            coverage = counts / day[expected_column]
            good = day_good & (coverage >= coverage_threshold)
            band_good[band_name] = good
            band_sums[band_name] = sums
            day[f"{band_name}_observed_count"] = counts
            day[f"{band_name}_mean_kwh"] = means.where(good)
            day[f"{band_name}_median_kwh"] = medians.where(good)
            day[f"{band_name}_coverage_fraction"] = coverage

        night_sum = band_sums["night"]
        ratio_mask = band_good["midday"] & band_good["night"] & night_sum.notna() & night_sum.ne(0)
        day["midday_night_ratio"] = (band_sums["midday"] / night_sum).where(ratio_mask)
        day["customer_id"] = str(customer)
        day["is_boundary_day"] = day["local_date"].eq(first_date) | day["local_date"].eq(last_date)
        day["weekday"] = pd.to_datetime(day["local_date"]).dt.weekday
        day["month"] = pd.to_datetime(day["local_date"]).dt.month
        customer_results.append(day[_FEATURE_COLUMNS])

    return pd.concat(customer_results, ignore_index=True)[_FEATURE_COLUMNS]


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
