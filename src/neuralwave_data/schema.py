"""Versioned, boring data contract for the NeuralWave preprocessing pipeline."""
from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
import pyarrow as pa

SCHEMA_VERSION = 1
LOCAL_TZ = "Europe/Zurich"

MEASUREMENT_COLUMNS = [
    "customer_id", "timestamp_epoch", "timestamp_local", "local_date", "local_slot",
    "dst_fold", "utc_offset_minutes", "energy_kwh", "value_status",
]
LABEL_COLUMNS = ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"]
EXTENSION_COLUMNS = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]
SPLIT_COLUMNS = ["customer_id", "fold"]

MEASUREMENT_ARROW_SCHEMA = pa.schema([
    pa.field("customer_id", pa.string(), nullable=False),
    pa.field("timestamp_epoch", pa.int64(), nullable=False),
    pa.field("timestamp_local", pa.timestamp("s", tz=LOCAL_TZ), nullable=False),
    pa.field("local_date", pa.date32(), nullable=False),
    pa.field("local_slot", pa.uint8(), nullable=False),
    pa.field("dst_fold", pa.uint8(), nullable=False),
    pa.field("utc_offset_minutes", pa.int16(), nullable=False),
    pa.field("energy_kwh", pa.float64(), nullable=True),
    pa.field("value_status", pa.string(), nullable=False),
])
LABEL_ARROW_SCHEMA = pa.schema([
    pa.field("customer_id", pa.string(), nullable=False), pa.field("pv_installed", pa.bool_(), nullable=False),
    pa.field("pv_installation_date", pa.date32(), nullable=True), pa.field("pv_capacity_kwp", pa.float64(), nullable=True),
])
EXTENSION_ARROW_SCHEMA = pa.schema([
    pa.field("customer_id", pa.string(), nullable=False), pa.field("extension_date", pa.date32(), nullable=False),
    pa.field("pv_capacity_after_extension_kwp", pa.float64(), nullable=False),
])
SPLIT_ARROW_SCHEMA = pa.schema([pa.field("customer_id", pa.string(), nullable=False), pa.field("fold", pa.int64(), nullable=False)])


def empty_measurements() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": pd.Series(dtype="string"), "timestamp_epoch": pd.Series(dtype="int64"),
        "timestamp_local": pd.Series(pd.to_datetime([], utc=True), dtype=f"datetime64[ns, {LOCAL_TZ}]"),
        "local_date": pd.Series([], dtype="object"), "local_slot": pd.Series(dtype="uint8"),
        "dst_fold": pd.Series(dtype="uint8"), "utc_offset_minutes": pd.Series(dtype="int16"),
        "energy_kwh": pd.Series(dtype="float64"), "value_status": pd.Series(dtype="string"),
    }, columns=MEASUREMENT_COLUMNS)


def empty_labels() -> pd.DataFrame:
    return pd.DataFrame({"customer_id": pd.Series(dtype="string"), "pv_installed": pd.Series(dtype="bool"),
                         "pv_installation_date": pd.Series([], dtype="object"), "pv_capacity_kwp": pd.Series(dtype="float64")}, columns=LABEL_COLUMNS)


def empty_extensions() -> pd.DataFrame:
    return pd.DataFrame({"customer_id": pd.Series(dtype="string"), "extension_date": pd.Series([], dtype="object"),
                         "pv_capacity_after_extension_kwp": pd.Series(dtype="float64")}, columns=EXTENSION_COLUMNS)


def empty_splits() -> pd.DataFrame:
    return pd.DataFrame({"customer_id": pd.Series(dtype="string"), "fold": pd.Series(dtype="int64")}, columns=SPLIT_COLUMNS)


def arrow_table(frame: pd.DataFrame, kind: str) -> pa.Table:
    """Convert a frame to the exact persisted schema, including empty outputs."""
    schemas = {"measurements": MEASUREMENT_ARROW_SCHEMA, "labels": LABEL_ARROW_SCHEMA,
               "extensions": EXTENSION_ARROW_SCHEMA, "splits": SPLIT_ARROW_SCHEMA}
    if kind not in schemas:
        raise ValueError(f"unsupported schema kind: {kind}")
    schema = schemas[kind]
    frame = frame.copy()
    frame = frame.reindex(columns=schema.names)
    # Arrow handles Python date objects and nullable values reliably; force strings to avoid ID coercion.
    if "customer_id" in frame:
        frame["customer_id"] = frame["customer_id"].astype("string")
    if "timestamp_local" in frame:
        frame["timestamp_local"] = pd.to_datetime(frame["timestamp_local"], utc=False).dt.tz_convert(LOCAL_TZ)
    return pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)


def validate_schema_version(value: object) -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {value!r}; expected {SCHEMA_VERSION}")
