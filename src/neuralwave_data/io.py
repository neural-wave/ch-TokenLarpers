from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import LOCAL_TZ, MEASUREMENT_ARROW_SCHEMA, arrow_table, empty_measurements

_TZ = ZoneInfo(LOCAL_TZ)
_ALLOWED_ROLES = {"development_measurements", "unlabelled_measurements", "inference_measurements", "development_labels", "extension_labels"}


def _columns(frame: pd.DataFrame, required: str) -> str:
    aliases = {
        "customer_id": ("customer_id", "CustomerID", "customer", "pod_id", "POD"),
        "timestamp": ("timestamp", "timestamp_local", "datetime", "date_time", "time", "utc_timestamp"),
        "timestamp_epoch": ("timestamp_epoch", "epoch", "unix_timestamp"),
        "energy": ("energy_kwh", "energy_consumption_kwh", "energy", "value", "consumption", "power_kwh"),
    }
    for name in aliases[required]:
        if name in frame.columns:
            return name
    raise ValueError(f"input is missing required {required} column")


def _parse_epoch(value: Any, column: str) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        raise ValueError("missing timestamp")
    if column == "timestamp_epoch" or column in ("epoch", "unix_timestamp"):
        try:
            number = float(value)
        except Exception as exc:
            raise ValueError("timestamp_epoch is not numeric") from exc
        if not number.is_integer():
            raise ValueError("timestamp_epoch must be integral seconds")
        epoch = int(number)
    else:
        text = str(value).strip()
        # Numeric timestamps are accepted only when the explicit epoch column is used.
        try:
            stamp = pd.Timestamp(text)
        except Exception as exc:
            raise ValueError("invalid timestamp") from exc
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("naive timestamps are unsupported; provide an explicit offset")
        epoch = int(stamp.timestamp())
        if stamp.timestamp() != epoch:
            raise ValueError("timestamp must have whole-second precision")
    if epoch % 900 != 0:
        raise ValueError("timestamp is not aligned to a 900-second grid")
    return epoch

def _validate_epoch_backed_timestamp(value: Any, epoch: int, column: str) -> None:
    if value is None or str(value).strip() == "":
        raise ValueError("missing timestamp")
    try:
        stamp = pd.Timestamp(str(value).strip())
    except Exception as exc:
        raise ValueError("invalid timestamp") from exc
    local = datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(_TZ)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        # Challenge files may supply a naive Swiss wall-clock alongside epoch seconds.
        if stamp.to_pydatetime().replace(tzinfo=None) != local.replace(tzinfo=None):
            raise ValueError("timestamp fields disagree")
    elif stamp.timestamp() != epoch:
        raise ValueError("timestamp fields disagree")


def _parse_energy(value: Any, decimal: str) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if decimal != ".":
        text = text.replace(decimal, ".")
    try:
        number = float(text)
    except Exception as exc:
        raise ValueError("energy is not numeric") from exc
    if not pd.notna(number) or number in (float("inf"), float("-inf")):
        raise ValueError("energy must be finite")
    return number


def _source_path(manifest_dir: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("manifest source path must be a non-empty string")
    expanded = os.path.expandvars(raw)
    path = Path(expanded)
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError("manifest source path does not exist")
    return path


def load_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is unreadable or invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list) or not payload["sources"]:
        raise ValueError("manifest must contain a non-empty sources list")
    for source in payload["sources"]:
        if not isinstance(source, dict) or source.get("role") not in _ALLOWED_ROLES:
            raise ValueError("manifest contains an unsupported source role")
        if "path" not in source:
            raise ValueError("manifest source is missing path")
        source.setdefault("delimiter", ";")
        source.setdefault("decimal", ".")
        source.setdefault("encoding", "utf-8")
        if not isinstance(source["delimiter"], str) or len(source["delimiter"]) != 1:
            raise ValueError("delimiter must be one character")
        if source["decimal"] not in (".", ","):
            raise ValueError("decimal must be '.' or ','")
        if not isinstance(source["encoding"], str) or not source["encoding"]:
            raise ValueError("encoding must be explicit")
        source["_path"] = str(_source_path(manifest_path.parent, source["path"]))
    return payload, manifest_path


def source_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS raw (customer_id TEXT NOT NULL, timestamp_epoch INTEGER NOT NULL, energy_kwh REAL, source_id INTEGER NOT NULL, role TEXT NOT NULL)")
    conn.execute("CREATE INDEX IF NOT EXISTS raw_customer_epoch ON raw(customer_id, timestamp_epoch)")


def stage_measurements(sources: list[dict[str, Any]], conn: sqlite3.Connection) -> dict[str, Any]:
    """Read CSVs in chunks into temporary SQLite; never materialize all input rows."""
    _ensure_table(conn)
    counts: dict[str, Any] = {"input_rows": {}, "retained_rows": {}, "excluded_null_customer": 0,
                               "excluded_non_pod": 0, "duplicate_rows": 0, "gap_intervals": 0}
    for source_id, source in enumerate(sources):
        role = source["role"]
        path = source["_path"]
        input_count = retained_count = 0
        try:
            chunks = pd.read_csv(path, sep=source["delimiter"], encoding=source["encoding"],
                                 dtype=str, chunksize=10_000, keep_default_na=False, na_filter=False)
            for chunk in chunks:
                input_count += len(chunk)
                customer_col = _columns(chunk, "customer_id")
                epoch_col = next((c for c in ("timestamp_epoch", "epoch", "unix_timestamp") if c in chunk.columns), None)
                timestamp_col = _columns(chunk, "timestamp") if any(c in chunk.columns for c in ("timestamp", "timestamp_local", "datetime", "date_time", "time", "utc_timestamp")) else None
                parse_timestamp_col = epoch_col or timestamp_col
                if parse_timestamp_col is None:
                    raise ValueError("input is missing timestamp column")
                energy_col = _columns(chunk, "energy")
                supplied_columns = [customer_col, parse_timestamp_col, energy_col] + ([timestamp_col] if timestamp_col and timestamp_col != parse_timestamp_col else [])
                supplied_columns = list(dict.fromkeys(supplied_columns))
                supplied_positions = {name: index for index, name in enumerate(supplied_columns)}
                epoch_position = supplied_positions[parse_timestamp_col]
                timestamp_position = supplied_positions.get(timestamp_col) if timestamp_col else None
                energy_position = supplied_positions[energy_col]
                # This is intentionally before timestamp/energy parsing for unlabelled junk rows.
                ids = chunk[customer_col]
                if role == "unlabelled_measurements":
                    null_mask = ids.isna() | ids.eq("")
                    pod_mask = ids.notna() & ids.ne("") & ids.str.startswith("POD", na=False)
                    counts["excluded_null_customer"] += int(null_mask.sum())
                    counts["excluded_non_pod"] += int((~null_mask & ~pod_mask).sum())
                    chunk = chunk.loc[pod_mask].copy()
                else:
                    bad = ids.isna() | ids.eq("")
                    if bool(bad.any()):
                        raise ValueError(f"{role} contains null or empty customer_id")
                rows: list[tuple[str, int, float | None, int, str]] = []
                for record in chunk[supplied_columns].itertuples(index=False, name=None):
                    customer = record[supplied_positions[customer_col]]
                    timestamp = record[epoch_position]
                    energy = record[energy_position]
                    customer = str(customer)
                    if not customer:
                        raise ValueError(f"{role} contains empty customer_id")
                    epoch = _parse_epoch(timestamp, parse_timestamp_col)
                    if timestamp_position is not None and timestamp_position != epoch_position:
                        _validate_epoch_backed_timestamp(record[timestamp_position], epoch, timestamp_col)
                    parsed_energy = _parse_energy(energy, source["decimal"])
                    rows.append((customer, epoch, parsed_energy, source_id, role))
                if rows:
                    conn.executemany("INSERT INTO raw(customer_id,timestamp_epoch,energy_kwh,source_id,role) VALUES(?,?,?,?,?)", rows)
                    retained_count += len(rows)
            conn.commit()
        except (pd.errors.ParserError, UnicodeError, OSError) as exc:
            raise ValueError(f"unable to parse measurement source for role {role}") from exc
        counts["input_rows"][role] = counts["input_rows"].get(role, 0) + input_count
        counts["retained_rows"][role] = counts["retained_rows"].get(role, 0) + retained_count
    conflict = conn.execute("SELECT customer_id,timestamp_epoch FROM raw GROUP BY customer_id,timestamp_epoch HAVING (COUNT(*) > 1 AND COUNT(energy_kwh) > 0 AND COUNT(energy_kwh) < COUNT(*)) OR (COUNT(*) > 1 AND MIN(energy_kwh) <> MAX(energy_kwh))").fetchone()
    if conflict:
        raise ValueError("conflicting duplicate measurement values")
    overlap = conn.execute("SELECT customer_id FROM raw GROUP BY customer_id HAVING COUNT(DISTINCT source_id) > 1").fetchone()
    if overlap:
        raise ValueError("measurement sources overlap on a customer_id")
    duplicate_count = conn.execute("SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM raw GROUP BY customer_id,timestamp_epoch)").fetchone()[0]
    counts["duplicate_rows"] = int(duplicate_count)
    return counts


def _timestamp_fields(epoch: int) -> tuple[datetime, str, int, int, int]:
    local = datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(_TZ)
    return local, local.date().isoformat(), local.hour * 4 + local.minute // 15, int(local.fold), int(local.utcoffset().total_seconds() // 60)


def write_canonical(conn: sqlite3.Connection, output: str | Path, counts: dict[str, Any]) -> int:
    output = Path(output)
    writer: pq.ParquetWriter | None = None
    total = 0
    customer_rows = conn.execute("SELECT customer_id, MIN(timestamp_epoch), MAX(timestamp_epoch) FROM raw GROUP BY customer_id ORDER BY customer_id")
    for customer, first, last in customer_rows:
        values = {row[0]: row[1] for row in conn.execute("SELECT timestamp_epoch, energy_kwh FROM raw WHERE customer_id=?", (customer,))}
        frame_rows = []
        epoch = int(first)
        while epoch <= int(last):
            if epoch in values:
                energy = values[epoch]
                status = "missing_input" if energy is None else "observed"
            else:
                energy = None
                status = "absent_interval"
                counts["gap_intervals"] += 1
            local, local_date, slot, fold, offset = _timestamp_fields(epoch)
            frame_rows.append((customer, epoch, local, datetime.fromisoformat(local_date).date(), slot, fold, offset, energy, status))
            epoch += 900
        frame = pd.DataFrame(frame_rows, columns=["customer_id", "timestamp_epoch", "timestamp_local", "local_date", "local_slot", "dst_fold", "utc_offset_minutes", "energy_kwh", "value_status"])
        frame["local_slot"] = frame["local_slot"].astype("uint8")
        frame["dst_fold"] = frame["dst_fold"].astype("uint8")
        frame["utc_offset_minutes"] = frame["utc_offset_minutes"].astype("int16")
        table = arrow_table(frame, "measurements")
        if writer is None:
            writer = pq.ParquetWriter(output, MEASUREMENT_ARROW_SCHEMA)
        writer.write_table(table)
        total += len(frame)
    if writer is None:
        pq.write_table(arrow_table(empty_measurements(), "measurements"), output)
    else:
        writer.close()
    return total
