from __future__ import annotations

import numpy as np
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any
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


def _vector_error(masks: list[tuple[np.ndarray, str]]) -> None:
    first: tuple[int, str] | None = None
    for mask, message in masks:
        positions = np.flatnonzero(mask)
        if len(positions):
            candidate = (int(positions[0]), message)
            if first is None or candidate[0] < first[0]:
                first = candidate
    if first is not None:
        raise ValueError(first[1])


def _parse_epoch_vectorized(values: pd.Series, column: str) -> np.ndarray:
    raw = values.astype("string").str.strip()
    missing = (raw.isna() | raw.eq("")).to_numpy(dtype=bool)
    numeric = pd.to_numeric(raw, errors="coerce")
    numbers = numeric.to_numpy(dtype=float, na_value=np.nan)
    nonfinite_tokens = raw.str.fullmatch(r"[+-]?(?:nan|inf(?:inity)?)", case=False).fillna(False).to_numpy(dtype=bool)
    not_numeric = numeric.isna().to_numpy(dtype=bool) & ~missing & ~nonfinite_tokens
    not_integral = ~missing & ~not_numeric & (~np.isfinite(numbers) | (numbers != np.floor(numbers)))
    not_aligned = ~missing & ~not_numeric & ~not_integral & (np.remainder(numbers, 900) != 0)
    if column not in ("timestamp_epoch", "epoch", "unix_timestamp"):
        raise AssertionError("epoch vector parser requires an epoch column")
    _vector_error([
        (missing, "missing timestamp"),
        (not_numeric, "timestamp_epoch is not numeric"),
        (not_integral, "timestamp_epoch must be integral seconds"),
        (not_aligned, "timestamp is not aligned to a 900-second grid"),
    ])
    return numbers.astype(np.int64)


def _validate_epoch_backed_vectorized(values: pd.Series, epochs: np.ndarray) -> None:
    raw = values.astype("string").str.strip()
    missing = (raw.isna() | raw.eq("")).to_numpy(dtype=bool)
    aware_hint = raw.str.contains(r"(?:[zZ]|[+-]\d{2}:?\d{2}|UTC|GMT)$", case=False, regex=True, na=False).to_numpy(dtype=bool)
    aware = pd.to_datetime(raw.where(aware_hint), errors="coerce", utc=True, format="mixed")
    naive = pd.to_datetime(raw.where(~aware_hint), errors="coerce", format="mixed")
    invalid = (~missing) & (
        (aware_hint & aware.isna().to_numpy(dtype=bool))
        | (~aware_hint & naive.isna().to_numpy(dtype=bool))
    )
    _vector_error([(missing, "missing timestamp"), (invalid, "invalid timestamp")])
    expected_utc = pd.Series(pd.to_datetime(epochs, unit="s", utc=True))
    expected_local = expected_utc.dt.tz_convert(_TZ).dt.tz_localize(None)
    expected_utc_ns = epochs.astype(np.int64) * 1_000_000_000
    expected_local_ns = expected_local.astype("int64").to_numpy()
    aware_ns = aware.astype("int64").to_numpy()
    naive_ns = naive.astype("int64").to_numpy()
    disagree = (
        (aware_hint & ~invalid & (aware_ns != expected_utc_ns))
        | (~aware_hint & ~invalid & (naive_ns != expected_local_ns))
    )
    _vector_error([(disagree, "timestamp fields disagree")])


def _parse_energy_vectorized(values: pd.Series, decimal: str) -> np.ndarray:
    raw = values.astype("string").str.strip()
    missing = (raw.isna() | raw.eq("")).to_numpy(dtype=bool)
    if decimal != ".":
        raw = raw.str.replace(decimal, ".", regex=False)
    numeric = pd.to_numeric(raw, errors="coerce")
    numbers = numeric.to_numpy(dtype=float, na_value=np.nan)
    nonfinite_tokens = raw.str.fullmatch(r"[+-]?(?:nan|inf(?:inity)?)", case=False).fillna(False).to_numpy(dtype=bool)
    not_numeric = numeric.isna().to_numpy(dtype=bool) & ~missing & ~nonfinite_tokens
    nonfinite = ~missing & ~not_numeric & ~np.isfinite(numbers)
    _vector_error([(not_numeric, "energy is not numeric"), (nonfinite, "energy must be finite")])
    result = np.empty(len(raw), dtype=object)
    result[:] = None
    present = ~missing
    result[present] = numbers[present]
    return result


def stage_measurements(sources: list[dict[str, Any]], conn: sqlite3.Connection) -> dict[str, Any]:
    """Read CSVs in bounded chunks into temporary SQLite."""
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
                if epoch_col is not None:
                    customers = chunk[customer_col].astype(str).to_numpy()
                    epochs = _parse_epoch_vectorized(chunk[parse_timestamp_col], parse_timestamp_col)
                    if timestamp_col is not None and timestamp_col != parse_timestamp_col:
                        _validate_epoch_backed_vectorized(chunk[timestamp_col], epochs)
                    energies = _parse_energy_vectorized(chunk[energy_col], source["decimal"])
                    rows = list(zip(customers.tolist(), epochs.tolist(), energies.tolist(),
                                    [source_id] * len(chunk), [role] * len(chunk)))
                else:
                    supplied_columns = [customer_col, parse_timestamp_col, energy_col] + ([timestamp_col] if timestamp_col and timestamp_col != parse_timestamp_col else [])
                    supplied_columns = list(dict.fromkeys(supplied_columns))
                    supplied_positions = {name: index for index, name in enumerate(supplied_columns)}
                    epoch_position = supplied_positions[parse_timestamp_col]
                    timestamp_position = supplied_positions.get(timestamp_col) if timestamp_col else None
                    energy_position = supplied_positions[energy_col]
                    rows = []
                    for record in chunk[supplied_columns].itertuples(index=False, name=None):
                        customer = str(record[supplied_positions[customer_col]])
                        if not customer:
                            raise ValueError(f"{role} contains empty customer_id")
                        epoch = _parse_epoch(record[epoch_position], parse_timestamp_col)
                        if timestamp_position is not None and timestamp_position != epoch_position:
                            _validate_epoch_backed_timestamp(record[timestamp_position], epoch, timestamp_col)
                        parsed_energy = _parse_energy(record[energy_position], source["decimal"])
                        rows.append((customer, epoch, parsed_energy, source_id, role))
                if rows:
                    conn.executemany("INSERT INTO raw(customer_id,timestamp_epoch,energy_kwh,source_id,role) VALUES(?,?,?,?,?)", rows)
                    retained_count += len(rows)
            conn.commit()
        except (pd.errors.ParserError, UnicodeError, OSError) as exc:
            raise ValueError(f"unable to parse measurement source for role {role}") from exc
        counts["input_rows"][role] = counts["input_rows"].get(role, 0) + input_count
        counts["retained_rows"][role] = counts["retained_rows"].get(role, 0) + retained_count
    conn.execute("CREATE INDEX IF NOT EXISTS raw_customer_epoch ON raw(customer_id, timestamp_epoch)")
    conn.commit()
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
        epochs = np.arange(int(first), int(last) + 900, 900, dtype=np.int64)
        observed_rows = conn.execute("SELECT timestamp_epoch, energy_kwh FROM raw WHERE customer_id=? ORDER BY timestamp_epoch", (customer,)).fetchall()
        observed_epochs = np.asarray([row[0] for row in observed_rows], dtype=np.int64)
        positions = np.searchsorted(epochs, observed_epochs)
        present = np.zeros(len(epochs), dtype=bool)
        present[positions] = True
        energy = np.empty(len(epochs), dtype=object)
        energy[:] = None
        observed_values = np.asarray([row[1] for row in observed_rows], dtype=object)
        nonmissing = pd.notna(observed_values)
        energy[positions[nonmissing]] = observed_values[nonmissing]
        counts["gap_intervals"] += int((~present).sum())
        utc = pd.Series(pd.to_datetime(epochs, unit="s", utc=True))
        local = utc.dt.tz_convert(_TZ)
        utc_naive = utc.dt.tz_localize(None)
        local_naive = local.dt.tz_localize(None)
        offset = ((local_naive - utc_naive).dt.total_seconds() // 60).astype("int16")
        # Include a bounded transition context so fold remains correct when a
        # customer's observed range starts or ends inside the repeated hour.
        probe_epochs = np.arange(int(first) - 7200, int(first) - 7200 + len(epochs) * 900 + 14400, 900, dtype=np.int64)
        probe_local = pd.Series(pd.to_datetime(probe_epochs, unit="s", utc=True)).dt.tz_convert(_TZ).dt.tz_localize(None)
        fold = probe_local.duplicated(keep="first").astype("uint8").iloc[8:8 + len(epochs)].to_numpy()
        status = np.where(~present, "absent_interval", np.where(pd.isna(energy), "missing_input", "observed"))
        frame = pd.DataFrame({
            "customer_id": [customer] * len(epochs),
            "timestamp_epoch": epochs,
            "timestamp_local": local,
            "local_date": local_naive.dt.date,
            "local_slot": (local.dt.hour * 4 + local.dt.minute // 15).astype("uint8"),
            "dst_fold": fold,
            "utc_offset_minutes": offset,
            "energy_kwh": energy,
            "value_status": status,
        }, columns=["customer_id", "timestamp_epoch", "timestamp_local", "local_date", "local_slot", "dst_fold", "utc_offset_minutes", "energy_kwh", "value_status"])
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


