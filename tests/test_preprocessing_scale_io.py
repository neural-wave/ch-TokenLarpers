from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from neuralwave_data.io import stage_measurements, write_canonical

def _write(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep=";", index=False, lineterminator="\n")


def _source(path: Path, role: str = "development_measurements") -> dict[str, object]:
    return {"_path": path, "role": role, "delimiter": ";", "decimal": ".", "encoding": "utf-8"}

def _run(path: Path, tmp_path: Path, role: str = "development_measurements") -> tuple[pd.DataFrame, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(":memory:")
    try:
        counts = stage_measurements([_source(path, role)], conn)
        output = tmp_path / "measurements.parquet"
        write_canonical(conn, output, counts)
    finally:
        conn.close()
    return pq.read_table(output).to_pandas(), counts


def test_epoch_vectorized_matches_scalar_across_dst_fold_and_gap(tmp_path: Path) -> None:
    epochs = [1729987200, 1729988100, 1729990800, 1711843200, 1711846800]
    # The explicit Zurich wall-clock values are used to avoid depending on host TZ.
    local = [
        "2024-10-27T02:00:00+02:00", "2024-10-27T02:15:00+02:00",
        "2024-10-27T02:00:00+01:00", "2024-03-31T01:00:00+01:00",
        "2024-03-31T03:00:00+02:00",
    ]
    energies = [1, 2, 3, 4, 5]
    epoch_path = tmp_path / "epoch.csv"
    scalar_path = tmp_path / "scalar.csv"
    _write(epoch_path, [{"customer_id": "PODSYN001", "timestamp_epoch": epoch, "timestamp_local": stamp, "energy_consumption_kwh": energy} for epoch, stamp, energy in zip(epochs, local, energies)])
    _write(scalar_path, [{"customer_id": "PODSYN001", "timestamp": stamp, "energy_consumption_kwh": energy} for stamp, energy in zip(local, energies)])
    epoch_frame, _ = _run(epoch_path, tmp_path / "epoch")
    scalar_frame, _ = _run(scalar_path, tmp_path / "scalar")
    columns = ["timestamp_epoch", "timestamp_local", "local_date", "local_slot", "dst_fold", "utc_offset_minutes", "energy_kwh", "value_status"]
    pd.testing.assert_frame_equal(epoch_frame[columns].reset_index(drop=True), scalar_frame[columns].reset_index(drop=True))
    assert epoch_frame.loc[epoch_frame.timestamp_epoch == 1729990800, "dst_fold"].item() == 1
    assert epoch_frame.loc[epoch_frame.timestamp_epoch == 1711846800, "local_slot"].item() == 12


def test_unlabelled_junk_is_filtered_before_vectorized_parsing(tmp_path: Path) -> None:
    path = tmp_path / "junk.csv"
    _write(path, [
        {"customer_id": "NOPE", "timestamp_epoch": "not-a-time", "energy_consumption_kwh": "not-a-number"},
        {"customer_id": "", "timestamp_epoch": "not-a-time", "energy_consumption_kwh": "not-a-number"},
        {"customer_id": "PODSYN001", "timestamp_epoch": 0, "energy_consumption_kwh": 1},
    ])
    frame, counts = _run(path, tmp_path / "out", "unlabelled_measurements")
    assert frame.customer_id.tolist() == ["PODSYN001"]
    assert counts["excluded_non_pod"] == 1 and counts["excluded_null_customer"] == 1


def test_conflicting_duplicates_are_detected_across_chunks(tmp_path: Path) -> None:
    rows = [{"customer_id": "PODSYN001", "timestamp_epoch": index * 900, "energy_consumption_kwh": 1} for index in range(10000)]
    rows.append({"customer_id": "PODSYN001", "timestamp_epoch": 0, "energy_consumption_kwh": 2})
    path = tmp_path / "conflict.csv"
    _write(path, rows)
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ValueError, match="conflicting duplicate"):
            stage_measurements([_source(path)], conn)
    finally:
        conn.close()


def test_missing_zero_negative_and_absent_statuses(tmp_path: Path) -> None:
    path = tmp_path / "status.csv"
    _write(path, [
        {"customer_id": "PODSYN001", "timestamp_epoch": 0, "energy_consumption_kwh": ""},
        {"customer_id": "PODSYN001", "timestamp_epoch": 900, "energy_consumption_kwh": 0},
        {"customer_id": "PODSYN001", "timestamp_epoch": 1800, "energy_consumption_kwh": -2},
        {"customer_id": "PODSYN001", "timestamp_epoch": 3600, "energy_consumption_kwh": 3},
    ])
    frame, _ = _run(path, tmp_path / "out")
    selected = frame.set_index("timestamp_epoch")
    assert pd.isna(selected.loc[0, "energy_kwh"]) and selected.loc[0, "value_status"] == "missing_input"
    assert selected.loc[900, "energy_kwh"] == 0 and selected.loc[900, "value_status"] == "observed"
    assert selected.loc[1800, "energy_kwh"] == -2 and selected.loc[1800, "value_status"] == "observed"
    assert pd.isna(selected.loc[2700, "energy_kwh"]) and selected.loc[2700, "value_status"] == "absent_interval"


def test_valid_epoch_rows_cross_chunk_boundary(tmp_path: Path) -> None:
    path = tmp_path / "cross_chunk.csv"
    rows = [{"customer_id": "PODSYN001", "timestamp_epoch": index * 900, "energy_consumption_kwh": index} for index in range(10001)]
    _write(path, rows)
    frame, counts = _run(path, tmp_path / "out")
    assert counts["retained_rows"]["development_measurements"] == 10001
    assert len(frame) == 10001 and frame.timestamp_epoch.is_monotonic_increasing
