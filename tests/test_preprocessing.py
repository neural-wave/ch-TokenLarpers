from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).parents[1]


def write_source(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def run_cli(tmp_path: Path, sources: list[dict], *, folds: int | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for item in sources:
        source_path = Path(item["path"])
        if source_path.is_absolute() and source_path.parent != tmp_path:
            shutil.copy2(source_path, tmp_path / source_path.name)
            item["path"] = source_path.name
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    output = tmp_path / "out"
    command = [sys.executable, "-m", "neuralwave_data.prepare", "--manifest", str(manifest), "--out", str(output)]
    if folds is not None:
        command += ["--folds", str(folds), "--seed", "11"]
    result = subprocess.run(command, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return output


def source(path: Path, role: str) -> dict:
    return {"role": role, "path": str(path.resolve()), "delimiter": ";", "decimal": ".", "encoding": "utf-8"}


def test_cli_filters_pod_before_malformed_parse_and_preserves_status_dst(tmp_path: Path):
    data = tmp_path / "measurements.csv"
    # The transition contains both 02:00 wall clocks. Junk records intentionally have malformed fields.
    rows = ["customer_id;timestamp;energy_kwh"]
    rows += [
        "POD001;2024-10-27T00:00:00+02:00;1",
        "POD001;2024-10-27T00:15:00+02:00;",
        "POD001;2024-10-27T00:45:00+02:00;-2",
        "POD001;2024-10-27T02:00:00+02:00;3",
        "POD001;2024-10-27T02:00:00+01:00;4",
        "pod001;not-a-time;not-a-number",
        "NOPE;not-a-time;not-a-number",
        ";not-a-time;not-a-number",
    ]
    write_source(data, "\n".join(rows))
    output = run_cli(tmp_path, [source(data, "unlabelled_measurements")])
    measurements = pq.read_table(output / "measurements.parquet").to_pandas()
    assert set(measurements.customer_id) == {"POD001"}
    assert measurements.energy_kwh.dropna().tolist() == [1.0, -2.0, 3.0, 4.0]
    assert "missing_input" in set(measurements.value_status)
    assert "absent_interval" in set(measurements.value_status)
    assert measurements.dst_fold.max() == 1
    assert measurements.utc_offset_minutes.nunique() == 2
    report = json.loads((output / "qa_report.json").read_text())
    assert report["counts"]["excluded_non_pod"] == 2
    assert report["counts"]["excluded_null_customer"] == 1
    assert report["counts"]["gap_intervals"] > 0


def test_daily_dst_expected_and_null_partial_statistics(tmp_path: Path):
    data = tmp_path / "m.csv"
    write_source(data, "customer_id;timestamp;energy_kwh\nPOD;2024-03-31T00:00:00+01:00;1\nPOD;2024-03-31T03:00:00+02:00;2\n")
    output = run_cli(tmp_path, [source(data, "development_measurements")])
    daily = pq.read_table(output / "daily_features.parquet").to_pandas()
    assert daily.expected_intervals.tolist() == [92]
    assert daily.observed_intervals.tolist() == [2]
    assert daily.observed_energy_kwh_sum.tolist() == [3.0]
    assert daily.midday_mean_kwh.isna().all()


def test_labels_and_customer_disjoint_deterministic_folds(tmp_path: Path):
    data = tmp_path / "m.csv"
    labels = tmp_path / "labels.csv"
    write_source(data, "customer_id;timestamp_epoch;energy_kwh\n001;0;1\n002;0;2\n003;0;3\n004;0;4\n")
    write_source(labels, "customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n001;true;2020-01-01;2\n002;false;;\n003;true;2020-01-01;1\n004;false;;\n")
    first = run_cli(tmp_path / "a", [source(data, "development_measurements"), source(labels, "development_labels")], folds=2)
    second = run_cli(tmp_path / "b", [source(data, "development_measurements"), source(labels, "development_labels")], folds=2)
    labels_out = pq.read_table(first / "labels.parquet").to_pandas()
    splits_a = pq.read_table(first / "splits.parquet").to_pandas().sort_values("customer_id")
    splits_b = pq.read_table(second / "splits.parquet").to_pandas().sort_values("customer_id")
    assert labels_out.customer_id.tolist() == ["001", "002", "003", "004"]
    assert splits_a.equals(splits_b)
    assert splits_a.groupby("fold").size().min() >= 2
    assert set(splits_a.customer_id) == set(labels_out.customer_id)


def test_conflicting_duplicate_fails_but_exact_duplicate_collapses(tmp_path: Path):
    data = tmp_path / "m.csv"
    write_source(data, "customer_id;timestamp_epoch;energy_kwh\nP;0;1\nP;0;1\n")
    output = run_cli(tmp_path, [source(data, "development_measurements")])
    assert len(pq.read_table(output / "measurements.parquet")) == 1
    bad = tmp_path / "bad.csv"
    write_source(bad, "customer_id;timestamp_epoch;energy_kwh\nP;0;1\nP;0;2\n")
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"sources": [source(bad, "development_measurements")]}))
    result = subprocess.run([sys.executable, "-m", "neuralwave_data.prepare", "--manifest", str(manifest), "--out", str(tmp_path / "badout")], cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode != 0
    assert "conflicting duplicate" in result.stderr


def test_unsupported_timestamp_and_measurement_only_are_clear(tmp_path: Path):
    data = tmp_path / "m.csv"
    write_source(data, "customer_id;timestamp;energy_kwh\nP;2024-01-01 00:00:00;1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": [source(data, "development_measurements")]}))
    result = subprocess.run([sys.executable, "-m", "neuralwave_data.prepare", "--manifest", str(manifest), "--out", str(tmp_path / "out")], cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode != 0
    assert "naive timestamps" in result.stderr
