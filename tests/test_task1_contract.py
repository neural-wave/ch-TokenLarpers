"""The Task 1 detector and wild inference on the shared contract; fabricated data only."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src"), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
HEADER = "customer_id;repeat;fold;prob;pv_installed;pv_installation_date;pv_capacity_kwp"


def _canonical(tmp_path: Path, days: int = 40) -> tuple[Path, list[str]]:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    times = pd.date_range("2025-03-01", periods=days * 96, freq="15min", tz="Europe/Zurich")
    rng = np.random.default_rng(0)
    parts, labels, ids = [], [], []
    for i in range(8):
        customer = f"ARTIFICIAL_{i:02}"
        ids.append(customer)
        positive = i >= 4
        onset = pd.Timestamp("2025-03-13").date()
        hour = times.hour.to_numpy() + times.minute.to_numpy() / 60
        load = 3 + .1 * i + .2 * np.cos(hour * np.pi / 12) + rng.normal(0, .05, len(times))
        effect = 2.5 * np.maximum(0, np.sin((hour - 6) * np.pi / 12))
        active = np.array([d >= onset for d in times.date]) & positive
        values = np.maximum(load - effect * active, 0)
        parts.append(pd.DataFrame({"customer_id": customer, "timestamp_epoch": times.as_unit("s").asi8,
                                   "energy_kwh": values, "value_status": "observed"}))
        labels.append({"customer_id": customer, "pv_installed": positive,
                       "pv_installation_date": onset if positive else None,
                       "pv_capacity_kwp": float(8 + i) if positive else np.nan})
    pd.concat(parts).to_parquet(canonical / "measurements.parquet", index=False)
    pd.DataFrame(labels).to_parquet(canonical / "labels.parquet", index=False)
    return canonical, ids


def _run(module: str, *args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", module, *args], cwd=REPO, env=ENV,
                          capture_output=True, text=True, timeout=timeout)


def test_detector_runs_one_repeat_per_split_file_and_writes_the_shared_oof_file(tmp_path: Path):
    pytest.importorskip("torch")
    canonical, ids = _canonical(tmp_path)
    folds_a = [i % 2 for i in range(8)]                 # customers 0..7 alternate
    folds_b = [0, 0, 1, 1, 0, 0, 1, 1]
    pd.DataFrame({"customer_id": ids, "fold": folds_a}).to_parquet(tmp_path / "split_a.parquet", index=False)
    (tmp_path / "split_b.csv").write_text("customer_id;fold\n" + "".join(f"{c};{f}\n" for c, f in zip(ids, folds_b)))
    out = tmp_path / "results"
    oof = tmp_path / "oof.csv"
    result = _run("neuralwave_experiment.task1_detector", "--canonical-dir", str(canonical),
                  "--splits", f"{tmp_path / 'split_a.parquet'},{tmp_path / 'split_b.csv'}",
                  "--oof-out", str(oof), "--out-dir", str(out), "--epochs", "1", "--stride-days", "2", "--max-windows", "12")
    assert result.returncode == 0, result.stderr

    lines = oof.read_text().splitlines()
    assert lines[0] == HEADER
    assert len(lines) == 1 + 2 * len(ids)
    rows = [line.split(";") for line in lines[1:]]
    assert {r[0] for r in rows} == set(ids)
    assert sorted({int(r[1]) for r in rows}) == [0, 1]
    assert all(r[4] in {"TRUE", "FALSE"} and r[5] == "" and r[6] == "" for r in rows)
    assert all(0.0 <= float(r[3]) <= 1.0 for r in rows)
    fold_of = {(r[0], int(r[1])): int(r[2]) for r in rows}
    assert [fold_of[(c, 0)] for c in ids] == folds_a
    assert [fold_of[(c, 1)] for c in ids] == folds_b

    summary = json.loads((out / "task1_summary.json").read_text())
    assert summary["n_repeats"] == 2 and len(summary["repeats"]) == 2
    assert summary["pooled"] == summary["repeats"][0]["pooled"]
    assert (out / "task1_model_fold_0.pt").exists() and (out / "task1_model_fold_1.pt").exists()
    assert not list(out.glob("task1_model_r1_*"))

    scored = _run("neuralwave_experiment.oof_contract", "--labels", str(canonical / "labels.parquet"), str(oof))
    assert scored.returncode == 0, scored.stderr
    assert scored.stdout.startswith("file=oof.csv repeats=2 score=")
    assert "ARTIFICIAL" not in scored.stdout

    # wild inference on a fabricated unlabelled CSV with the repeat-0 checkpoint
    times = pd.date_range("2025-04-01", periods=20 * 96, freq="15min", tz="Europe/Zurich")
    wild = tmp_path / "unlabelled.csv"
    with open(wild, "w") as handle:
        handle.write("customer_id;timestamp_local;timestamp_epoch;energy_consumption_kwh\n")
        for cid, level in (("PODSYN100", 2.0), ("PODSYN101", 40.0), ("OTHER1", 1.0)):
            for t in times:
                handle.write(f"{cid};{t.strftime('%Y-%m-%dT%H:%M:%S.000')};{int(t.timestamp())};{level:.4f}\n")
    wild_out = tmp_path / "wild.json"
    result = _run("neuralwave_experiment.wild_inference", "--unlabelled-csv", str(wild),
                  "--weights", str(out / "task1_model_fold_0.pt"), "--out", str(wild_out),
                  "--id-prefix", "PODSYN", "--id-start", "100", "--id-count", "3", "--stride-days", "3")
    assert result.returncode == 0, result.stderr
    report = json.loads(wild_out.read_text())
    assert report["summary"]["requested"] == 3 and report["summary"]["scored"] == 2
    assert report["summary"]["insufficient_data"] == 1
    assert report["summary"]["pv_installed"] + report["summary"]["no_pv"] == 2
    assert "PODSYN100" not in result.stdout and "PODSYN101" not in result.stdout
    shown = _run("neuralwave_experiment.wild_inference", "--unlabelled-csv", str(wild),
                 "--weights", str(out / "task1_model_fold_0.pt"), "--out", str(wild_out),
                 "--customer-ids", "PODSYN100,PODSYN101", "--show-customers")
    assert shown.returncode == 0, shown.stderr
    assert "PODSYN100" in shown.stdout and "PODSYN101" in shown.stdout


def test_detector_refuses_a_split_file_that_misses_a_labelled_customer(tmp_path: Path):
    pytest.importorskip("torch")
    canonical, ids = _canonical(tmp_path, days=10)
    pd.DataFrame({"customer_id": ids[1:], "fold": [i % 2 for i in range(7)]}).to_parquet(tmp_path / "short.parquet", index=False)
    result = _run("neuralwave_experiment.task1_detector", "--canonical-dir", str(canonical),
                  "--splits", str(tmp_path / "short.parquet"), "--out-dir", str(tmp_path / "r"), "--epochs", "1")
    assert result.returncode != 0
    assert "does not cover labelled customers: 1 missing" in result.stderr
    assert ids[0] not in result.stderr
