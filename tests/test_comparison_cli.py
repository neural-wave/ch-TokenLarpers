"""Exercise both actual training paths; no downloaded or organizer inputs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_two_backends_preserve_customer_folds_and_baseline(tmp_path):
    pytest.importorskip("torch")
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    times = pd.date_range("2025-01-01", periods=16 * 96, freq="15min", tz="Europe/Zurich")
    parts, labels, splits = [], [], []
    for i in range(8):
        customer = f"ARTIFICIAL_{i:02}"
        positive = i >= 4
        onset = pd.Timestamp("2025-01-09").date()
        hour = times.hour.to_numpy() + times.minute.to_numpy() / 60
        load = 3 + .1 * i + .2 * np.cos(hour * np.pi / 12)
        effect = np.maximum(0, np.sin((hour - 6) * np.pi / 12))
        active = np.array([d >= onset for d in times.date]) & positive
        values = load - effect * active
        parts.append(pd.DataFrame({"customer_id": customer,
            "timestamp_epoch": times.as_unit("s").asi8,
            "timestamp_local": times, "local_date": times.date,
            "local_slot": (times.hour * 4 + times.minute // 15).astype("uint8"),
            "dst_fold": 0, "utc_offset_minutes": 60,
            "energy_kwh": values, "value_status": "observed"}))
        labels.append({"customer_id": customer, "pv_installed": positive,
            "pv_installation_date": onset if positive else None,
            "pv_capacity_kwp": float(8 + i) if positive else np.nan})
        splits.append({"customer_id": customer, "fold": i % 2})
    pd.concat(parts).to_parquet(canonical / "measurements.parquet", index=False)
    pd.DataFrame(labels).to_parquet(canonical / "labels.parquet", index=False)
    pd.DataFrame(splits).to_parquet(canonical / "splits.parquet", index=False)
    (canonical / "run_manifest.json").write_text(json.dumps({"schema_version": 1}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"context_length": 96, "stride": 96,
        "patch_length": 16, "d_model": 8, "n_heads": 2, "n_layers": 1,
        "dropout": 0, "pretrain_epochs": 1, "finetune_epochs": 1,
        "max_windows_per_customer": 4, "batch_size": 8}))
    out = tmp_path / "comparison"
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "neuralwave_experiment.comparison",
        "--canonical-dir", str(canonical), "--out", str(out),
        "--config", str(config), "--seed", "17"], cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo / "src"),
             "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "comparison.json").read_text())
    assert report["baseline_metrics_equal"] is True
    expected = {row["customer_id"] for row in labels}
    for backend in ("custom", "official_patchtst"):
        audit = report["backends"][backend]["lineage_audit"]
        for fold in audit:
            assert set(fold["training_customers"]).isdisjoint(fold["heldout_customers"])
        for arm in ("baseline", "frozen", "finetuned"):
            predictions = pd.read_csv(out / backend / f"predictions_{arm}.csv", sep=";")
            assert set(predictions.customer_id) == expected
            assert not predictions.customer_id.duplicated().any()
            negative = ~predictions.pv_installed
            assert predictions.loc[negative, "predicted_installation_date"].isna().all()
            assert predictions.loc[negative, "predicted_capacity_kwp"].isna().all()
