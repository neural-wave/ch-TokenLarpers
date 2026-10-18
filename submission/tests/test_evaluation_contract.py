"""Shared split membership and inner calibration through the real corrected CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def _canonical(tmp_path):
    root = tmp_path / "canonical"
    root.mkdir()
    epoch = 1735689600 + np.arange(52 * 96, dtype=np.int64) * 900
    day = np.arange(len(epoch)) // 96
    hour = (np.arange(len(epoch)) % 96) / 4
    parts, labels = [], []
    for i in range(8):
        cid = f"PODSYN{i:03d}"
        values = .8 + .1 * np.cos(2 * np.pi * hour / 24)
        if i >= 4:
            values = values - .5 * np.maximum(0, np.sin(np.pi * (hour - 6) / 12)) * (day >= 12)
        parts.append(pd.DataFrame(dict(customer_id=cid, timestamp_epoch=epoch,
                                       energy_kwh=values, value_status="observed")))
        labels.append(dict(customer_id=cid, pv_installed=i >= 4,
                           pv_installation_date=pd.Timestamp("2025-01-13").date() if i >= 4 else None))
    pd.concat(parts).to_parquet(root / "measurements.parquet", index=False)
    pd.DataFrame(labels).to_parquet(root / "labels.parquet", index=False)
    return root, [row["customer_id"] for row in labels]


def test_corrected_cli_preserves_supplied_folds_and_inner_exclusion(tmp_path):
    pytest.importorskip("torch")
    canonical, ids = _canonical(tmp_path)
    maps = [dict(zip(ids, [4, 9] * 4)), dict(zip(ids, [7, 7, 11, 11] * 2))]
    paths = []
    for r, mapping in enumerate(maps):
        path = tmp_path / f"folds{r}.csv"
        pd.DataFrame(mapping.items(), columns=["customer_id", "fold"]).to_csv(path, sep=";", index=False)
        paths.append(str(path))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(dict(d_model=8, n_layers=1, n_heads=2, dropout=0.,
                                   epochs=1, pretrain_epochs=1, batch_size=32,
                                   inner_folds=2, thresholds=[.2, .5, .8])))
    out = tmp_path / "results"
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "neuralwave_experiment.evaluation",
        "--canonical-dir", str(canonical), "--out", str(out), "--config", str(cfg),
        "--data-kind", "artificial", "--splits", ",".join(paths)], cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo / "src"), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    for arm in ("hourly_logistic", "patchtst_scratch", "patchtst_pretrained"):
        rows = pd.read_csv(out / f"oof_{arm}.csv", sep=";", dtype={"customer_id": str})
        assert not rows.duplicated(["repeat", "customer_id"]).any()
        for r, mapping in enumerate(maps):
            part = rows[rows["repeat"] == r]
            assert dict(zip(part.customer_id, part.fold)) == mapping
            for fold in set(mapping.values()):
                directory = out / f"repeat_{r}" / f"fold_{fold}" / arm
                train = {c for c in ids if mapping[c] != fold}
                validation = set(ids) - train
                calibration = pd.read_csv(directory / "inner_oof.csv", sep=";")
                assert set(calibration.customer_id) == train
                assert calibration.customer_id.is_unique
                for inner in range(2):
                    fitted = set(json.loads((directory / f"inner_{inner}" / "training_ids.json").read_text()))
                    trace = pd.read_csv(directory / f"inner_{inner}" / "traces.csv", sep=";")
                    assert fitted <= train and fitted.isdisjoint(trace.customer_id)
                    assert fitted.isdisjoint(validation)
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["supplied_splits"]) == 2
    assert manifest["blind_evaluation"]["performed"] is False


def test_supplied_split_missing_customer_fails_before_training(tmp_path):
    pytest.importorskip("torch")
    from neuralwave_experiment.evaluation import run_evaluation
    canonical, ids = _canonical(tmp_path)
    path = tmp_path / "missing.csv"
    pd.DataFrame({"customer_id": ids[1:], "fold": [0, 1, 0, 1, 0, 1, 0]}).to_csv(path, sep=";", index=False)
    with pytest.raises(ValueError) as error:
        run_evaluation(canonical, tmp_path / "out", data_kind="artificial", split_files=[path])
    assert ids[0] not in str(error.value)
    assert not (tmp_path / "out").exists()
