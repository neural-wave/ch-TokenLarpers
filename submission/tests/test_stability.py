from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pvdetect import stability
from pvdetect.io import load_timeseries
from pvdetect.model import load_model, save_model
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.procedure import fit
from pvdetect.synthetic import generate_dataset
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels


@pytest.fixture(scope="module")
def models_and_pool(tmp_path_factory) -> tuple[Path, Path, Path]:
    """Model A fitted on 12 dev-seed customers; model B = A with threshold 0 (every customer PV);
    6 PODSYN unlabelled customers over two years."""
    root = tmp_path_factory.mktemp("stability")
    ts, labels, _ = generate_dataset(12, seed=4)
    labels = typed_labels(labels)
    table = build_feature_table(as_loaded(ts))
    model = fit(table, labels, labels["customer_id"].tolist(), options=frozenset())
    save_model(model, root / "a" / "model.json")
    save_model({**model, "classifier": {**model["classifier"], "threshold": 0.0}}, root / "b" / "model.json")
    unl, _, _ = generate_dataset(6, seed=8, start="2023-01-01", end="2024-12-31")
    unl["customer_id"] = "PODSYN" + unl["customer_id"].astype(str)
    unl.to_csv(root / "unl.csv", index=False, float_format="%.4f")
    return root / "a" / "model.json", root / "b" / "model.json", root / "unl.csv"


def test_same_model_is_perfectly_stable(models_and_pool, capsys):
    a, _, unl = models_and_pool
    assert stability.main([str(a), str(a), str(unl), "--chunksize", "20000"]) == 0
    out = capsys.readouterr().out
    assert "pool customers=6 max=300" in out
    assert "flips=0 share=0.000 off->on=0 on->off=0 mean_abs_dprob=0.0000" in out
    assert "PODSYN" not in out


def test_flips_and_shares_match_direct_predictions(models_and_pool, capsys):
    a, b, unl = models_and_pool
    table = build_feature_table(load_timeseries(unl))
    pa, _ = predict(load_model(a), table, table.ids)
    off = int((~pa["pv_installed"].to_numpy(bool)).sum())
    assert 0 < off < 6
    assert stability.main([str(a), str(b), str(unl), "--pool", "4"]) == 0
    out = capsys.readouterr().out
    assert "pool customers=4 max=4" in out and "B=1.000" in out
    assert stability.main([str(a), str(b), str(unl)]) == 0
    out = capsys.readouterr().out
    share = np.mean(pa["pv_installed"].to_numpy(bool))
    assert f"predicted_pv_share A={share:.3f} B=1.000" in out
    assert f"flips={off} share={off / 6:.3f} off->on={off} on->off=0 mean_abs_dprob=0.0000" in out


def test_empty_pool_is_reported(models_and_pool, capsys):
    a, b, unl = models_and_pool
    assert stability.main([str(a), str(b), str(unl), "--id-prefix", "NOPE"]) == 1
    assert "empty pool" in capsys.readouterr().out
