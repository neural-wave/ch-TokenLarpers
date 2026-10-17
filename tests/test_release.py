from __future__ import annotations

from pathlib import Path

import pvdetect.procedure as procedure
from pvdetect.model import referenced_files, save_model
from pvdetect.pipeline import build_feature_table
from pvdetect.synthetic import generate_dataset
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels


def _model_relabelled_data():
    ts, labels, _ = generate_dataset(12, seed=5)
    mapping = {cid: f"SYNREL-{i:03d}" for i, cid in enumerate(sorted(labels["customer_id"]))}
    ts = ts.assign(customer_id=ts["customer_id"].map(mapping))
    labels = labels.assign(customer_id=labels["customer_id"].map(mapping))
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


def _model_saved_files(model: dict, path: Path) -> list[Path]:
    save_model(model, path)
    files = [path] + [path.parent / f for f in referenced_files(model)]
    assert all(f.exists() for f in files)
    return files


def _model_leaked_ids(files: list[Path], ids: list[str]) -> list[tuple[str, str]]:
    return [(f.name, cid) for f in files for cid in ids if cid.encode() in f.read_bytes()]


def test_saved_model_files_carry_no_training_customer_ids(tmp_path, monkeypatch):
    """No saved model file (`model.json` or any referenced `.cbm`) contains a training customer_id.

    Pitfalls handled:
    1. Synthetic ids are zero-padded digits (`000003`) that can occur inside JSON floats, so
       customers are relabelled consistently in time series and labels with distinctive ids
       (`SYNREL-007`) before the table is built; a hit is then a real leak, not a digit run.
    2. A 12-customer fit usually selects logreg/ridge and writes no `.cbm`, so a second fit
       forces CatBoost classifier and capacity through the real `fit`/`save_model` path
       (only the inner selection is replaced) and asserts both `.cbm` files exist.

    This check is necessary but not sufficient: Step 2.5-G adds human review and organizer approval.
    """
    table, labels = _model_relabelled_data()
    ids = labels["customer_id"].tolist()
    assert all(cid.startswith("SYNREL-") for cid in ids)

    selected = _model_saved_files(procedure.fit(table, labels, ids), tmp_path / "selected" / "model.json")
    assert _model_leaked_ids(selected, ids) == []

    forced_sel = {"classifier": "catboost", "threshold": 0.5, "capacity": "catboost", "kappa": 1.0, "insufficient_rule": "weak_candidate"}
    monkeypatch.setattr(procedure, "_inner_oof", lambda *args: None)
    monkeypatch.setattr(procedure, "_select", lambda oof: dict(forced_sel))
    forced = procedure.fit(table, labels, ids)
    assert forced["classifier"]["kind"] == "catboost" and forced["capacity"]["kind"] == "catboost"
    files = _model_saved_files(forced, tmp_path / "catboost" / "model.json")
    assert sorted(f.name for f in files if f.suffix == ".cbm") == ["catboost_cap.cbm", "catboost_clf.cbm"]
    assert _model_leaked_ids(files, ids) == []
