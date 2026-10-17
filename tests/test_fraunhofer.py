"""CNN contracts on fabricated customers; no accuracy claims."""
from functools import partial

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from pvdetect import fraunhofer as fh
from pvdetect.model import CURRENT_VERSION, load_model, save_model
from pvdetect.pipeline import build_feature_table
from pvdetect.procedure import _select
from pvdetect.synthetic import CustomerSpec, PVSpec, generate_customer
from tests.helpers import as_loaded
from tests.test_evaluation import _truth8
from tests.test_precision import _oof


@pytest.fixture(scope="module")
def cnn_data():
    specs = [
        CustomerSpec(f"{i:06d}", "2024-01-01", "2024-02-25", missing=False,
                     pv=PVSpec("2024-01-15", 5.0) if i % 2 else None)
        for i in range(4)
    ]
    ts = pd.concat([generate_customer(s, seed=80 + i) for i, s in enumerate(specs)])
    labels = pd.DataFrame({
        "customer_id": [s.customer_id for s in specs],
        "pv_installed": [s.pv is not None for s in specs],
        "pv_installation_date": pd.to_datetime([s.pv.date if s.pv else None for s in specs]),
        "pv_capacity_kwp": [5.0 if s.pv else np.nan for s in specs],
    })
    return build_feature_table(as_loaded(ts)), labels


def test_undated_positive_is_not_negative_training_data(cnn_data):
    table, labels = cnn_data
    positive = labels.loc[labels.pv_installed, "customer_id"].iloc[0]
    undated = labels.assign(pv_installation_date=pd.NaT)
    ds = fh.build_window_dataset(table.customers, undated, [positive])
    assert ds.labels.size == 0
    with pytest.raises(ValueError, match="at least one labelled"):
        fh.train_fraunhofer(ds, device="cpu")


def test_install_week_is_dropped_and_other_windows_have_correct_labels(cnn_data):
    table, labels = cnn_data
    positive = labels.loc[labels.pv_installed, "customer_id"].iloc[0]
    labels = labels.copy()
    labels.loc[labels.customer_id == positive, "pv_installation_date"] = pd.Timestamp("2024-01-17")
    ds = fh.build_window_dataset(table.customers, labels, [positive])
    starts = ds.week_start_dates.astype("datetime64[D]")
    assert np.datetime64("2024-01-15") not in starts
    np.testing.assert_array_equal(ds.labels, (starts >= np.datetime64("2024-01-17")).astype(float))
    assert ds.labels.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_inner_fold_scores_do_not_use_validation_labels(cnn_data, monkeypatch):
    table, labels = cnn_data
    ids = table.ids
    splits = [(ids[2:], ids[:2]), (ids[:2], ids[2:])]
    monkeypatch.setattr(fh, "train_fraunhofer", partial(fh.train_fraunhofer, epochs=(1, 1, 1), device="cpu"))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        original = fh.inner_customer_scores(table, labels, ids, splits)
        flipped = labels.copy()
        heldout = flipped.customer_id.isin(ids[:2])
        flipped.loc[heldout, "pv_installed"] = ~flipped.loc[heldout, "pv_installed"]
        flipped.loc[heldout, "pv_installation_date"] = pd.Timestamp("2024-01-01")
        changed = fh.inner_customer_scores(table, flipped, ids, splits)
    finally:
        torch.set_num_threads(old_threads)
    np.testing.assert_array_equal(original[:2], changed[:2])
    with pytest.raises(ValueError, match="same customer"):
        fh.inner_customer_scores(table, labels, ids, [(ids, ids)])


def test_cnn_selection_requires_gain_and_learns_threshold():
    truth = _truth8().set_index("customer_id")
    y = truth.pv_installed.to_numpy()
    weak = np.full(len(truth), 0.5)
    oof = _oof(truth, weak, weak)
    oof["p_fraunhofer"] = np.where(y, 0.8, 0.6)
    selected = _select(oof)
    assert selected["classifier"] == "fraunhofer"
    assert 0.6 < selected["threshold"] <= 0.8
    oof["p_fraunhofer"] = weak
    assert _select(oof)["classifier"] == "logreg"


def test_cnn_artifact_roundtrip_preserves_customer_scores(cnn_data, tmp_path):
    table, _ = cnn_data
    torch.manual_seed(4)
    net = fh.CombinedClassifier(fh.MLPBranch(), fh.CNNBranch()).eval()
    expected = fh.fraunhofer_customer_scores(net, table, table.ids)
    path = tmp_path / "model.json"
    save_model({"version": CURRENT_VERSION, "classifier": {"kind": "fraunhofer", "fraunhofer_file": "fraunhofer.pt"},
                "_objects": {"fraunhofer": net}}, path)
    loaded = load_model(path)
    actual = fh.fraunhofer_customer_scores(loaded["_objects"]["fraunhofer"], table, table.ids)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
