from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from pvdetect.evaluation import outer_splits
from pvdetect.model import save_model
from pvdetect.pipeline import build_feature_table, predict
from pvdetect import procedure
from pvdetect.procedure import DEFAULT_OPTIONS, fit
from pvdetect.synthetic import generate_dataset
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels


@pytest.fixture(scope="module")
def data():
    ts, labels, _ = generate_dataset(12, seed=4)
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


def _fold_predictions(table, labels, train_ids, test_ids, options=DEFAULT_OPTIONS) -> pd.DataFrame:
    return predict(fit(table, labels, train_ids, options=options), table, test_ids)[0]


@pytest.mark.parametrize(
    "options",
    [DEFAULT_OPTIONS, DEFAULT_OPTIONS | {"schedule"}, DEFAULT_OPTIONS | {"sun"}, frozenset({"recent", "smallpv"}), frozenset()],
    ids=["default", "with-schedule", "with-sun", "recall-blocks", "no-e1e2"],
)
def test_flipping_held_out_labels_changes_nothing_in_their_fold(data, options):
    """Any cache, climatology or selection fitted on held-out labels would change these predictions."""
    table, labels = data
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    flipped = labels.copy()
    for x in test_ids:
        row = flipped["customer_id"] == x
        was_pv = bool(flipped.loc[row, "pv_installed"].iloc[0])
        flipped.loc[row, "pv_installed"] = not was_pv
        flipped.loc[row, "pv_installation_date"] = pd.NaT if was_pv else pd.Timestamp("2023-05-01")
        flipped.loc[row, "pv_capacity_kwp"] = np.nan if was_pv else 7.5
    a = _fold_predictions(table, labels, train_ids, test_ids, options)
    b = _fold_predictions(table, flipped, train_ids, test_ids, options)
    pdt.assert_frame_equal(a, b)


def test_label_flip_with_date_debias_and_cap_ceiling_forced_on(data, monkeypatch):
    """Every candidate adopted (margin and bootstrap rule disabled), so the per-regime date bias and the
    learned capacity band are both in the model; flipping held-out labels still changes nothing."""
    monkeypatch.setattr(procedure, "CLASSIFIER_MARGIN", -np.inf)
    monkeypatch.setattr(procedure, "BOOTSTRAP_MIN", 0.0)
    table, labels = data
    options = frozenset({"date_debias", "cap_ceiling"})
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    model = fit(table, labels, train_ids, options=options)
    assert model["selection"]["date_debias"] == model["selection"]["cap_ceiling"] == "on"
    assert "date_bias" in model["date"] and "band" in model["capacity"]
    flipped = labels.copy()
    for x in test_ids:
        row = flipped["customer_id"] == x
        was_pv = bool(flipped.loc[row, "pv_installed"].iloc[0])
        flipped.loc[row, "pv_installed"] = not was_pv
        flipped.loc[row, "pv_installation_date"] = pd.NaT if was_pv else pd.Timestamp("2023-05-01")
        flipped.loc[row, "pv_capacity_kwp"] = np.nan if was_pv else 7.5
    a = _fold_predictions(table, labels, train_ids, test_ids, options)
    b = _fold_predictions(table, flipped, train_ids, test_ids, options)
    pdt.assert_frame_equal(a, b)


@pytest.fixture(scope="module")
def shared_weather_data():
    ts, labels, _ = generate_dataset(36, 5, scenario="shared_weather")
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


def _flip_test_ids(labels: pd.DataFrame, test_ids: list) -> pd.DataFrame:
    flipped = labels.copy()
    for x in test_ids:
        row = flipped["customer_id"] == x
        was_pv = bool(flipped.loc[row, "pv_installed"].iloc[0])
        flipped.loc[row, "pv_installed"] = not was_pv
        flipped.loc[row, "pv_installation_date"] = pd.NaT if was_pv else pd.Timestamp("2023-05-01")
        flipped.loc[row, "pv_capacity_kwp"] = np.nan if was_pv else 7.5
    return flipped


def test_label_flip_with_schedule_forced_on(data, monkeypatch):
    """Schedule feature block adopted (margin disabled); held-out label flip changes nothing.
    Verifies the schedule feature is label-free and fitted only on training customers."""
    monkeypatch.setattr(procedure, "CLASSIFIER_MARGIN", -np.inf)
    monkeypatch.setattr(procedure, "BOOTSTRAP_MIN", 0.0)
    table, labels = data
    options = frozenset({"schedule"})
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    model = fit(table, labels, train_ids, options=options)
    assert model["selection"]["schedule_block"] == "on"
    flipped = _flip_test_ids(labels, test_ids)
    a = _fold_predictions(table, labels, train_ids, test_ids, options)
    b = _fold_predictions(table, flipped, train_ids, test_ids, options)
    pdt.assert_frame_equal(a, b)


def test_label_flip_with_sun_forced_on(shared_weather_data, monkeypatch):
    """Sun block adopted (margin disabled); held-out label flip changes nothing.
    Verifies the sun index is fitted only on training PV customers, not on held-out labels."""
    monkeypatch.setattr(procedure, "CLASSIFIER_MARGIN", -np.inf)
    monkeypatch.setattr(procedure, "BOOTSTRAP_MIN", 0.0)
    table, labels = shared_weather_data
    options = frozenset({"sun"})
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    model = fit(table, labels, train_ids, options=options)
    assert model["selection"]["sun_block"] == "on"
    flipped = _flip_test_ids(labels, test_ids)
    a = _fold_predictions(table, labels, train_ids, test_ids, options)
    b = _fold_predictions(table, flipped, train_ids, test_ids, options)
    pdt.assert_frame_equal(a, b)


def test_fit_is_deterministic(data, tmp_path):
    table, labels = data
    ids = labels["customer_id"].tolist()
    save_model(fit(table, labels, ids), tmp_path / "a" / "model.json")
    save_model(fit(table, labels, ids), tmp_path / "b" / "model.json")
    assert json.loads((tmp_path / "a" / "model.json").read_text()) == json.loads((tmp_path / "b" / "model.json").read_text())


def test_label_flip_with_onset_first_and_tail_guard_forced_on(data, monkeypatch):
    """onset_first and tail_guard forced on (margin/bootstrap disabled); flipping held-out labels changes nothing."""
    monkeypatch.setattr(procedure, "CLASSIFIER_MARGIN", -np.inf)
    monkeypatch.setattr(procedure, "BOOTSTRAP_MIN", 0.0)
    table, labels = data
    options = frozenset({"onset_first", "tail_guard"})
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    model = fit(table, labels, train_ids, options=options)
    assert model["selection"]["onset_first"] == "on"
    assert model["selection"]["tail_guard"] == "on"
    flipped = labels.copy()
    for x in test_ids:
        row = flipped["customer_id"] == x
        was_pv = bool(flipped.loc[row, "pv_installed"].iloc[0])
        flipped.loc[row, "pv_installed"] = not was_pv
        flipped.loc[row, "pv_installation_date"] = pd.NaT if was_pv else pd.Timestamp("2023-05-01")
        flipped.loc[row, "pv_capacity_kwp"] = np.nan if was_pv else 7.5
    a = _fold_predictions(table, labels, train_ids, test_ids, options)
    b = _fold_predictions(table, flipped, train_ids, test_ids, options)
    pdt.assert_frame_equal(a, b)


def test_label_flip_with_tobit_forced_on(data, monkeypatch):
    """`tobit` forced on (margin/bootstrap disabled): `log_kwp_tobit` uses the fit's onset and yield shape, so it is
    computed inside fit/predict; flipping held-out labels changes nothing in their fold."""
    monkeypatch.setattr(procedure, "CLASSIFIER_MARGIN", -np.inf)
    monkeypatch.setattr(procedure, "BOOTSTRAP_MIN", 0.0)
    table, labels = data
    options = frozenset({"tobit"})
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    model = fit(table, labels, train_ids, options=options)
    assert model["selection"]["tobit"] == "on"
    assert "log_kwp_tobit" in model["capacity_feature_names"]
    a = _fold_predictions(table, labels, train_ids, test_ids, options)
    b = _fold_predictions(table, _flip_test_ids(labels, test_ids), train_ids, test_ids, options)
    pdt.assert_frame_equal(a, b)


def test_models_without_tobit_predict_unchanged(data, monkeypatch):
    """A model fitted without `tobit` (as every older model) never reads `log_kwp_tobit`: its predictions are the
    same whether or not the feature is computed."""
    from pvdetect import features

    table, labels = data
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    model = fit(table, labels, train_ids, options=DEFAULT_OPTIONS - {"tobit"})
    assert "tobit" not in model["selection"] and "log_kwp_tobit" not in model["capacity_feature_names"]
    a = predict(model, table, test_ids)[0]
    monkeypatch.setattr(features, "tobit_features", lambda *args, **kw: {"log_kwp_tobit": np.nan})
    b = predict(model, table, test_ids)[0]
    pdt.assert_frame_equal(a.drop(columns="log_kwp_tobit"), b.drop(columns="log_kwp_tobit"))


def test_label_flip_with_avg_cnn_forced_on(data, monkeypatch):
    """With `fraunhofer` in the pool and every candidate adopted (margin and bootstrap disabled), the last
    candidate, `avg_cnn`, is the classifier; flipping held-out labels changes no prediction in their fold."""
    torch = pytest.importorskip("torch")
    from pvdetect import fraunhofer as fh

    train = fh.train_fraunhofer
    # one epoch per stage on CPU: the CNN's fold safety is the subject, not its accuracy
    monkeypatch.setattr(fh, "train_fraunhofer", lambda ds, *a, **k: train(ds, epochs=(1, 1, 1), device="cpu", seed=0))
    monkeypatch.setattr(procedure, "CLASSIFIER_MARGIN", -np.inf)
    monkeypatch.setattr(procedure, "BOOTSTRAP_MIN", 0.0)
    table, labels = data
    options = frozenset({"fraunhofer"})
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = fit(table, labels, train_ids, options=options)
        assert model["classifier"]["kind"] == "avg_cnn"
        a = predict(model, table, test_ids)[0]
        b = predict(fit(table, _flip_test_ids(labels, test_ids), train_ids, options=options), table, test_ids)[0]
    finally:
        torch.set_num_threads(old_threads)
    pdt.assert_frame_equal(a, b)
