"""CNN contracts on fabricated customers; no accuracy claims."""
from functools import partial
from dataclasses import replace
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

def test_kde_and_cnn_inputs_are_scale_invariant_for_commercial_load():
    spec = CustomerSpec("900000", "2024-01-01", "2024-03-01",
                        profile="commercial", missing=False,
                        pv=PVSpec("2024-01-15", 30.0))
    commercial = generate_customer(spec, seed=901)
    commercial_table = build_feature_table(as_loaded(commercial))
    cid = spec.customer_id
    original = fh.build_window_dataset(commercial_table.customers, pd.DataFrame({
        "customer_id": [cid],
        "pv_installed": [True],
        "pv_installation_date": [pd.Timestamp("2024-01-15")],
    }), ids=[cid])
    scaled_customer = replace(
        commercial_table.customers[cid],
        M=commercial_table.customers[cid].M * 37.0,
    )
    scaled_table = replace(
        commercial_table,
        customers={cid: scaled_customer},
    )
    scaled = fh.build_window_dataset(scaled_table.customers, pd.DataFrame({
        "customer_id": [cid],
        "pv_installed": [True],
        "pv_installation_date": [pd.Timestamp("2024-01-15")],
    }), ids=[cid])
    np.testing.assert_allclose(scaled.series, original.series, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scaled.kdes, original.kdes, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        original.kdes,
        fh.fast_kde_batch(original.series[:, 0, :]),
        rtol=1e-6,
        atol=1e-6,
    )


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


def test_persistent_evidence_survives_long_preinstallation_history():
    dates = np.datetime64("2022-01-03") + np.arange(156) * np.timedelta64(7, "D")
    probabilities = np.r_[np.full(152, 0.05), [0.92, 0.85, 0.96, 0.88]]
    assert fh._persistent_week_score(probabilities, dates) == pytest.approx(0.85)
    assert fh._persistent_week_score(probabilities[-4:], dates[-4:]) == pytest.approx(0.85)


@pytest.mark.parametrize("high_weeks", [[], [0], [0, 1, 2], [0, 2, 4, 6]])
def test_isolated_or_short_bursts_do_not_establish_persistence(high_weeks):
    dates = np.datetime64("2024-01-01") + np.arange(8) * np.timedelta64(7, "D")
    probabilities = np.full(8, 0.1)
    probabilities[high_weeks] = 0.99
    assert fh._persistent_week_score(probabilities, dates) == pytest.approx(0.1)


def test_missing_week_breaks_run_and_dates_are_ordered():
    dates = np.datetime64("2024-01-01") + np.array([0, 1, 3, 4]) * np.timedelta64(7, "D")
    assert fh._persistent_week_score(np.full(4, 0.9), dates) == 0.0
    assert fh._persistent_week_score(np.full(3, 0.9), dates[:3]) == 0.0
    dates = np.datetime64("2024-01-01") + np.arange(4) * np.timedelta64(7, "D")
    order = [2, 0, 3, 1]
    assert fh._persistent_week_score(np.array([0.9, 0.8, 0.7, 0.6])[order], dates[order]) == pytest.approx(0.6)


def test_customer_weights_equalize_history_mass_and_weight_class_balance():
    owners = np.array(["a", "a"] + ["b"] * 6)
    labels = np.array([0., 1.] + [1.] * 6, dtype=np.float32)
    weights, positive_weight = fh._training_weights(owners, labels)
    assert weights[owners == "a"].sum() == pytest.approx(4.)
    assert weights[owners == "b"].sum() == pytest.approx(4.)
    assert positive_weight == pytest.approx(1. / 3.)


def test_repeating_one_customer_history_preserves_loss_and_gradient():
    owners = np.array(["a", "a", "b", "b"])
    labels = np.array([0., 1., 0., 0.], dtype=np.float32)
    features = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]])

    def objective(indices):
        weights, positive_weight = fh._training_weights(owners[indices], labels[indices])
        coefficient = torch.tensor([0.3, -0.2], requires_grad=True)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            features[indices] @ coefficient, torch.tensor(labels[indices]),
            pos_weight=torch.tensor(positive_weight), reduction="none",
        )
        loss = (losses * torch.tensor(weights)).mean()
        loss.backward()
        return loss.detach(), coefficient.grad

    loss, gradient = objective(np.arange(4))
    repeated_loss, repeated_gradient = objective(np.array([0, 1] + [2, 3] * 9))
    torch.testing.assert_close(repeated_loss, loss)
    torch.testing.assert_close(repeated_gradient, gradient)
