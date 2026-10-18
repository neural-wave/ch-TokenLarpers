"""CNN contracts on fabricated customers; no accuracy claims."""
from functools import partial
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from pvdetect import fraunhofer as fh
from pvdetect import procedure
from pvdetect.evaluation import outer_splits
from pvdetect.model import CURRENT_VERSION, linear_score, load_model, referenced_files, save_model, sigmoid
from pvdetect.pipeline import build_feature_table, classifier_proba, compute_features, predict
from pvdetect.procedure import _select, fit
from pvdetect.season import Climatology
from pvdetect.synthetic import CustomerSpec, PVSpec, generate_customer
from tests.helpers import as_loaded
from tests.test_evaluation import _truth8
from tests.test_precision import _oof, _split_probs, _truth


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


@pytest.fixture(scope="module")
def cnn8():
    """8 fabricated customers, half with PV installed mid-series."""
    specs = [
        CustomerSpec(f"{i:06d}", "2024-01-01", "2024-03-10", missing=False,
                     pv=PVSpec("2024-02-05", 5.0) if i % 2 else None)
        for i in range(8)
    ]
    ts = pd.concat([generate_customer(s, seed=90 + i) for i, s in enumerate(specs)])
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


# ------------------------------------------------------------- avg_cnn kind


def _fast_cnn(monkeypatch):
    """One epoch per stage on CPU, whatever the caller asks for; the CNN is not the subject here."""
    train = fh.train_fraunhofer

    def fast(ds, *args, **kwargs):
        return train(ds, epochs=(1, 1, 1), batch_size=64, lr=1e-3, device="cpu", seed=0)

    monkeypatch.setattr(fh, "train_fraunhofer", fast)


def test_avg_cnn_is_the_mean_with_the_pre_cnn_classifier_and_learns_its_threshold():
    """Logreg misranks the first 3 PV and non-PV customers, the CNN the next 3: neither alone clears the
    margin over the other, their mean separates everyone."""
    t = _truth()
    t_pv = t["pv_installed"].to_numpy(bool)
    p_lr, p_cnn = _split_probs(t_pv, 3)
    oof = _oof(t, p_lr, p_lr)
    oof["p_fraunhofer"] = p_cnn
    expected = np.where(t_pv, 0.95, 0.05)
    pos, neg = np.flatnonzero(t_pv), np.flatnonzero(~t_pv)
    expected[pos[:6]] = (0.30 + 0.95) / 2
    expected[neg[:6]] = (0.70 + 0.05) / 2
    np.testing.assert_array_equal(procedure._proba(oof, "avg_cnn", "logreg"), expected)
    sel = _select(oof)
    assert sel["classifier"] == "avg_cnn" and sel["classifier_base"] == "logreg"
    assert sel["threshold"] == 0.5  # 0.375 < t <= 0.625 all separate; ties go to the one nearest 0.5
    assert sel["inner_score_avg_cnn"] == pytest.approx(100.0)
    assert sel["inner_score_fraunhofer"] == pytest.approx(sel["inner_score_logreg"])
    assert sel["inner_score_logreg"] < 100.0
    assert sel["bootstrap_p_fraunhofer"] is None
    assert sel["bootstrap_p_avg_cnn"] >= procedure.BOOTSTRAP_MIN
    np.testing.assert_array_equal(procedure._sel_proba(oof, sel), expected)


def test_avg_cnn_needs_the_margin_over_the_best_other_kind():
    t = _truth()
    t_pv = t["pv_installed"].to_numpy(bool)
    p_lr, _ = _split_probs(t_pv, 3, flawed=("logreg",))
    oof = _oof(t, p_lr, p_lr)
    oof["p_fraunhofer"] = np.where(t_pv, 0.95, 0.05)  # perfect CNN: adopted alone, the average adds nothing
    sel = _select(oof)
    assert sel["classifier"] == "fraunhofer" and "classifier_base" not in sel
    assert sel["bootstrap_p_avg_cnn"] is None
    del oof["p_fraunhofer"]
    assert "inner_score_avg_cnn" not in _select(oof)


def test_fit_with_avg_cnn_saves_both_components_and_predicts_their_mean(cnn8, monkeypatch, tmp_path):
    table, labels = cnn8
    _fast_cnn(monkeypatch)
    # 8 customers cannot give a bootstrap of 0.8; the margin still decides (the full rule is tested above)
    monkeypatch.setattr(procedure, "BOOTSTRAP_MIN", 0.0)

    def stub(table, labels, lab, train_ids, splits, names, cap_names, setting=procedure.BASE_SETTING, **_):
        t = lab.loc[train_ids]
        return _oof(t, *(2 * [_split_probs(t["pv_installed"].to_numpy(bool), 1)[0]]))

    def cnn_stub(table, labels, ids, splits):
        t_pv = labels.set_index("customer_id").loc[list(ids), "pv_installed"].to_numpy(bool)
        return _split_probs(t_pv, 1)[1]

    monkeypatch.setattr(procedure, "_inner_oof", stub)
    monkeypatch.setattr(fh, "inner_customer_scores", cnn_stub)
    ids = labels["customer_id"].tolist()
    model = fit(table, labels, ids, options=frozenset({"fraunhofer"}))
    clf = model["classifier"]
    assert clf["kind"] == "avg_cnn" and clf["base_kind"] == "logreg"
    assert clf["logreg"] is not None and clf["fraunhofer_file"] == "fraunhofer.pt" and clf["catboost_file"] is None
    assert model["selection"]["classifier_base"] == "logreg"
    save_model(model, tmp_path / "model.json")
    loaded = load_model(tmp_path / "model.json")
    assert referenced_files(loaded) == ["fraunhofer.pt"]
    before, _ = predict(model, table, ids)
    after, _ = predict(loaded, table, ids)
    pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=1e-5, atol=1e-6)
    F, _ = compute_features(table, ids, Climatology.from_json(loaded["climatology"]))
    p_lr = sigmoid(linear_score(clf["logreg"], F[loaded["feature_names"]].to_numpy(float)))
    p_cnn = fh.fraunhofer_customer_scores(loaded["_objects"]["fraunhofer"], table, ids)
    np.testing.assert_allclose(after["prob"].to_numpy(float), (p_lr + p_cnn) / 2, rtol=0, atol=1e-6)


def test_models_without_base_kind_predict_as_before():
    """Old (pre-`avg_cnn`) classifier sections carry no `base_kind` and are untouched by the new branch."""
    F = pd.DataFrame({"a": [0.0, 1.0, -2.0, 3.0]})
    clf = {"kind": "logreg", "logreg": {"impute": [0.0], "mean": [0.0], "scale": [1.0], "coef": [0.5], "intercept": 0.1}}
    fh_scores = np.array([0.2, 0.4, 0.6, 0.8])
    p_lr = classifier_proba(clf, {}, F, ["a"])
    np.testing.assert_allclose(p_lr, sigmoid(linear_score(clf["logreg"], F[["a"]].to_numpy(float))))
    np.testing.assert_array_equal(classifier_proba({"kind": "fraunhofer"}, {}, F, ["a"], fraunhofer=fh_scores), fh_scores)
    both = classifier_proba({**clf, "kind": "avg_cnn", "base_kind": "logreg"}, {}, F, ["a"], fraunhofer=fh_scores)
    np.testing.assert_allclose(both, (p_lr + fh_scores) / 2, rtol=0, atol=1e-15)

