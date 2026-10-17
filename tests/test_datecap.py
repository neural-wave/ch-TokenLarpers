"""`date_debias` (per-regime residual date bias from the fit's inner OOF true positives) and `cap_ceiling`
(learned capacity band with the training-median fallback): exact unit checks, inner-CV adoption, inference
and backward compatibility, and Tier R checks on the dev scenarios that reproduce the failure modes."""

from __future__ import annotations

import copy
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pvdetect import procedure
from pvdetect.changepoint import Commissioning
from pvdetect.evaluation import outer_splits
from pvdetect.model import load_model, save_model
from pvdetect.pipeline import apply_cap_band, build_feature_table, predict, regime_date
from pvdetect.procedure import (
    _band_oof,
    _choose_cap_ceiling,
    _choose_date_debias,
    _debias_oof,
    _full_inner_score,
    _select,
    fit,
    fit_cap_band,
    regime_bias,
)
from pvdetect.synthetic import generate_dataset
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels

T0 = pd.Timestamp("2023-05-01")
SEL = {"classifier": "logreg", "threshold": 0.5, "insufficient_rule": "weak_candidate"}


# ----------------------------------------------------------------- unit


def test_regime_bias_median_min_n_and_clip():
    err = np.array([10, 20, 30, 40, -5, 100, np.nan])
    reg = np.array(["event"] * 4 + ["present_at_start"] * 2 + ["insufficient"])
    assert regime_bias(err, reg, np.ones(7, bool)) == {"event": 25, "present_at_start": 0, "insufficient": 0}
    mask = np.array([True, True, True, False, True, True, True])
    assert regime_bias(err, reg, mask)["event"] == 20
    assert regime_bias(np.array([500.0, 600, 700]), np.array(["event"] * 3), np.ones(3, bool))["event"] == 365


def test_fit_cap_band_exact():
    band = fit_cap_band(np.array([2.0, 4, 6, 8, 10]))
    assert band["lo"] == pytest.approx(2.4 / 1.5)
    assert band["hi"] == pytest.approx(9.6 * 1.5)
    assert band["fallback"] == 6.0
    assert fit_cap_band(np.array([np.nan, -1.0, 3.0, 0.0])) is None


def test_apply_cap_band():
    band = {"lo": 1.6, "hi": 14.4, "fallback": 6.0}
    np.testing.assert_array_equal(apply_cap_band(np.array([0.5, 3.0, 500.0]), band), [6.0, 3.0, 6.0])
    np.testing.assert_array_equal(apply_cap_band(np.array([0.5, 500.0]), None), [0.5, 500.0])


def _oof(errors, folds, *, kwp=None, true_kwp=6.0, regime="event", band=(2.0, 15.0, 6.0), n_neg=0) -> pd.DataFrame:
    """Hand-built inner OOF: one TP per error (predicted − labelled days), then `n_neg` true negatives."""
    n = len(errors)
    ids = [f"C{i:02d}" for i in range(n + n_neg)]
    t_date = [T0 + pd.Timedelta(days=7 * i) for i in range(n)] + [pd.NaT] * n_neg
    pred = [t + pd.Timedelta(days=int(e)) for t, e in zip(t_date, errors)] + [None] * n_neg
    k = list(kwp if kwp is not None else [true_kwp] * n) + [5.0] * n_neg
    return pd.DataFrame(
        {
            "p_logreg": [1.0] * n + [0.0] * n_neg,
            "p_catboost": [1.0] * n + [0.0] * n_neg,
            "kwp_ridge": k,
            "kwp_catboost": np.nan,
            "regime": regime,
            "date_weak": pred,
            "date_first": pred,
            "fold": list(folds) + [i % 3 for i in range(n_neg)],
            "cap_lo": band[0],
            "cap_hi": band[1],
            "cap_fb": band[2],
            "t_pv": [True] * n + [False] * n_neg,
            "t_date": t_date,
            "t_kwp": [true_kwp] * n + [np.nan] * n_neg,
        },
        index=pd.Index(ids, name="customer_id"),
    )


def test_debias_is_cross_fitted_and_final_bias_uses_all_rows():
    oof = _oof([10, 10, 10, 40, 40, 40], [0, 0, 0, 1, 1, 1])
    out, bias = _debias_oof(oof, SEL)
    shift = (pd.to_datetime(oof["date_weak"]) - out["date_weak"]).dt.days.tolist()
    assert shift == [40, 40, 40, 10, 10, 10]  # each fold by the other fold's bias
    assert bias == {"event": 25, "present_at_start": 0, "insufficient": 0}
    moved = oof.copy()
    moved["t_date"] = moved["t_date"].where(moved["fold"] == 1, moved["t_date"] + pd.Timedelta(days=90))
    pd.testing.assert_series_equal(_debias_oof(moved, SEL)[0]["date_weak"][:3], out["date_weak"][:3])


def test_band_oof_applies_each_folds_band_and_skips_folds_without_one():
    oof = _oof([0, 0, 0], [0, 1, 2], kwp=[0.5, 7.0, 500.0])
    oof.loc["C02", ["cap_lo", "cap_hi", "cap_fb"]] = np.nan
    assert _band_oof(oof)["kwp_ridge"].tolist() == [6.0, 7.0, 500.0]


def _scored(oof):
    sel = _select(oof)
    sel["inner_score"] = _full_inner_score(oof, sel)
    return sel


def test_date_debias_adopted_for_a_consistent_shift():
    oof = _oof([40] * 12, [i % 3 for i in range(12)], n_neg=6)
    sel = _scored(oof)
    out, c_sel, info, bias = _choose_date_debias(oof, sel)
    assert info["date_debias"] == "on" and bias == {"event": 40, "present_at_start": 0, "insufficient": 0}
    opts = info["date_debias_options"]
    assert opts["off"] == pytest.approx(85.0) and opts["on"] == pytest.approx(100.0)
    assert opts["bootstrap_p"] >= procedure.BOOTSTRAP_MIN
    assert (out["date_weak"] == pd.to_datetime(out["t_date"])).iloc[:12].all()


def test_date_debias_rejected_when_the_late_errors_are_a_tail():
    """Exact majority plus a far-late tail: the per-fold medians are 0, nothing to adopt."""
    errs = [0] * 9 + [200] * 3
    oof = _oof(errs, [i % 3 for i in range(12)], n_neg=6)
    _, _, info, bias = _choose_date_debias(oof, _scored(oof))
    assert info["date_debias"] == "off" and bias is None
    assert info["date_debias_options"]["bootstrap_p"] is None


def test_cap_ceiling_adopted_when_clip_hits_fall_back():
    oof = _oof([0] * 12, [i % 3 for i in range(12)], kwp=[6.0] * 9 + [500.0] * 3, n_neg=6)
    sel = _scored(oof)
    out, c_sel, info = _choose_cap_ceiling(oof, sel)
    assert info["cap_ceiling"] == "on"
    assert info["cap_ceiling_options"]["on"] - info["cap_ceiling_options"]["off"] >= procedure.CLASSIFIER_MARGIN
    assert out["kwp_ridge"].iloc[:12].tolist() == [6.0] * 12


def test_cap_ceiling_off_when_band_changes_nothing():
    oof = _oof([0] * 12, [i % 3 for i in range(12)], n_neg=6)
    _, _, info = _choose_cap_ceiling(oof, _scored(oof))
    assert info["cap_ceiling"] == "off" and info["cap_ceiling_options"]["bootstrap_p"] is None


# ------------------------------------------------------------ inference


def test_regime_date_subtracts_the_regime_bias():
    cd = SimpleNamespace(first_date=dt.date(2023, 3, 1))
    cp = Commissioning(dt.date(2023, 6, 1), 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, None, None)
    params = {"offset_event": 5, "start_lag": 10, "date_bias": {"event": 7, "present_at_start": -3, "insufficient": 2}}
    assert regime_date(cp, "event", cd, params, "weak_candidate") == dt.date(2023, 5, 30)  # +5 − 7
    assert regime_date(None, "present_at_start", cd, params, "first_date") == dt.date(2023, 2, 22)  # −10 + 3
    assert regime_date(None, "insufficient", cd, params, "first_date") == dt.date(2023, 2, 27)
    old = {"offset_event": 5, "start_lag": 10}  # model JSON without the key
    assert regime_date(cp, "event", cd, old, "weak_candidate") == dt.date(2023, 6, 6)


@pytest.fixture(scope="module")
def data():
    ts, labels, _ = generate_dataset(12, seed=4)
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


def test_model_without_new_keys_predicts_as_before_and_keys_act(data, tmp_path):
    table, labels = data
    ids = labels["customer_id"].tolist()
    model = fit(table, labels, ids, inner=False, options=frozenset())
    assert "date_bias" not in model["date"] and "band" not in model["capacity"]
    save_model(model, tmp_path / "old" / "model.json")
    base = predict(load_model(tmp_path / "old" / "model.json"), table, ids)[0]
    pd.testing.assert_frame_equal(predict(model, table, ids)[0], base)
    new = copy.deepcopy({k: v for k, v in model.items() if not k.startswith("_")})
    new["_objects"] = model["_objects"]
    new["date"]["date_bias"] = {"event": 7, "present_at_start": 7, "insufficient": 7}
    new["capacity"]["band"] = {"lo": 1e6, "hi": 2e6, "fallback": 8.0}
    save_model(new, tmp_path / "new" / "model.json")
    got = predict(load_model(tmp_path / "new" / "model.json"), table, ids)[0]
    on = base["pv_installed"].to_numpy(bool)
    assert on.any() and (got["pv_installed"] == base["pv_installed"]).all()
    assert ((base["raw_date"] - got["raw_date"]).dt.days.dropna() == 7).all()
    np.testing.assert_allclose(got["raw_kwp"], np.clip(model["capacity"]["kappa"] * 8.0, 0.5, 500))


def _stub_oof(late: int):
    def stub(table, labels, lab, train_ids, splits, names, cap_names, setting=procedure.BASE_SETTING, refit=None, **kw):
        t = lab.loc[train_ids]
        t_pv = t["pv_installed"].to_numpy(bool)
        dates = pd.Series(t["pv_installation_date"].to_numpy(), index=t.index) + pd.Timedelta(days=late)
        oof = pd.DataFrame(
            {
                "p_logreg": t_pv.astype(float),
                "p_catboost": t_pv.astype(float),
                "kwp_ridge": np.where(t_pv, t["pv_capacity_kwp"].to_numpy(float), 5.0),
                "kwp_catboost": np.nan,
                "regime": "event",
                "date_weak": dates,
                "date_first": dates,
                "fold": np.arange(len(t)) % len(splits),
                "cap_lo": 0.5,
                "cap_hi": 500.0,
                "cap_fb": 6.0,
            },
            index=t.index,
        )
        oof["t_pv"] = t_pv
        oof["t_date"] = t["pv_installation_date"].to_numpy()
        oof["t_kwp"] = t["pv_capacity_kwp"].to_numpy(float)
        return oof

    return stub


def test_fit_stores_the_adopted_bias_and_inference_applies_it(data, monkeypatch):
    table, labels = data
    ids = labels["customer_id"].tolist()
    opts = frozenset({"date_debias", "cap_ceiling"})
    monkeypatch.setattr(procedure, "_inner_oof", _stub_oof(40))
    model = fit(table, labels, ids, options=opts)
    sel = model["selection"]
    assert sel["date_debias"] == "on" and model["date"]["date_bias"]["event"] == 40
    assert sel["cap_ceiling"] == "off" and "band" not in model["capacity"]
    monkeypatch.setattr(procedure, "_inner_oof", _stub_oof(0))
    plain = fit(table, labels, ids, options=opts)
    assert plain["selection"]["date_debias"] == "off" and "date_bias" not in plain["date"]
    a, b = predict(model, table, ids)[0], predict(plain, table, ids)[0]
    ev = (a["regime"] == "event").to_numpy()
    assert ev.any() and ((b["raw_date"] - a["raw_date"]).dt.days[ev] == 40).all()


# --------------------------------------------- Tier R: dev scenarios (seeds < 1000)


def _fold(scenario, seed, n=36, fold=0):
    ts, lab, _ = generate_dataset(n, seed, scenario=scenario)
    table, labels = build_feature_table(as_loaded(ts)), typed_labels(lab)
    tr, te = outer_splits(labels, repeats=1, folds=3)[0][fold]
    return table, labels, tr, te


def _tp_errors(pred, labels, te):
    t = labels.set_index("customer_id").loc[te]
    p = pred.set_index("customer_id").loc[te]
    tp = (t["pv_installed"] & p["pv_installed"]).to_numpy(bool)
    err = (p["pv_installation_date"] - t["pv_installation_date"]).dt.days.to_numpy(float)[tp]
    return err, p["regime"].to_numpy()[tp]


def test_tier_r_label_lead_late_bias_is_corrected():
    """`label_lead_60` (labels before the visible onset): event dates run late under the shrunk offset;
    the fit adopts `date_debias` and the held-out event dates move toward the labels."""
    table, labels, tr, te = _fold("label_lead_60", 11)
    base = fit(table, labels, tr, options=frozenset())
    deb = fit(table, labels, tr, options=frozenset({"date_debias"}))
    assert deb["selection"]["date_debias"] == "on" and deb["date"]["date_bias"]["event"] > 0
    e0, r0 = _tp_errors(predict(base, table, te)[0], labels, te)
    e1, r1 = _tp_errors(predict(deb, table, te)[0], labels, te)
    assert np.median(e0[r0 == "event"]) >= 20  # the reproduced late bias
    assert abs(np.median(e1[r1 == "event"])) < np.median(e0[r0 == "event"])
    D = lambda e: np.maximum(0, 1 - np.abs(e) / 30).mean()  # noqa: E731
    assert D(e1) > D(e0)


def test_tier_r_cap_outlier_clip_hit_falls_back_into_the_band():
    """`cap_outlier`: a held-out outlier-scale customer's log-space capacity lands at a clip; the fit's
    learned band (from its training labels) replaces it by the training median, inside the band."""
    for fold in range(3):
        table, labels, tr, te = _fold("cap_outlier", 11, fold=fold)
        model = fit(table, labels, tr, inner=False, options=frozenset())
        pred = predict(model, table, te)[0].set_index("customer_id")
        t = labels.set_index("customer_id").loc[te]
        tp = (t["pv_installed"] & pred["pv_installed"]).to_numpy(bool)
        if np.isin(pred["pv_capacity_kwp"].to_numpy()[tp], [0.5, 500.0]).any():
            break
    else:
        pytest.fail("no clip hit reproduced")
    lab = labels.set_index("customer_id").loc[tr]
    band = fit_cap_band(lab.loc[lab["pv_installed"], "pv_capacity_kwp"].to_numpy(float))
    banded = {**model, "capacity": {**model["capacity"], "band": band}}
    got = predict(banded, table, te)[0].set_index("customer_id")
    true_k = t["pv_capacity_kwp"].to_numpy(float)[tp]
    P = lambda k: np.maximum(0, 1 - np.abs(k - true_k) / true_k / 0.5).mean()  # noqa: E731
    assert not np.isin(got["pv_capacity_kwp"].to_numpy()[tp], [0.5, 500.0]).any()
    assert P(got["pv_capacity_kwp"].to_numpy()[tp]) > P(pred["pv_capacity_kwp"].to_numpy()[tp])
