"""Recall-targeted feature blocks `recent` and `smallpv` (synthetic and hand-built data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import train
from pvdetect import procedure
from pvdetect.features import (
    RECENT_FEATURES,
    SMALLPV_FEATURES,
    feature_names,
    label_free_features,
    recent_features,
    smallpv_features,
)
from pvdetect.model import load_model, save_model
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.procedure import fit
from pvdetect.synthetic import generate_dataset
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels
from tests.test_precision import _oof

# ------------------------------------------------------------------ hand-built features


def _daily(start="2022-01-01", end="2024-12-31", s=-0.5, tail_s=None, tail_zero=0.0, mid=1.0, night=1.0, eve=1.0):
    """Valid days with constant `s`; the last 84 days get `tail_s` and `tail_zero` when given."""
    idx = pd.date_range(start, end, freq="D")
    sv = np.full(idx.size, float(s))
    zero = np.zeros(idx.size)
    if tail_s is not None:
        sv[-84:] = tail_s
    zero[-84:] = tail_zero
    return pd.DataFrame(
        {
            "mid": mid,
            "night": night,
            "eve": eve,
            "s": sv,
            "zero": zero,
            "valid": True,
            "month": idx.month.to_numpy(),
            "woy": np.minimum(idx.isocalendar().week.to_numpy().astype(int), 52),
        },
        index=idx,
    )


def test_recent_features_exact_on_a_deeper_tail():
    f = recent_features(_daily(tail_s=-1.5, tail_zero=0.25))
    assert f["recent_matched_step"] == pytest.approx(1.0)
    assert f["recent_prior_step"] == pytest.approx(1.0)
    assert f["recent_zero_frac"] == pytest.approx(0.25)


def test_recent_features_are_zero_without_a_change():
    f = recent_features(_daily())
    assert f["recent_matched_step"] == pytest.approx(0.0)
    assert f["recent_prior_step"] == pytest.approx(0.0)
    assert f["recent_zero_frac"] == pytest.approx(0.0)


def test_recent_matched_step_ignores_other_calendar_weeks():
    v = _daily(tail_s=-1.5)
    other = v.index[(v.index >= "2023-01-01") & (v.index < "2023-02-01")]  # winter weeks, not the tail's weeks
    v.loc[other, "s"] = 5.0
    assert recent_features(v)["recent_matched_step"] == pytest.approx(1.0)


def test_recent_features_need_enough_days_and_history():
    assert all(np.isnan(x) for x in recent_features(_daily().iloc[:0]).values())
    short = _daily(end="2022-01-15")  # 15 valid days: no tail
    assert all(np.isnan(x) for x in recent_features(short).values())
    one_year = recent_features(_daily(end="2022-07-31", tail_s=-1.5, tail_zero=0.5))
    assert np.isnan(one_year["recent_matched_step"])  # no earlier calendar year to match
    assert one_year["recent_prior_step"] == pytest.approx(1.0)
    assert one_year["recent_zero_frac"] == pytest.approx(0.5)


def test_smallpv_features_exact_on_a_summer_deficit():
    v = _daily()
    summer = v["month"].isin((4, 5, 6, 7, 8, 9)).to_numpy()
    v["mid"] = np.where(summer, 0.5, 1.0)
    f = smallpv_features(v)
    expected = np.log(0.51 / 1.01)
    assert f["deficit_contrast"] == pytest.approx(expected)
    assert f["deficit_q10_contrast"] == pytest.approx(expected)


def test_smallpv_features_are_scale_free_and_ignore_a_flat_load():
    v = _daily()
    summer = v["month"].isin((4, 5, 6, 7, 8, 9)).to_numpy()
    v["mid"] = np.where(summer, 0.5, 1.0)
    big = v.assign(mid=v["mid"] * 1000, night=v["night"] * 1000, eve=v["eve"] * 1000)
    assert smallpv_features(big)["deficit_contrast"] == pytest.approx(-np.log(2), abs=1e-4)  # ε is negligible here
    assert smallpv_features(v)["deficit_contrast"] == pytest.approx(-np.log(2), abs=0.02)
    assert smallpv_features(_daily())["deficit_contrast"] == pytest.approx(0.0)


def test_smallpv_features_need_summer_and_winter_days():
    assert all(np.isnan(x) for x in smallpv_features(_daily(start="2024-05-01", end="2024-09-30")).values())
    assert all(np.isnan(x) for x in smallpv_features(_daily().iloc[:0]).values())


def test_blocks_are_label_free_and_not_in_the_base_names():
    ts, _, _ = generate_dataset(4, seed=7)
    table = build_feature_table(as_loaded(ts))
    cd = table.customers[table.ids[0]]
    lf = label_free_features(cd)
    assert set(RECENT_FEATURES + SMALLPV_FEATURES) <= set(lf)
    assert not set(RECENT_FEATURES + SMALLPV_FEATURES) & set(feature_names(True, persistence=True))


# ------------------------------------------------------------------ option selection


@pytest.fixture(scope="module")
def data():
    ts, labels, _ = generate_dataset(12, seed=4)
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


NO_OTHER = procedure.DEFAULT_OPTIONS - {"weekly", "ctune", "offset_tri", "recur_strict", "e1", "e2", "smallpv", "recent", "onset_first", "tail_guard"}


@pytest.mark.parametrize("key,extra", [("recent", RECENT_FEATURES), ("smallpv", SMALLPV_FEATURES)])
@pytest.mark.parametrize("prefer_on", [False, True])
def test_block_follows_inner_cv(data, monkeypatch, tmp_path, key, extra, prefer_on):
    table, labels = data
    seen = []

    def stub(table, labels, lab, train_ids, splits, names, cap_names, setting=procedure.BASE_SETTING, **_):
        seen.append((list(names), list(cap_names), len(splits)))
        t = lab.loc[train_ids]
        on = t["pv_installed"].to_numpy(float)
        return _oof(t, on, on, ok=(extra[0] in names) == prefer_on)

    monkeypatch.setattr(procedure, "_inner_oof", stub)
    ids = labels["customer_id"].tolist()
    model = fit(table, labels, ids, options=NO_OTHER | {key})
    sel = model["selection"]
    base = feature_names(model["export_enabled"])
    assert [s[0] for s in seen] == [base, base + list(extra)]
    assert len({tuple(map(str, s[1:])) for s in seen}) == 1  # same capacity features and inner splits
    opts = sel[f"{key}_options"]
    if prefer_on:
        assert sel[f"{key}_block"] == "on" and model["feature_names"] == base + list(extra)
        assert opts["on"] - opts["off"] >= procedure.CLASSIFIER_MARGIN
        assert opts["bootstrap_p"] >= procedure.BOOTSTRAP_MIN
    else:
        assert sel[f"{key}_block"] == "off" and not set(extra) & set(model["feature_names"])
        assert opts["on"] < opts["off"] and opts["bootstrap_p"] is None
    save_model(model, tmp_path / "model.json")
    pred, _ = predict(load_model(tmp_path / "model.json"), table, ids)
    assert len(pred) == len(ids)


def test_both_blocks_stack_when_both_are_chosen(data, monkeypatch):
    table, labels = data
    seen = []

    def stub(table, labels, lab, train_ids, splits, names, cap_names, setting=procedure.BASE_SETTING, **_):
        seen.append(list(names))
        t = lab.loc[train_ids]
        on = t["pv_installed"].to_numpy(float)
        return _oof(t, on, on, ok=SMALLPV_FEATURES[0] in names)

    monkeypatch.setattr(procedure, "_inner_oof", stub)
    model = fit(table, labels, labels["customer_id"].tolist(), options=NO_OTHER | {"recent", "smallpv"})
    base = feature_names(model["export_enabled"])
    assert seen == [base, base + list(RECENT_FEATURES), base + list(SMALLPV_FEATURES)]
    assert model["selection"]["recent_block"] == "off" and model["selection"]["smallpv_block"] == "on"


def test_blocks_are_in_the_default_pool_and_can_be_dropped():
    assert {"recent", "smallpv"} <= procedure.DEFAULT_OPTIONS <= procedure.ALL_OPTIONS


def test_fit_without_the_blocks_has_no_trace_of_them(data):
    table, labels = data
    model = fit(table, labels, labels["customer_id"].tolist(), inner=False, options=frozenset())
    assert not {"recent_block", "smallpv_block", "recent_options", "smallpv_options"} & set(model["selection"])
    assert not set(RECENT_FEATURES + SMALLPV_FEATURES) & set(model["feature_names"])


def test_inference_runs_with_and_without_an_adopted_block(data, tmp_path):
    """A model that adopted a block lists its columns; one that did not (an older model) never needs them."""
    table, labels = data
    ids = labels["customer_id"].tolist()
    old = fit(table, labels, ids, inner=False, options=frozenset())
    save_model(old, tmp_path / "old" / "model.json")
    pred_old, _ = predict(load_model(tmp_path / "old" / "model.json"), table, ids)
    new = fit(table, labels, ids, inner=False, options=frozenset())
    new["feature_names"] = new["feature_names"] + list(RECENT_FEATURES) + list(SMALLPV_FEATURES)
    new["classifier"] = {**new["classifier"], "logreg": _refit_logreg(table, labels, ids, new["feature_names"])}
    save_model(new, tmp_path / "new" / "model.json")
    pred_new, _ = predict(load_model(tmp_path / "new" / "model.json"), table, ids)
    assert len(pred_old) == len(pred_new) == len(ids)
    assert pred_new["pv_installed"].isin([True, False]).all()


def _refit_logreg(table, labels, ids, names):
    from pvdetect.model import fit_logreg
    from pvdetect.pipeline import compute_features
    from pvdetect.season import fit_climatology

    lab = labels.set_index("customer_id")
    pv = {c: d.date() for c, d in lab.loc[lab["pv_installed"], "pv_installation_date"].items() if pd.notna(d)}
    clim = fit_climatology(table.dailies(), lab.index[~lab["pv_installed"]].tolist(), pv)
    F, _ = compute_features(table, ids, clim)
    return fit_logreg(F[names].to_numpy(float), lab.loc[ids, "pv_installed"].to_numpy(bool), C=0.5)


def test_selection_line_reports_the_blocks():
    common = {"classifier": "logreg", "capacity": "ridge", "threshold": 0.5, "kappa": 1.0, "offset_event": 0}
    infos = [{**common, "recent_block": "on", "smallpv_block": "off"}, {**common, "recent_block": "off", "smallpv_block": "off"}]
    line = train._selection_line(infos)
    assert "recent off=1/2 on=1/2" in line and "smallpv off=2/2 on=0/2" in line
    assert "recent" not in train._selection_line([common])


# ------------------------------------------------------------------ Tier R (dev seeds, variant A)


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(score)
    r = pd.Series(score[ok]).rank().to_numpy()
    pos = y[ok]
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def test_recent_features_separate_late_installs_that_the_seasonal_contrast_misses():
    """Dev scenario `late_install` (installation 8–16 weeks before the end): the tail features rank PV
    customers above the rest while the 365-day seasonal contrast cannot. Regression check, not accuracy evidence."""
    from pvdetect.pipeline import compute_features
    from pvdetect.season import zero_climatology

    ts, lab, _ = generate_dataset(32, 11, scenario="late_install")
    table, labels = build_feature_table(as_loaded(ts)), typed_labels(lab)
    F, _ = compute_features(table, labels["customer_id"].tolist(), zero_climatology())
    y = labels["pv_installed"].to_numpy(bool)
    assert _auc(F["recent_zero_frac"].to_numpy(float), y) >= 0.9
    assert _auc(F["recent_prior_step"].to_numpy(float), y) >= 0.9
    assert _auc(F["recent_matched_step"].to_numpy(float), y) >= 0.85
    assert _auc(F["seasonal_contrast"].to_numpy(float), y) <= 0.7
