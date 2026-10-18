"""Schedule-robust feature block and the E1/E2 baseline switch (synthetic and hand-built data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import train
from pvdetect import procedure
from pvdetect.features import FEATURE_NAMES, SCHEDULE_FEATURES, _solar_shape_consistency, feature_names
from pvdetect.model import load_model, save_model
from pvdetect.pipeline import build_feature_table, compute_features, predict
from pvdetect.procedure import _select, fit
from pvdetect.season import fit_climatology
from pvdetect.synthetic import generate_dataset
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels
from tests.test_precision import _oof, _split_probs, _truth

SCHED = SCHEDULE_FEATURES[0]
CUT = pd.Timestamp("2023-01-01")

# ------------------------------------------------------------------ feature


def _daily(start="2022-01-01", end="2024-12-31", mid_drop=0.0, all_drop=0.0, weekday_only=False, seed=0):
    """Valid days with night 1.0, eve 1.2, mid 2.0 (+ small noise); after CUT the midday block drops by
    `mid_drop` (weekdays only if `weekday_only`) and every block by `all_drop`."""
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    weekend = idx.dayofweek.to_numpy() >= 5
    after = idx >= CUT
    mid = 2.0 + rng.normal(0, 0.05, idx.size) - np.where(after & (~weekend | (not weekday_only)), mid_drop, 0.0)
    shift = np.where(after, all_drop, 0.0)
    night, eve, mid = 1.0 - shift, 1.2 - shift, mid - shift
    return pd.DataFrame(
        {
            "mid": mid,
            "night": night,
            "eve": eve,
            "total": 96 * (mid + night + eve) / 3,
            "valid": True,
            "month": idx.month.to_numpy(),
            "weekend": weekend,
        },
        index=idx,
    )


def test_midday_reduction_on_both_day_types_is_clearly_positive():
    assert _solar_shape_consistency(_daily(mid_drop=1.0), CUT) > 0.5


def test_equal_all_day_reduction_is_about_zero():
    assert abs(_solar_shape_consistency(_daily(all_drop=0.5), CUT)) <= 0.05


def test_weekday_only_midday_reduction_is_about_zero():
    assert abs(_solar_shape_consistency(_daily(mid_drop=1.0, weekday_only=True), CUT)) <= 0.05


def test_insufficient_summer_coverage_is_nan():
    v = _daily(mid_drop=1.0)
    # pre side: only June 2022 before the cut (≈ 8 weekend days < 20)
    assert np.isnan(_solar_shape_consistency(v, pd.Timestamp("2022-07-01")))
    # post side: the first 14 days after the cut are excluded; mid-August cut leaves no post summer in 2024
    assert np.isnan(_solar_shape_consistency(v, pd.Timestamp("2024-08-20")))
    # weekend stratum only: drop post-cut summer weekends below 20 days
    jja_we = v.index[(v.index >= CUT) & v["month"].isin((6, 7, 8)).to_numpy() & v["weekend"].to_numpy()]
    assert np.isnan(_solar_shape_consistency(v.drop(jja_we[19:]), CUT))
    assert np.isfinite(_solar_shape_consistency(v.drop(jja_we[20:]), CUT))
    # no positive normaliser
    assert np.isnan(_solar_shape_consistency(v.assign(total=0.0), CUT))


def test_column_always_present_and_opt_in():
    assert SCHED not in FEATURE_NAMES and SCHED not in feature_names(True, persistence=True)
    ts, labels, _ = generate_dataset(6, seed=7)
    table = build_feature_table(as_loaded(ts))
    ids = labels["customer_id"].tolist()
    lab = typed_labels(labels).set_index("customer_id")
    pv = {c: d.date() for c, d in lab.loc[lab["pv_installed"], "pv_installation_date"].items() if pd.notna(d)}
    clim = fit_climatology(table.dailies(), lab.index[~lab["pv_installed"]].tolist(), pv)
    F, _ = compute_features(table, ids, clim)
    assert SCHED in F.columns


# ---------------------------------------------------------- option selection


@pytest.fixture(scope="module")
def data():
    ts, labels, _ = generate_dataset(12, seed=4)
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


@pytest.mark.parametrize("prefer_on", [False, True])
def test_schedule_block_follows_inner_cv(data, monkeypatch, tmp_path, prefer_on):
    table, labels = data
    seen = []

    def stub(table, labels, lab, train_ids, splits, names, cap_names, setting=procedure.BASE_SETTING, **_):
        seen.append((list(names), list(cap_names), len(splits)))
        t = lab.loc[train_ids]
        on = t["pv_installed"].to_numpy(float)
        return _oof(t, on, on, ok=(SCHED in names) == prefer_on)

    monkeypatch.setattr(procedure, "_inner_oof", stub)
    ids = labels["customer_id"].tolist()
    # the stub scores no weekly head / C / date options
    options = procedure.DEFAULT_OPTIONS - {"weekly", "ctune", "offset_tri", "recur_strict", "recent", "smallpv", "onset_first", "tail_guard", "tobit"} | {"schedule"}
    model = fit(table, labels, ids, options=options)
    sel = model["selection"]
    base = feature_names(model["export_enabled"])
    assert [s[0] for s in seen] == [base, base + ["summer_suppressed_week_frac"], base + [SCHED]]
    assert len({tuple(map(str, s[1:])) for s in seen}) == 1  # same capacity features and inner splits
    opts = sel["schedule_options"]
    if prefer_on:
        assert sel["schedule_block"] == "on" and model["feature_names"] == base + [SCHED]
        assert opts["on"] - opts["off"] >= procedure.CLASSIFIER_MARGIN
        assert opts["bootstrap_p"] >= procedure.BOOTSTRAP_MIN
    else:
        assert sel["schedule_block"] == "off" and SCHED not in model["feature_names"]
        assert opts["on"] < opts["off"] and opts["bootstrap_p"] is None
    save_model(model, tmp_path / "model.json")
    pred, _ = predict(load_model(tmp_path / "model.json"), table, ids)
    assert len(pred) == len(ids)


def test_schedule_runs_without_e1e2(data, monkeypatch):
    table, labels = data
    seen = []

    def stub(table, labels, lab, train_ids, splits, names, cap_names, setting=procedure.BASE_SETTING, **_):
        seen.append(list(names))
        t = lab.loc[train_ids]
        on = t["pv_installed"].to_numpy(float)
        return _oof(t, on, on, ok=SCHED in names)

    monkeypatch.setattr(procedure, "_inner_oof", stub)
    model = fit(table, labels, labels["customer_id"].tolist(), options=frozenset({"schedule"}))
    base = feature_names(model["export_enabled"])
    assert seen == [base, base + [SCHED]]
    assert model["selection"]["schedule_block"] == "on" and "feature_block" not in model["selection"]


# ------------------------------------------------------------ baseline switch


def test_select_without_avg_is_the_pre_e2_record():
    t = _truth()
    oof = _oof(t, *_split_probs(t["pv_installed"].to_numpy(bool), 3))
    assert _select(oof)["classifier"] == "avg"
    sel = _select(oof, avg=False)
    assert sel["classifier"] in ("logreg", "catboost")
    assert "inner_score_avg" not in sel and "bootstrap_p_avg" not in sel
    assert list(sel)[:9] == [
        "classifier",
        "threshold",
        "threshold_initial",
        "inner_score_logreg",
        "inner_score_catboost",
        "inner_score",
        "inner_score_initial",
        "bootstrap_p",
        "capacity",
    ]


def test_fit_without_e1e2_skips_both_options(data, monkeypatch):
    table, labels = data
    seen = []

    def stub(table, labels, lab, train_ids, splits, names, cap_names, setting=procedure.BASE_SETTING, **_):
        seen.append(list(names))
        t = lab.loc[train_ids]
        return _oof(t, *_split_probs(t["pv_installed"].to_numpy(bool), 2))

    monkeypatch.setattr(procedure, "_inner_oof", stub)
    model = fit(table, labels, labels["customer_id"].tolist(), options=frozenset())
    sel = model["selection"]
    assert len(seen) == 1 and model["feature_names"] == feature_names(model["export_enabled"])
    assert sel["classifier"] != "avg"
    assert not {"feature_block", "feature_options", "inner_score_avg", "bootstrap_p_avg", "schedule_block"} & set(sel)


def test_unknown_option_is_rejected(data):
    table, labels = data
    with pytest.raises(ValueError, match="unknown options"):
        fit(table, labels, labels["customer_id"].tolist(), inner=False, options=frozenset({"e3"}))


def test_selection_line_reports_schedule():
    common = {"classifier": "logreg", "capacity": "ridge", "threshold": 0.5, "kappa": 1.0, "offset_event": 0}
    line = train._selection_line([{**common, "schedule_block": "on"}, {**common, "schedule_block": "off"}])
    assert "schedule off=1/2 on=1/2" in line
    assert "schedule" not in train._selection_line([common])
