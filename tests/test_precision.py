"""Precision options (Step 2.5-E): E1 persistence feature block and E2 probability averaging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import train
from pvdetect import procedure
from pvdetect.features import (
    FEATURE_NAMES,
    PERSISTENCE_FEATURES,
    _summer_suppressed_week_frac,
    feature_names,
)
from pvdetect.model import linear_score, load_model, referenced_files, save_model, sigmoid
from pvdetect.pipeline import build_feature_table, compute_features, predict
from pvdetect.procedure import _select, fit
from pvdetect.season import Climatology, fit_climatology
from pvdetect.synthetic import CustomerSpec, Event, PVSpec, generate_customer, generate_dataset
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels

PERSIST = PERSISTENCE_FEATURES[0]
RELEASE_DIR = Path(__file__).resolve().parents[1] / "scripts"
# The stubbed inner OOFs below score neither the weekly head nor other logreg Cs (see tests/test_weekly.py).
E1E2 = procedure.DEFAULT_OPTIONS - {"weekly", "ctune", "offset_tri", "recur_strict", "recent", "smallpv"}


# ------------------------------------------------------------------ E1 feature


def test_persistence_block_is_opt_in():
    assert PERSIST not in FEATURE_NAMES
    for export in (False, True):
        assert feature_names(export, persistence=True) == feature_names(export) + [PERSIST]


def _week_frame(days: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"month": days.month}, index=days)


def _hand_built(winter_days: int, summer_weeks: int, week_days: int = 7, suppressed: int = 0, drop: float = 1.0):
    """`winter_days` January days at ds=0; `summer_weeks` ISO weeks from Mon 2024-04-01 with `week_days`
    valid days each; the first `suppressed` weeks sit `drop` below the winter median."""
    winter = pd.date_range("2024-01-01", periods=winter_days, freq="D")
    summer = pd.DatetimeIndex(
        [pd.Timestamp("2024-04-01") + pd.Timedelta(days=7 * w + d) for w in range(summer_weeks) for d in range(week_days)]
    )
    ds_summer = np.repeat(np.where(np.arange(summer_weeks) < suppressed, -drop, 0.0), week_days)
    return _week_frame(winter.append(summer)), np.r_[np.zeros(winter_days), ds_summer]


def test_persistence_nan_rules_on_hand_built_frames():
    assert _summer_suppressed_week_frac(*_hand_built(30, 8, suppressed=2)) == pytest.approx(0.25)
    assert np.isnan(_summer_suppressed_week_frac(*_hand_built(29, 8, suppressed=2)))  # < 30 winter days
    assert np.isnan(_summer_suppressed_week_frac(*_hand_built(30, 7, suppressed=2)))  # < 8 summer weeks
    assert np.isnan(_summer_suppressed_week_frac(*_hand_built(30, 26, week_days=3, suppressed=26)))  # 3-day weeks
    assert _summer_suppressed_week_frac(*_hand_built(30, 26, week_days=4, suppressed=13)) == pytest.approx(0.5)


def test_persistence_threshold_is_half_below_winter_median():
    assert _summer_suppressed_week_frac(*_hand_built(30, 10, suppressed=10, drop=0.5)) == 1.0
    assert _summer_suppressed_week_frac(*_hand_built(30, 10, suppressed=10, drop=0.49)) == 0.0


def test_persistence_vacation_low_summer_install_high():
    """R (variant A, dev seeds): a 3-week vacation without PV stays < 0.2; a PV installed in April of the
    last 365 days, with almost every summer week after the install, scores > 0.8."""
    ref = [CustomerSpec(f"{900 + i:06d}", "2023-01-01", "2025-02-28") for i in range(4)]
    vacation = CustomerSpec("000001", "2023-01-01", "2024-12-31", events=(Event("vacation", "2024-07-08", "2024-07-29"),))
    installed = CustomerSpec("000002", "2023-01-01", "2025-02-28", pv=PVSpec("2024-04-15", 6.0))
    specs = ref + [vacation, installed]
    table = build_feature_table(as_loaded(pd.concat([generate_customer(s, 30 + i) for i, s in enumerate(specs)])))
    clim = fit_climatology(table.dailies(), [s.customer_id for s in ref], {})
    F, _ = compute_features(table, table.ids, clim)
    assert PERSIST in F.columns
    assert F.at["000001", PERSIST] < 0.2
    assert F.at["000002", PERSIST] > 0.8


# -------------------------------------------------------- stubbed inner OOF


def _oof(t: pd.DataFrame, p_logreg: np.ndarray, p_catboost: np.ndarray, ok: bool = True) -> pd.DataFrame:
    """Inner OOF frame for truth `t`; dates and capacities exact when `ok`, else 90 days / 3× off."""
    t_pv = t["pv_installed"].to_numpy(bool)
    kwp = t["pv_capacity_kwp"].to_numpy(float) * (1.0 if ok else 3.0)
    dates = pd.Series(t["pv_installation_date"].to_numpy(), index=t.index) + pd.Timedelta(days=0 if ok else 90)
    oof = pd.DataFrame(
        {
            "p_logreg": p_logreg,
            "p_catboost": p_catboost,
            "kwp_ridge": np.where(t_pv, kwp, 5.0),
            "kwp_catboost": np.where(t_pv, kwp, 5.0),
            "regime": "event",
            "date_weak": dates,
            "date_first": dates,
        },
        index=t.index,
    )
    oof["t_pv"] = t_pv
    oof["t_date"] = t["pv_installation_date"].to_numpy()
    oof["t_kwp"] = t["pv_capacity_kwp"].to_numpy(float)
    return oof


def _split_probs(t_pv: np.ndarray, k: int, flawed: tuple[str, ...] = ("logreg", "catboost")):
    """Logreg misranks the first k positives and negatives, CatBoost the next k; their mean separates all."""
    base = np.where(t_pv, 0.95, 0.05)
    pos, neg = np.flatnonzero(t_pv), np.flatnonzero(~t_pv)
    out = {}
    for j, kind in enumerate(("logreg", "catboost")):
        p = base.copy()
        if kind in flawed:
            p[pos[j * k : (j + 1) * k]] = 0.30
            p[neg[j * k : (j + 1) * k]] = 0.70
        out[kind] = p
    return out["logreg"], out["catboost"]


def _truth(n: int = 40) -> pd.DataFrame:
    on = np.arange(n) % 2 == 0
    return pd.DataFrame(
        {
            "pv_installed": on,
            "pv_installation_date": pd.to_datetime(np.where(on, "2023-05-01", None)),
            "pv_capacity_kwp": np.where(on, 8.0, np.nan),
        },
        index=pd.Index([f"c{i:02d}" for i in range(n)], name="customer_id"),
    )


# ---------------------------------------------------------------- E2 selection


def test_avg_adopted_when_it_beats_the_rule_winner():
    t = _truth()
    sel = _select(_oof(t, *_split_probs(t["pv_installed"].to_numpy(bool), 3)))
    assert sel["classifier"] == "avg"
    assert sel["inner_score_avg"] == pytest.approx(100.0)
    assert sel["inner_score_avg"] - sel["inner_score_logreg"] >= procedure.CLASSIFIER_MARGIN
    assert sel["bootstrap_p_avg"] >= procedure.BOOTSTRAP_MIN
    assert sel["inner_score"] == pytest.approx(100.0)


@pytest.mark.parametrize("flawed, expect", [((), "logreg"), (("logreg",), "catboost")])
def test_avg_not_adopted_without_margin(flawed, expect):
    t = _truth()
    sel = _select(_oof(t, *_split_probs(t["pv_installed"].to_numpy(bool), 3, flawed)))
    assert sel["classifier"] == expect
    assert sel["bootstrap_p_avg"] is None
    assert sel["inner_score_avg"] == pytest.approx(sel[f"inner_score_{expect}"])


def test_avg_skipped_without_catboost_probabilities():
    t = _truth()
    p_lr, _ = _split_probs(t["pv_installed"].to_numpy(bool), 3)
    sel = _select(_oof(t, p_lr, np.full(len(t), np.nan)))
    assert sel["classifier"] == "logreg" and np.isnan(sel["inner_score_avg"])


@pytest.fixture(scope="module")
def data():
    ts, labels, _ = generate_dataset(12, seed=4)
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


def test_fit_with_avg_saves_both_sub_models_and_predicts_their_mean(data, monkeypatch, tmp_path):
    table, labels = data

    def stub(table, labels, lab, train_ids, splits, names, cap_names, setting=procedure.BASE_SETTING, **_):
        t = lab.loc[train_ids]
        return _oof(t, *_split_probs(t["pv_installed"].to_numpy(bool), 2))

    monkeypatch.setattr(procedure, "_inner_oof", stub)
    ids = labels["customer_id"].tolist()
    model = fit(table, labels, ids, options=E1E2)
    clf = model["classifier"]
    assert clf["kind"] == "avg" and clf["logreg"] is not None and clf["catboost_file"] == "catboost_clf.cbm"
    assert model["selection"]["feature_block"] == "base"
    save_model(model, tmp_path / "model.json")
    loaded = load_model(tmp_path / "model.json")
    assert referenced_files(loaded) == ["catboost_clf.cbm"]
    monkeypatch.syspath_prepend(str(RELEASE_DIR))
    from check_release import model_files

    assert model_files(loaded) == ["catboost_clf.cbm"]
    pred, _ = predict(loaded, table, ids)
    F, _ = compute_features(table, ids, Climatology.from_json(loaded["climatology"]))
    X = F[loaded["feature_names"]].to_numpy(float)
    p_lr = sigmoid(linear_score(loaded["classifier"]["logreg"], X))
    p_cb = loaded["_objects"]["catboost_clf"].predict_proba(X)[:, 1]
    np.testing.assert_allclose(pred["prob"].to_numpy(float), (p_lr + p_cb) / 2, rtol=0, atol=1e-12)


# ------------------------------------------------------- feature-block option


@pytest.mark.parametrize("prefer_persistence", [False, True])
def test_fit_feature_block_follows_inner_cv(data, monkeypatch, tmp_path, prefer_persistence):
    table, labels = data
    seen = []

    def stub(table, labels, lab, train_ids, splits, names, cap_names, setting=procedure.BASE_SETTING, **_):
        seen.append((list(names), list(cap_names), len(splits)))
        t = lab.loc[train_ids]
        on = t["pv_installed"].to_numpy(float)
        return _oof(t, on, on, ok=(PERSIST in names) == prefer_persistence)

    monkeypatch.setattr(procedure, "_inner_oof", stub)
    ids = labels["customer_id"].tolist()
    model = fit(table, labels, ids, options=E1E2)
    sel = model["selection"]
    base_names = feature_names(model["export_enabled"])
    assert [s[0] for s in seen] == [base_names, base_names + [PERSIST]]
    assert seen[0][1:] == seen[1][1:]  # same capacity features and the same inner splits
    opts = sel["feature_options"]
    if prefer_persistence:
        assert sel["feature_block"] == "persistence"
        assert model["feature_names"] == base_names + [PERSIST]
        assert opts["persistence"] - opts["base"] >= procedure.CLASSIFIER_MARGIN
        assert opts["bootstrap_p"] >= procedure.BOOTSTRAP_MIN
    else:
        assert sel["feature_block"] == "base"
        assert PERSIST not in model["feature_names"]
        assert opts["persistence"] < opts["base"] and opts["bootstrap_p"] is None
    save_model(model, tmp_path / "model.json")
    pred, _ = predict(load_model(tmp_path / "model.json"), table, ids)
    assert len(pred) == len(ids) and PERSIST in pred.columns


def test_selection_line_reports_avg_and_feature_block():
    common = {"capacity": "ridge", "threshold": 0.5, "kappa": 1.0, "offset_event": 0}
    infos = [
        {**common, "classifier": "avg", "feature_block": "persistence"},
        {**common, "classifier": "logreg", "feature_block": "base"},
    ]
    line = train._selection_line(infos)
    assert "classifier logreg=1/2 catboost=0/2 avg=1/2" in line
    assert "features base=1/2 persistence=1/2" in line
