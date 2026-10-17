"""Weekly-evidence head and the logreg C option (Step 2.5-E); synthetic and hand-built data only."""

from __future__ import annotations

import json
import socket

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import inference
import train
from pvdetect import pipeline, procedure, pseudolabel, weekly
from pvdetect.evaluation import outer_splits
from pvdetect.io import load_labels, load_timeseries
from pvdetect.model import load_model, save_model
from pvdetect.pipeline import build_feature_table, customer_data, predict
from pvdetect.procedure import _choose_logreg_c, _select, fit
from pvdetect.season import Climatology, fit_climatology
from pvdetect.synthetic import CustomerSpec, Event, PVSpec, generate_customer, generate_dataset
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels
from tests.test_precision import _oof, _truth

NEW_KEYS = {
    "inner_score_weekly",
    "inner_score_avg_weekly",
    "bootstrap_p_weekly",
    "bootstrap_p_avg_weekly",
    "logreg_C",
    "logreg_C_options",
}

# ------------------------------------------------------------------ windows


def _hand_built_customer():
    """13 Mon–Sun weeks from Mon 2024-01-01 at 0.2 kWh per slot. Week 2 has 3 invalid days (night missing),
    week 4 has 2; in week 6 the 12:00–12:59 slots are missing, in week 7 they read 0; week 10 has no data."""
    dates = np.arange("2024-01-01", "2024-04-01", dtype="datetime64[D]")
    M = np.full((dates.size, 96), 0.2)
    M[14:17, 0:20] = np.nan  # week 2: Mon–Wed invalid
    M[28:30, 0:20] = np.nan  # week 4: Mon–Tue invalid
    M[42:49, 48:52] = np.nan  # week 6: noon hour missing (16 of 20 midday slots observed: still valid)
    M[49:56, 48:52] = 0.0  # week 7: noon hour reads zero
    M[70:77] = np.nan  # week 10: nothing
    return customer_data(dates, M, M[np.isfinite(M)])


WEEK_STARTS = ["2024-01-01", "2024-01-08", "2024-01-22", "2024-01-29", "2024-02-05", "2024-02-12",
               "2024-02-19", "2024-02-26", "2024-03-04", "2024-03-18", "2024-03-25"]  # fmt: skip


def test_kept_weeks_coverage_and_missing_slots_are_not_zeros():
    weeks = weekly.week_windows(_hand_built_customer())
    assert weeks.start.tolist() == [np.datetime64(d, "D") for d in WEEK_STARTS]  # week 2 (4 valid days), week 10 out
    lf = dict(zip(weekly.WEEK_FEATURES[2:], weeks.lf.T))
    assert lf["coverage"].tolist() == [1.0, 1.0, 1.0, 5 / 7] + [1.0] * 7
    gap, zero = WEEK_STARTS.index("2024-02-12"), WEEK_STARTS.index("2024-02-19")
    assert lf["zero_frac"][gap] == 0.0 and lf["zero_frac"][zero] == pytest.approx(4 / 20)
    assert np.isnan(lf["zero_run_h"][gap]) and lf["zero_run_h"][zero] == 1.0
    assert lf["zero_run_h"][0] == 0.0
    assert lf["night_rel"][0] == pytest.approx(0.0)
    assert lf["amp"][0] == pytest.approx(np.log1p(96 * 0.2))
    assert np.isfinite(weeks.s).sum(axis=1).tolist() == [7, 7, 7, 5] + [7] * 7  # masked, never zero-filled


def _labels(pv: bool, date) -> tuple[list[str], list[bool]]:
    cd = _hand_built_customer()
    weeks = weekly.week_windows(cd)
    keep, y = weekly.window_labels(weeks.start, cd.first_date, pv, pd.Timestamp(date) if date else pd.NaT)
    kept = [str(s) for s in weeks.start[keep]]
    return kept, y[keep].tolist()


def test_window_labels_drop_the_seven_day_band():
    # Monday 2024-02-12: weeks ending ≤ 02-05 are 0, weeks starting ≥ 02-19 are 1; 02-05 and 02-12 dropped
    kept, y = _labels(True, "2024-02-12")
    assert kept == WEEK_STARTS[:4] + WEEK_STARTS[6:]
    assert y == [False] * 4 + [True] * 5
    # Wednesday 2024-02-14: band [02-07, 02-21] also drops the week starting 02-19
    kept, y = _labels(True, "2024-02-14")
    assert kept == WEEK_STARTS[:4] + WEEK_STARTS[7:]
    assert y == [False] * 4 + [True] * 4


def test_window_labels_present_at_start_non_pv_and_undated():
    for date in ("2023-06-01", "2023-12-29"):  # before the first data, even within 7 days of it: all 1
        assert _labels(True, date) == (WEEK_STARTS, [True] * len(WEEK_STARTS))
    assert _labels(False, None) == (WEEK_STARTS, [False] * len(WEEK_STARTS))
    assert _labels(True, None) == ([], [])


def test_weights_share_one_per_customer_then_balance_classes():
    customer = np.array([0, 0, 0, 0, 1, 1])
    y = np.array([False, False, True, True, False, False])
    w = weekly.balanced_weights(customer, y)
    np.testing.assert_allclose(w, [1 / 6, 1 / 6, 0.5, 0.5, 1 / 3, 1 / 3])
    assert w[y].sum() == pytest.approx(w[~y].sum())
    np.testing.assert_allclose(weekly.balanced_weights(customer, np.zeros(6, bool)), [0.25] * 4 + [0.5] * 2)


# ---------------------------------------------------------------- aggregation


def test_rolling_median_max_aggregation():
    low = [0.1] * 10
    assert weekly.aggregate(low + [0.9] * 5) == pytest.approx(0.9)  # late run of 5 high weeks
    assert weekly.aggregate(low + [0.8, 0.8] + low) == pytest.approx(0.45)  # isolated 2-week spike: midpoint
    assert weekly.aggregate(low + [0.95] + low) == pytest.approx(0.1)  # single-week spike
    assert weekly.aggregate([0.7] * 30) == pytest.approx(0.7)  # present at start
    assert weekly.aggregate([0.2, 0.9, 0.4]) == pytest.approx(0.4)  # < 4 weeks: plain median
    assert weekly.aggregate([]) == 0.5
    assert weekly.aggregate([np.nan, 0.9, 0.9, 0.9, np.nan]) == pytest.approx(0.9)  # NaN-safe
    assert weekly.aggregate([np.nan] * 5) == 0.5
    assert weekly.aggregate([0.1, 0.2, 0.3, 0.9, 0.9, 0.9], window=3) == pytest.approx(0.9)


# ------------------------------------------------------------------ R test


def test_head_scores_late_install_high_and_vacation_low():
    """R (variant A, dev seeds): a head trained on generate_dataset(24, seed=1) scores a PV installed ~10
    weeks before the series end > 0.5 and a non-PV customer with a 3-week vacation < 0.5."""
    ts, labels, _ = generate_dataset(24, seed=1)
    table = build_feature_table(as_loaded(ts))
    lab = typed_labels(labels).set_index("customer_id")
    pv = {c: d.date() for c, d in lab.loc[lab["pv_installed"], "pv_installation_date"].items() if pd.notna(d)}
    clim = fit_climatology(table.dailies(), lab.index[~lab["pv_installed"]].tolist(), pv)
    head = weekly.fit_head(table, lab, lab.index.tolist(), clim)
    assert head["features"] == list(weekly.WEEK_FEATURES) and head["aggregation"] == {"window": 4}
    late = CustomerSpec("900001", "2023-01-01", "2025-06-30", pv=PVSpec("2025-04-21", 6.0))
    vacation = CustomerSpec("900002", "2023-01-01", "2025-06-30", events=(Event("vacation", "2024-07-08", "2024-07-29"),))
    new = build_feature_table(as_loaded(pd.concat([generate_customer(late, 501), generate_customer(vacation, 502)])))
    s_late, s_vac = weekly.customer_scores(head, new, ["900001", "900002"], clim)
    assert s_late > 0.5 and s_vac < 0.5


# ---------------------------------------------------------------- selection


def _flawed(t_pv: np.ndarray, rows: slice) -> np.ndarray:
    """Confident truth, except positives and negatives at positions `rows` (per class) are misranked."""
    p = np.where(t_pv, 0.95, 0.05)
    pos, neg = np.flatnonzero(t_pv), np.flatnonzero(~t_pv)
    p[pos[rows]], p[neg[rows]] = 0.30, 0.70
    return p


A, B = slice(0, 3), slice(3, 6)


@pytest.mark.parametrize(
    "p_weekly_rows, expect",
    [(None, "weekly"), (B, "avg_weekly"), (slice(0, 6), "logreg")],
)
def test_weekly_candidates_follow_the_margin_rule(p_weekly_rows, expect):
    t = _truth()
    t_pv = t["pv_installed"].to_numpy(bool)
    p_lr = _flawed(t_pv, A)
    oof = _oof(t, p_lr, p_lr)
    oof.insert(2, "p_weekly", np.where(t_pv, 0.95, 0.05) if p_weekly_rows is None else _flawed(t_pv, p_weekly_rows))
    sel = _select(oof)
    assert sel["classifier"] == expect
    winner = sel[f"inner_score_{expect}"]
    if expect == "logreg":
        assert sel["bootstrap_p_weekly"] is None and sel["inner_score_weekly"] < sel["inner_score_logreg"]
    else:
        assert winner == pytest.approx(100.0) and sel["inner_score"] == pytest.approx(100.0)
        assert winner - sel["inner_score_logreg"] >= procedure.CLASSIFIER_MARGIN
        assert sel[f"bootstrap_p_{expect}"] >= procedure.BOOTSTRAP_MIN
    if expect == "avg_weekly":
        assert sel["bootstrap_p_weekly"] is None  # weekly alone is no better than logreg
    # without weekly scores the record is the pre-weekly one, key for key
    assert not NEW_KEYS & set(_select(oof.drop(columns="p_weekly")))


@pytest.mark.parametrize("better", [0.1, 2.0, None])
def test_logreg_c_follows_the_margin_rule(better):
    t = _truth()
    t_pv = t["pv_installed"].to_numpy(bool)
    good, bad = np.where(t_pv, 0.95, 0.05), _flawed(t_pv, A)
    oof = _oof(t, bad, bad)
    oof["p_logreg_C0.1"] = good if better == 0.1 else bad
    oof["p_logreg_C2.0"] = good if better == 2.0 else bad
    C, oof2, sel, info = _choose_logreg_c(oof, _select(oof))
    assert "p_logreg_C0.1" not in oof2 and "p_logreg_C2.0" not in oof2
    opts = info["logreg_C_options"]
    assert set(opts) == {"0.1", "0.5", "2.0", "bootstrap_p"}
    if better is None:
        assert C == info["logreg_C"] == 0.5 and opts["bootstrap_p"] == {"0.1": None, "2.0": None}
        pdt.assert_series_equal(oof2["p_logreg"], oof["p_logreg"])
    else:
        assert C == info["logreg_C"] == better
        assert opts[str(better)] - opts["0.5"] >= procedure.CLASSIFIER_MARGIN
        assert opts["bootstrap_p"][str(better)] >= procedure.BOOTSTRAP_MIN
        np.testing.assert_array_equal(oof2["p_logreg"].to_numpy(), good)
        assert sel["classifier"] == "logreg" and sel["inner_score"] == pytest.approx(100.0)


def test_selection_line_reports_weekly_and_logreg_c():
    common = {"capacity": "ridge", "threshold": 0.5, "kappa": 1.0, "offset_event": 0}
    infos = [
        {**common, "classifier": "weekly", "inner_score_weekly": 90.0, "logreg_C": 0.5},
        {**common, "classifier": "avg_weekly", "inner_score_weekly": 80.0, "logreg_C": 2.0},
        {**common, "classifier": "logreg", "inner_score_weekly": 70.0, "logreg_C": 0.5},
    ]
    line = train._selection_line(infos)
    assert "classifier logreg=1/3 catboost=0/3 avg=0/3 weekly=1/3 avg_weekly=1/3 | logreg_C 0.1=0/3 0.5=2/3 2.0=1/3" in line
    plain = train._selection_line([{**common, "classifier": "logreg"}])
    assert "weekly" not in plain and "logreg_C" not in plain


# ----------------------------------------------------------- fit with stubs


@pytest.fixture(scope="module")
def data():
    ts, labels, _ = generate_dataset(12, seed=4)
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


def _stub(prefer: str | None = None, better_c: float | None = None, calls: list | None = None):
    """Inner OOF frames: logreg is perfect only at C == `better_c`; `p_weekly` (when asked for) is perfect
    for prefer="weekly", complementary to logreg for "avg_weekly", else as flawed as logreg."""

    def stub(
        table, labels, lab, train_ids, splits, names, cap_names, setting=procedure.BASE_SETTING, refit=None,
        *, C=procedure.LOGREG_C, alt_cs=(), weekly=False,
    ):
        if calls is not None:
            calls.append({"C": C, "alt_cs": tuple(alt_cs), "weekly": weekly, "n_splits": len(splits)})
        t = lab.loc[train_ids]
        t_pv = t["pv_installed"].to_numpy(bool)
        good, bad = np.where(t_pv, 0.95, 0.05), _flawed(t_pv, slice(0, 2))
        p_lr = good if C == better_c else bad
        oof = _oof(t, p_lr, bad)
        for c in alt_cs:
            oof[f"p_logreg_C{c}"] = good if c == better_c else bad
        if weekly:
            oof["p_weekly"] = {"weekly": good, "avg_weekly": _flawed(t_pv, slice(2, 4))}.get(prefer, bad)
        return oof

    return stub


@pytest.mark.parametrize("prefer", ["weekly", "avg_weekly", None])
def test_fit_weekly_choice_saves_loads_and_predicts(data, monkeypatch, tmp_path, prefer):
    table, labels = data
    calls = []
    monkeypatch.setattr(procedure, "_inner_oof", _stub(prefer, calls=calls))
    ids = labels["customer_id"].tolist()
    model = fit(table, labels, ids)
    assert all(c["weekly"] for c in calls) and calls[0]["alt_cs"] == (0.1, 2.0)
    assert len({c["n_splits"] for c in calls}) == 1  # every option on the same inner splits
    clf, sel = model["classifier"], model["selection"]
    assert clf["kind"] == sel["classifier"] == (prefer or "logreg")
    assert sel["logreg_C"] == 0.5 and {"inner_score_weekly", "bootstrap_p_avg_weekly"} <= set(sel)
    assert ("weekly" in clf) == (prefer is not None)
    assert (clf["logreg"] is not None) == (prefer != "weekly")
    save_model(model, tmp_path / "model.json")
    loaded = load_model(tmp_path / "model.json")
    assert loaded["classifier"] == json.loads(json.dumps(clf))
    pred, _ = predict(loaded, table, ids)
    assert len(pred) == len(ids)
    if prefer is not None:
        head = loaded["classifier"]["weekly"]
        assert head["aggregation"] == {"window": weekly.WINDOW} and head["features"] == list(weekly.WEEK_FEATURES)
        clim = Climatology.from_json(loaded["climatology"])
        s = weekly.customer_scores(head, table, ids, clim)
        if prefer == "avg_weekly":
            F, _ = pipeline.compute_features(table, ids, clim)
            s = (pipeline.classifier_proba(loaded["classifier"], {}, F, loaded["feature_names"], "logreg") + s) / 2
        np.testing.assert_allclose(pred["prob"].to_numpy(float), s, rtol=0, atol=1e-12)


@pytest.mark.parametrize("better", [0.1, 2.0, None])
def test_fit_logreg_c_choice_reaches_later_options_and_the_final_fit(data, monkeypatch, better):
    table, labels = data
    calls, fitted_c = [], []
    monkeypatch.setattr(procedure, "_inner_oof", _stub(better_c=better, calls=calls))
    real_fit_logreg = procedure.fit_logreg

    def spy(X, y, C, **kw):
        fitted_c.append(C)
        return real_fit_logreg(X, y, C, **kw)

    monkeypatch.setattr(procedure, "fit_logreg", spy)
    model = fit(table, labels, labels["customer_id"].tolist(), options=frozenset({"e1", "ctune"}))
    expect = better or procedure.LOGREG_C
    assert model["selection"]["logreg_C"] == expect and model["classifier"]["kind"] == "logreg"
    assert [c["C"] for c in calls] == [procedure.LOGREG_C, expect]  # stage 1, then E1 at the chosen C
    assert [c["alt_cs"] for c in calls] == [(0.1, 2.0), ()]
    assert fitted_c == [expect]  # the final logreg


def test_inner_weekly_head_never_scores_customers_it_trained_on(data, monkeypatch):
    table, labels = data
    lab = labels.set_index("customer_id")
    ids = labels["customer_id"].tolist()
    trained, scored = [], []
    real_fit, real_scores = procedure.fit_head, procedure.customer_scores
    monkeypatch.setattr(procedure, "fit_head", lambda t, l, tr, c: trained.append(list(tr)) or real_fit(t, l, tr, c))
    monkeypatch.setattr(
        procedure, "customer_scores", lambda h, t, va, c: scored.append(list(va)) or real_scores(h, t, va, c)
    )
    splits = procedure.inner_splits(labels, ids)
    export = procedure.export_enabled_for(table, ids)
    names, cap_names = procedure.feature_names(export), procedure.capacity_feature_names(export)
    oof = procedure._inner_oof(table, labels, lab, ids, splits, names, cap_names, weekly=True)
    assert len(trained) == len(scored) == len(splits)
    for (tr, va), a, b in zip(splits, trained, scored):
        assert a == tr and b == va and not set(a) & set(b)
    assert oof["p_weekly"].between(0, 1).all()


def test_pseudo_labeller_of_weekly_kind_scores_the_pool_with_the_weekly_head(data, monkeypatch):
    table, labels = data
    ts_u, _, _ = generate_dataset(6, seed=7)
    ts_u["customer_id"] = "PODSYN" + ts_u["customer_id"].astype(str)
    pooled = pipeline.FeatureTable(
        table.ids, table.customers, unlabelled_pool=build_feature_table(as_loaded(ts_u)).customers
    )
    lab = labels.set_index("customer_id")
    ids = labels["customer_id"].tolist()
    export = procedure.export_enabled_for(table, ids)
    names, cap_names = procedure.feature_names(export), procedure.capacity_feature_names(export)
    comp = procedure._fit_components(pooled, lab, ids, names, cap_names, ("logreg", "catboost"), ("ridge",), C=2.0)
    seen = {}
    real = pseudolabel.classifier_proba

    def spy(clf, objects, F, names_, kind, weekly=None):
        seen.update(kind=kind, weekly=weekly)
        return real(clf, objects, F, names_, kind, weekly=weekly)

    monkeypatch.setattr(pseudolabel, "classifier_proba", spy)
    pseudolabel.with_pseudo(pooled, lab, ids, names, comp, "avg_weekly")
    view = pseudolabel._pool_view(pooled)
    expect = weekly.customer_scores(weekly.fit_head(pooled, lab, ids, comp.clim), view, view.ids, comp.clim)
    assert seen["kind"] == "avg_weekly"
    np.testing.assert_array_equal(seen["weekly"], expect)
    assert set(pooled.cache["unlabelled_pool"]["weekly"]) == set(view.ids)  # pool windows cached apart
    assert not set(view.ids) & set(pooled.cache["weekly"])


# ----------------------------------------------------- switches and invariants


def test_without_weekly_and_ctune_no_new_code_path_runs(data, monkeypatch):
    table, labels = data

    def boom(*a, **k):
        raise AssertionError("weekly/C code path used")

    for mod, name in (
        (procedure, "fit_head"),
        (procedure, "customer_scores"),
        (procedure, "_choose_logreg_c"),
        (pipeline, "customer_scores"),
    ):
        monkeypatch.setattr(mod, name, boom)
    ids = labels["customer_id"].tolist()
    model = fit(table, labels, ids, options=procedure.DEFAULT_OPTIONS - {"weekly", "ctune"})
    assert not NEW_KEYS & set(model["selection"]) and "weekly" not in model["classifier"]
    assert model["classifier"]["kind"] in ("logreg", "catboost", "avg")
    predict(model, table, ids)


def test_train_switches_drop_the_options(syn_train, tmp_path, monkeypatch):
    seen = []
    real_fit = train.fit

    def spy(*a, options, **k):
        seen.append(options)
        return real_fit(*a, options=options, **k)

    monkeypatch.setattr(train, "fit", spy)
    argv = ["--timeseries", str(syn_train / "energy.csv"), "--labels", str(syn_train / "labels.csv")]
    assert train.main(argv + ["--out", str(tmp_path / "a" / "model.json"), "--no-weekly", "--no-ctune"]) == 0
    assert train.main(argv + ["--out", str(tmp_path / "b" / "model.json"), "--no-ctune"]) == 0
    assert seen == [procedure.DEFAULT_OPTIONS - {"weekly", "ctune"}, procedure.DEFAULT_OPTIONS - {"ctune"}]
    sel = json.loads((tmp_path / "a" / "model.json").read_text())["selection"]
    assert not NEW_KEYS & set(sel)


def _flip(labels: pd.DataFrame, test_ids: list[str]) -> pd.DataFrame:
    flipped = labels.copy()
    for x in test_ids:
        row = flipped["customer_id"] == x
        was_pv = bool(flipped.loc[row, "pv_installed"].iloc[0])
        flipped.loc[row, "pv_installed"] = not was_pv
        flipped.loc[row, "pv_installation_date"] = pd.NaT if was_pv else pd.Timestamp("2023-05-01")
        flipped.loc[row, "pv_capacity_kwp"] = np.nan if was_pv else 7.5
    return flipped


def _force_kind(monkeypatch, kind: str) -> None:
    """Every inner selection that has weekly scores picks `kind` (the weekly option forced on)."""
    real = procedure._select

    def forced(oof, avg=True):
        sel = real(oof, avg)
        if "p_weekly" in oof.columns:
            sel["classifier"] = kind
        return sel

    monkeypatch.setattr(procedure, "_select", forced)


@pytest.mark.parametrize("kind", ["weekly", "avg_weekly"])
def test_label_flip_invariant_and_determinism_with_weekly_forced(data, monkeypatch, tmp_path, kind):
    table, labels = data
    _force_kind(monkeypatch, kind)
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    m_a = fit(table, labels, train_ids)
    m_b = fit(table, _flip(labels, test_ids), train_ids)
    assert m_a["classifier"]["kind"] == kind and "weekly" in m_a["classifier"]
    pdt.assert_frame_equal(predict(m_a, table, test_ids)[0], predict(m_b, table, test_ids)[0])
    save_model(m_a, tmp_path / "a" / "model.json")
    save_model(fit(table, labels, train_ids), tmp_path / "b" / "model.json")
    assert (tmp_path / "a" / "model.json").read_text() == (tmp_path / "b" / "model.json").read_text()


def _no_network(*args, **kwargs):
    raise RuntimeError("network blocked")


def test_offline_inference_with_a_weekly_model(syn_train, syn_test, tmp_path, monkeypatch):
    table = build_feature_table(load_timeseries(syn_train / "energy.csv"))
    labels = load_labels(syn_train / "labels.csv")
    _force_kind(monkeypatch, "weekly")
    model = fit(table, labels, labels["customer_id"].tolist())
    assert model["classifier"]["kind"] == "weekly"
    save_model(model, tmp_path / "model" / "model.json")
    monkeypatch.setattr(inference, "MODEL_PATH", tmp_path / "model" / "model.json")
    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.chdir(tmp_path)
    assert inference.main([str(syn_test / "energy.csv")]) == 0
    lines = (tmp_path / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "customer_id;pv_installed;pv_installation_date;pv_capacity_kwp" and len(lines) == 13
