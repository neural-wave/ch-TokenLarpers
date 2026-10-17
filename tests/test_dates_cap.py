"""Dates/capacity round: κ on the final TPs, score-matched offsets (`offset_tri`), the start-lag
fallback, the short-series recurrence rule (`recur_strict`) and the dev generator scenarios."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pvdetect import pipeline, procedure
from pvdetect.changepoint import (
    MIN_PERSIST,
    MIN_PERSIST_SHORT,
    Commissioning,
    detect_commissioning,
    shrunk_median,
    triangle_offset,
)
from pvdetect.model import load_model, save_model
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.procedure import _select, fit, fit_date_params
from pvdetect.season import zero_climatology
from pvdetect.selection import choose_kappa
from pvdetect.synthetic import CustomerSpec, PVSpec, generate_customer, generate_dataset
from tests.helpers import as_loaded, customer_daily
from tests.test_evaluation import typed_labels
from tests.test_model import _inner_oof as b4_oof

LABEL = pd.Timestamp("2022-06-15")


# ------------------------------------------------------------- κ on final TPs


def test_kappa_follows_the_tp_set_of_the_final_threshold():
    """Ridge 1.15× true on the confident positives, exact on the low-probability one. At t=0.40 the TPs
    {a, b, c} give κ=0.90; B4 moves to t=0.50, whose TPs {a, b} give κ=0.85, and the score uses it."""
    oof = b4_oof(1.15, 1.0)
    sel = _select(oof)
    assert (sel["threshold_initial"], sel["threshold"]) == (0.40, 0.50)
    assert sel["kappa_initial"] == 0.90
    tp = oof["t_pv"].to_numpy(bool) & (oof["p_logreg"].to_numpy() >= 0.50)
    assert sel["kappa"] == choose_kappa(oof["kwp_ridge"].to_numpy()[tp], oof["t_kwp"].to_numpy()[tp]) == 0.85
    p = 1 - abs(1.15 * 0.85 - 1) / 0.5  # both TPs; dates 40 days off score 0
    assert sel["inner_score"] == pytest.approx(2 * 2 * (70 + 15 * p) / (2 + 3))


def test_kappa_unchanged_when_the_tp_set_is():
    sel = _select(b4_oof(1.5, 1.0))
    assert sel["kappa"] == sel["kappa_initial"] == 0.80


# ------------------------------------------------------ score-matched offsets


def test_triangle_optimum_ignores_spread_majority():
    errs = [0, 0, 0, 0, 40, 70, 100, 130, 160]  # a tight minority cluster plus far, spread errors
    assert shrunk_median(errs, -120, 120) == 26  # round(40 · 9/14)
    assert triangle_offset(errs, -120, 120) == 0


def test_triangle_ties_go_to_the_shrunk_median_then_smaller_magnitude():
    assert triangle_offset([0, 20], -120, 120) == 3  # flat on [0, 20]; shrunk median round(10 · 2/7)
    assert triangle_offset([-10, 10], -120, 120) == 0
    assert triangle_offset([], -120, 120) == 0
    assert triangle_offset([130], -120, 120) == 120  # the optimum inside the clip range
    assert triangle_offset([200], -120, 120) == 33  # no x in range scores: the shrunk median round(200/6)


def _params(errors: list[int], gaps: list[int], tri: bool) -> dict:
    """fit_date_params on a stub table: `errors` from found events, `gaps` from present-at-start customers."""
    ids = [f"SYN{i:03d}" for i in range(len(errors) + len(gaps))]
    cds, cps = {}, {}
    for i, c in enumerate(ids):
        gap = gaps[i - len(errors)] if i >= len(errors) else -400
        first = (LABEL + pd.Timedelta(days=gap)).date()
        days = pd.date_range(first, periods=30, freq="D")
        cds[c] = SimpleNamespace(
            first_date=first, n_valid=len(days), daily=pd.DataFrame({"valid": np.ones(len(days), bool)}, index=days)
        )
        if i < len(errors):
            d = (LABEL - pd.Timedelta(days=errors[i])).date()
            cps[c] = Commissioning(d, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, None, None)
        else:
            cps[c] = None
    lab = pd.DataFrame({"pv_installed": True, "pv_installation_date": LABEL}, index=pd.Index(ids, name="customer_id"))
    F = pd.DataFrame({"seasonal_contrast": 0.0}, index=ids)
    return fit_date_params(SimpleNamespace(customers=cds), lab, ids, F, cps, tri=tri)


def test_offset_tri_through_fit_date_params():
    errs = [0, 0, 0, 0, 40, 70, 100, 130, 160]
    assert _params(errs, [], tri=False)["offset_event"] == 26
    assert _params(errs, [], tri=True)["offset_event"] == 0
    assert _params([90] * 10, [], tri=True)["offset_event"] == 90  # no shrinkage on the score optimum


@pytest.mark.parametrize(
    "gaps, off, on",
    [
        ([60], 0, 10),  # round(60 · 1/6)
        ([60, 60], 0, 17),  # round(60 · 2/7)
        ([30, 40, 50], 15, 40),  # n ≥ 3: shrunk median vs the unshrunk triangle optimum
    ],
)
def test_start_lag_fallback(gaps, off, on):
    assert _params([], gaps, tri=False)["start_lag"] == off
    assert _params([], gaps, tri=True)["start_lag"] == on


# --------------------------------------------------- short-series recurrence


def _short_daily(low_until: str | None) -> pd.DataFrame:
    """2023-01-01..2023-10-31 (no ±1-year window inside), s = 1 + noise, 0.7 from May 1 to `low_until`."""
    idx = pd.date_range("2023-01-01", "2023-10-31", freq="D")
    s = 1.0 + np.random.default_rng(0).normal(0, 0.02, idx.size)
    if low_until is not None:
        s[(idx >= "2023-05-01") & (idx < low_until)] -= 0.3
    woy = np.minimum(idx.isocalendar().week.to_numpy().astype(int), 52)
    return pd.DataFrame({"valid": True, "woy": woy, "s": s}, index=idx)


def test_short_seasonal_looking_step_needs_strict_persistence():
    """A May–August drop (persistence 4/6 months) with no recurrence window: found by the default rule,
    not by `recur_strict`."""
    daily = _short_daily("2023-08-25")
    loose = detect_commissioning(daily, zero_climatology())
    strict = detect_commissioning(daily, zero_climatology(), recur_strict=True)
    assert np.isnan(loose.recur) and loose.found
    assert MIN_PERSIST <= loose.persist < MIN_PERSIST_SHORT
    assert not strict.found and strict.weak_date is not None


def test_short_real_install_stays_found_under_strict():
    spec = CustomerSpec("000001", "2023-01-01", "2023-10-31", pv=PVSpec("2023-04-15", 8.0))
    daily = customer_daily(spec, 11)
    loose = detect_commissioning(daily, zero_climatology())
    strict = detect_commissioning(daily, zero_climatology(), recur_strict=True)
    assert np.isnan(strict.recur) and strict.persist >= MIN_PERSIST_SHORT
    assert strict.found and loose.found and strict.date == loose.date


def test_strict_rule_ignores_series_with_a_recurrence_window():
    spec = CustomerSpec("000001", "2022-01-01", "2025-12-31", pv=PVSpec("2023-06-10", 8.0))
    daily = customer_daily(spec, 11)
    loose = detect_commissioning(daily, zero_climatology())
    strict = detect_commissioning(daily, zero_climatology(), recur_strict=True)
    assert np.isfinite(loose.recur) and loose.found
    assert (strict.found, strict.date, strict.evidence) == (loose.found, loose.date, loose.evidence)


# --------------------------------------------------------- inner-CV choices


@pytest.fixture(scope="module")
def data():
    ts, labels, _ = generate_dataset(12, seed=4)
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


def _oof(t: pd.DataFrame, good: bool, tri_good: bool | None) -> pd.DataFrame:
    t_pv = t["pv_installed"].to_numpy(bool)
    true_dates = pd.Series(t["pv_installation_date"].to_numpy(), index=t.index)
    dates = true_dates + pd.Timedelta(days=0 if good else 90)
    oof = pd.DataFrame(
        {
            "p_logreg": t_pv.astype(float),
            "p_catboost": t_pv.astype(float),
            "kwp_ridge": np.where(t_pv, t["pv_capacity_kwp"].to_numpy(float), 5.0),
            "kwp_catboost": np.nan,
            "regime": "event",
            "date_weak": dates,
            "date_first": dates,
        },
        index=t.index,
    )
    if tri_good is not None:
        tri = true_dates + pd.Timedelta(days=0 if tri_good else 90)
        oof["date_weak_tri"], oof["date_first_tri"] = tri, tri
    oof["t_pv"] = t_pv
    oof["t_date"] = t["pv_installation_date"].to_numpy()
    oof["t_kwp"] = t["pv_capacity_kwp"].to_numpy(float)
    return oof


def _stub(prefer: str | None, calls: list):
    """Inner OOF whose dates are exact only under the preferred date option (`offset_tri`/`recur_strict`)."""

    def stub(
        table, labels, lab, train_ids, splits, names, cap_names, setting=procedure.BASE_SETTING, refit=None,
        *, C=procedure.LOGREG_C, alt_cs=(), weekly=False, offset_tri=False, recur_strict=False, tri_columns=False,
    ):
        calls.append({"offset_tri": offset_tri, "recur_strict": recur_strict, "tri_columns": tri_columns, "n": len(splits)})
        on = {"offset_tri": offset_tri, "recur_strict": recur_strict}
        good = prefer is not None and on[prefer]
        tri_good = prefer == "offset_tri" if tri_columns else None
        return _oof(lab.loc[train_ids], good, tri_good)

    return stub


DATE_OPTIONS = frozenset({"offset_tri", "recur_strict"})


@pytest.mark.parametrize("prefer", ["offset_tri", "recur_strict", None])
def test_date_options_follow_inner_cv(data, monkeypatch, tmp_path, prefer):
    table, labels = data
    calls: list = []
    monkeypatch.setattr(procedure, "_inner_oof", _stub(prefer, calls))
    ids = labels["customer_id"].tolist()
    model = fit(table, labels, ids, options=DATE_OPTIONS)
    sel = model["selection"]
    assert [c["tri_columns"] for c in calls] == [True, False]  # tri dates ride on the first inner OOF
    assert calls[1]["recur_strict"] and calls[1]["offset_tri"] == (prefer == "offset_tri")
    assert len({c["n"] for c in calls}) == 1  # same inner splits
    for key in DATE_OPTIONS:
        opts = sel[f"{key}_options"]
        if key == prefer:
            assert sel[key] == "on"
            assert opts["on"] - opts["off"] >= procedure.CLASSIFIER_MARGIN and opts["bootstrap_p"] >= procedure.BOOTSTRAP_MIN
        else:
            assert sel[key] == "off" and opts["bootstrap_p"] is None
    assert model["date"]["recur_strict"] is (prefer == "recur_strict")
    seen = []
    real = pipeline.compute_features

    def spy(*a, recur_strict=False, **k):
        seen.append(recur_strict)
        return real(*a, recur_strict=recur_strict, **k)

    monkeypatch.setattr(pipeline, "compute_features", spy)
    save_model(model, tmp_path / "model.json")
    predict(load_model(tmp_path / "model.json"), table, ids)
    assert seen == [prefer == "recur_strict"]  # inference detects with the fit's rule


def test_offset_tri_adopted_uses_triangle_offsets_in_the_final_fit(data, monkeypatch):
    table, labels = data
    monkeypatch.setattr(procedure, "_inner_oof", _stub("offset_tri", []))
    ids = labels["customer_id"].tolist()
    model = fit(table, labels, ids, options=frozenset({"offset_tri"}))
    comp = procedure._fit_components(table, labels.set_index("customer_id"), ids, model["feature_names"],
                                     model["capacity_feature_names"], ("logreg",), ("ridge",), offset_tri=True)
    assert {k: model["date"][k] for k in ("offset_event", "start_lag")} == {
        k: comp.date_params[k] for k in ("offset_event", "start_lag")
    }
    assert "recur_strict" not in model["date"] and "recur_strict" not in model["selection"]


def test_without_date_options_no_new_keys_or_kwargs(data, monkeypatch):
    table, labels = data
    calls: list = []
    monkeypatch.setattr(procedure, "_inner_oof", _stub(None, calls))
    model = fit(table, labels, labels["customer_id"].tolist(), options=frozenset())
    assert calls == [{"offset_tri": False, "recur_strict": False, "tri_columns": False, "n": calls[0]["n"]}]
    assert not {"offset_tri", "offset_tri_options", "recur_strict", "recur_strict_options"} & set(model["selection"])
    assert "recur_strict" not in model["date"]


# ------------------------------------------------------------ dev scenarios


def test_label_lag_60_lags_only_the_labels():
    ts_b, lab_b, _ = generate_dataset(6, 3)
    ts_l, lab_l, _ = generate_dataset(6, 3, scenario="label_lag_60")
    pd.testing.assert_frame_equal(ts_b, ts_l)
    pv = lab_b["pv_installed"] == "TRUE"
    assert pv.any()
    lag = (pd.to_datetime(lab_l.loc[pv, "pv_installation_date"]) - pd.to_datetime(lab_b.loc[pv, "pv_installation_date"])).dt.days
    assert lag.between(30, 90).all()
    pd.testing.assert_frame_equal(lab_b[~pv], lab_l[~pv])
    pd.testing.assert_frame_equal(lab_b.drop(columns="pv_installation_date"), lab_l.drop(columns="pv_installation_date"))


def test_meter_spikes_are_isolated_single_slots():
    base = CustomerSpec("000001", "2022-01-01", "2023-12-31", pv=PVSpec("2022-06-01", 6.0), missing=False)
    plain = generate_customer(base, 5)
    spiky = generate_customer(CustomerSpec(**{**base.__dict__, "spikes_per_year": 3}), 5)
    e0, e1 = plain["energy_consumption_kwh"].to_numpy(), spiky["energy_consumption_kwh"].to_numpy()
    changed = np.flatnonzero(e0 != e1)
    assert changed.size == round(3 * 730 / 365)
    assert np.diff(changed).min() > 1  # isolated
    ratio = e1[changed] / np.median(e0[e0 > 0])
    assert ((ratio > 20 * 0.99) & (ratio < 50 * 1.01)).all()


def test_meter_spikes_scenario_and_variant_guard():
    ts_b, lab_b, _ = generate_dataset(4, 3)
    ts_s, lab_s, _ = generate_dataset(4, 3, scenario="meter_spikes")
    pd.testing.assert_frame_equal(lab_b, lab_s)
    assert (ts_b[["customer_id", "timestamp_epoch"]].to_numpy() == ts_s[["customer_id", "timestamp_epoch"]].to_numpy()).all()
    diff = ts_b["energy_consumption_kwh"].to_numpy() != ts_s["energy_consumption_kwh"].to_numpy()
    assert 0 < diff.sum() <= 4 * round(3 * 1461 / 365)
    for sc in ("label_lag_60", "meter_spikes", "shared_weather"):
        with pytest.raises(ValueError, match="variant A only"):
            generate_dataset(2, 3, variant="B", scenario=sc)


def test_holdout_default_scenarios_are_unchanged(monkeypatch, tmp_path):
    from pvdetect import benchmark
    from pvdetect.benchmark import Timing

    ran = []

    def fake(sc, n, repeats, seed, variant):
        ran.append((sc, seed, variant))
        return {"scenario": sc, "n": n, "repeats": repeats, "score": 0.0, "f1": 0.0, "D": 0.0, "P": 0.0, "regime_D": ""}, Timing(0.0, 0.0, 1.0)

    monkeypatch.setattr(benchmark, "run_scenario", fake)
    assert benchmark.main(["--suite", "holdout", "--log", str(tmp_path / "log.md"), "--n", "12", "--repeats", "1"]) == 0
    block = int((benchmark.REPO / "benchmarks" / "HOLDOUT_SEED_BLOCK").read_text().strip())
    held = [r for r in ran if r[2] == "B"]
    assert [s for s, _, _ in held] == [
        "base",
        "confounders_x2",
        "east_west_heavy",
        "battery_heavy",
        "short_coverage",
        "missing_5pct",
        "commercial_heavy",
        "pv_before_start",
    ]
    assert [seed for _, seed, _ in held] == [block + j for j in range(len(held))]
