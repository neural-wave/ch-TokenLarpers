from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from pvdetect.features import SUN_MIN_DAYS, sun_coupling
from pvdetect.pipeline import build_feature_table, compute_features, sun_anomaly
from pvdetect.season import Climatology, SunIndex, fit_climatology, fit_sun_index, sun_residual, zero_climatology
from pvdetect.synthetic import _dataset_specs, generate_dataset
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels

DAYS = pd.date_range("2023-01-01", "2024-12-31", freq="D")


def _series(values, index=DAYS) -> pd.Series:
    return pd.Series(np.asarray(values, float), index=index)


def test_sun_index_trims_extremes_of_post_install_contributors_and_excludes_each_from_its_own():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    base = np.linspace(-1, 1, 40)
    res = {k: _series(base + j, idx) for j, k in enumerate("abcde")}
    res["x"] = _series(np.full(40, 100.0), idx)  # an extreme contributor
    res["late"] = _series(np.full(40, -100.0), idx)  # installed after the window: never contributes
    pv = {k: dt.date(2023, 6, 1) for k in "abcdex"} | {"late": dt.date(2024, 3, 1)}
    sun = fit_sun_index(res, pv)
    # a..e and x: lowest (a) and highest (x) dropped, mean of b..e
    np.testing.assert_allclose(sun.values, base + 2.5)
    # x's own index: a..e, a and e dropped
    np.testing.assert_allclose(sun.series("x").to_numpy(), base + 2)
    # a's own index: b..e and x, b and x dropped
    np.testing.assert_allclose(sun.series("a").to_numpy(), base + 3)
    # a non-contributor gets the full index
    np.testing.assert_allclose(sun.series("zzz").to_numpy(), sun.values)


def test_sun_index_needs_five_contributors_per_day_and_waits_two_weeks_after_installation():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    res = {k: _series(np.arange(30.0) + j, idx) for j, k in enumerate("abcde")}
    late = {k: dt.date(2023, 1, 1) for k in "abcd"} | {"e": dt.date(2024, 1, 10)}
    sun = fit_sun_index(res, late)
    days = sun.series().index
    # e contributes from 2024-01-24 (installation + 14 days); before that only four contributors
    assert np.isnan(sun.series()[days < "2024-01-24"]).all()
    assert np.isfinite(sun.series()[days >= "2024-01-24"]).all()
    assert fit_sun_index({k: res[k] for k in "abcd"}, late) is None


def test_sun_coupling_sign_separates_pv_from_sun_driven_and_weather_free_loads():
    rng = np.random.default_rng(0)
    sun = _series(rng.normal(0, 1, len(DAYS)))
    noise = rng.normal(0, 0.3, len(DAYS))
    pv_like = sun.to_numpy() + noise  # midday load falls on the days the index customers' load falls
    cooling = -sun.to_numpy() + noise  # midday load rises on sunny days
    assert sun_coupling(_series(pv_like), sun) > 0.9
    assert sun_coupling(_series(cooling), sun) < -0.9
    assert abs(sun_coupling(_series(rng.normal(0, 1, len(DAYS))), sun)) < 0.2


def test_sun_coupling_uses_only_summer_days_of_the_last_year_and_needs_enough_of_them():
    rng = np.random.default_rng(1)
    sun = _series(rng.normal(0, 1, len(DAYS)))
    z = sun.copy()
    z[z.index < "2024-01-01"] = -z[z.index < "2024-01-01"]  # opposite sign in the older year
    winter = ~z.index.month.isin([4, 5, 6, 7, 8, 9])
    z[winter] = -z[winter]  # and in winter
    assert sun_coupling(z, sun) == pytest.approx(1.0)
    short = sun[(sun.index >= "2024-06-01")][: SUN_MIN_DAYS - 1]
    assert np.isnan(sun_coupling(short, sun))


def test_sun_residual_keeps_daily_weather_and_removes_slow_load_changes():
    rng = np.random.default_rng(2)
    weather = rng.normal(0, 1, len(DAYS))
    season = 2 * np.cos(2 * np.pi * DAYS.dayofyear.to_numpy() / 365)
    step = np.where(DAYS >= "2024-03-01", 3.0, 0.0)  # schedule change
    weekend = DAYS.dayofweek.to_numpy() >= 5
    mid = 10 + season + step + np.where(weekend, -4.0, 0.0) + 0.5 * weather
    daily = pd.DataFrame({"mid": mid, "valid": True, "weekend": weekend}, index=DAYS)
    z = sun_residual(daily)
    w = pd.Series(weather, index=DAYS).reindex(z.index)
    assert np.corrcoef(z, w)[0, 1] > 0.85
    after = z[(z.index >= "2024-03-15") & (z.index < "2024-06-01")]
    assert abs(after.median()) < 0.3  # the step is absorbed within about two weeks


def test_climatology_json_keeps_the_index_and_drops_leave_one_out_copies():
    sun = SunIndex(pd.Timestamp("2024-01-01"), np.array([0.5, np.nan, -1.0]), {"a": np.zeros(3)})
    clim = Climatology({}, np.zeros(52), sun=sun)
    back = Climatology.from_json(json.loads(json.dumps(clim.to_json())))
    np.testing.assert_array_equal(back.sun.values, sun.values)
    assert back.sun.start == sun.start and back.sun.loo == {}
    np.testing.assert_array_equal(back.sun.series("a").to_numpy(), sun.values)
    assert "sun" not in zero_climatology().to_json()


@pytest.fixture(scope="module")
def shared_weather():
    ts, labels, _ = generate_dataset(36, 5, scenario="shared_weather")
    specs = _dataset_specs(36, 5, "2022-01-01", "2025-12-31", True, False, "A", "shared_weather")
    return build_feature_table(as_loaded(ts)), typed_labels(labels), specs


def test_shared_weather_scenario_pv_couples_positively_and_mimicking_loads_do_not(shared_weather):
    table, labels, specs = shared_weather
    lab = labels.set_index("customer_id")
    pv = {c: d.date() for c, d in lab.loc[lab["pv_installed"], "pv_installation_date"].items()}
    clim = fit_climatology(table.dailies(), lab.index[~lab["pv_installed"]].tolist(), pv)
    sun = fit_sun_index({c: sun_anomaly(table, c) for c in pv}, pv)
    F, _ = compute_features(table, table.ids, Climatology(clim.curves, clim.yield_shape, sun=sun))
    kind = {s.customer_id: "pv" if s.pv else (s.events[0].kind if s.events else "plain") for s in specs}
    by = pd.Series(F["sun_coupling"].to_numpy(), index=F.index).groupby(pd.Series(kind)).median()
    assert by["pv"] > 0.3
    assert by["ac"] < -0.1
    assert abs(by["summer_absence"]) < 0.15


def test_an_adopted_sun_block_is_saved_with_its_index_and_used_at_inference(shared_weather, monkeypatch, tmp_path):
    from pvdetect import procedure
    from pvdetect.model import load_model, save_model
    from pvdetect.pipeline import classifier_proba, predict

    table, labels, _ = shared_weather
    monkeypatch.setattr(procedure, "_adopt", lambda base, cand: (True, 1.0))
    model = procedure.fit(table, labels, labels["customer_id"].tolist(), options=frozenset({"sun"}))
    assert model["selection"]["sun_block"] == "on" and model["feature_names"][-1] == "sun_coupling"
    save_model(model, tmp_path / "model.json")
    loaded = load_model(tmp_path / "model.json")
    clim = Climatology.from_json(loaded["climatology"])
    assert clim.sun is not None and clim.sun.loo == {}
    ids = table.ids
    F, _ = compute_features(table, ids, clim)
    assert F["sun_coupling"].notna().sum() >= len(ids) // 2
    pred, _ = predict(loaded, table, ids)
    expected = classifier_proba(loaded["classifier"], loaded.get("_objects", {}), F, loaded["feature_names"])
    np.testing.assert_allclose(pred["prob"].to_numpy(float), expected, rtol=0, atol=1e-12)
