from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvdetect.features import fitted_features, label_free_features
from pvdetect.pipeline import build_feature_table
from pvdetect.season import N_WEEKS, fit_climatology
from pvdetect.synthetic import CustomerSpec, Event, PVSpec, generate_customer
from tests.helpers import as_loaded

START, END = "2022-01-01", "2025-12-31"
NONPV = [f"{100 + i:06d}" for i in range(8)]

CASES = {
    "ac_seasonal": (CustomerSpec("ac", START, END, events=(Event("ac", "2022-06-01", None, 3.0), Event("heat_pump", START))), 5),
    "pv_winter": (CustomerSpec("pvw", START, END, pv=PVSpec("2023-12-05", 8.0)), 5),
    "pv_before": (CustomerSpec("pvb", START, END, pv=PVSpec("2021-06-15", 8.0)), 5),
    "pv_summer": (CustomerSpec("pvs", START, END, pv=PVSpec("2023-06-01", 8.0)), 5),
    "ev": (CustomerSpec("ev", START, END, events=(Event("ev", "2023-06-01"),)), 5),
    "boiler_night": (CustomerSpec("bo", START, END, events=(Event("boiler_night", "2023-06-01"),)), 5),
    "heat_pump": (CustomerSpec("hp", START, END, events=(Event("heat_pump", "2023-06-01"),)), 5),
}


@pytest.fixture(scope="module")
def world():
    frames = [generate_customer(CustomerSpec(c, START, END), 100 + i) for i, c in enumerate(NONPV)]
    frames += [generate_customer(spec, seed) for spec, seed in CASES.values()]
    table = build_feature_table(as_loaded(pd.concat(frames)))
    clim = fit_climatology(table.dailies(), NONPV, {})
    out = {}
    for name, (spec, _) in CASES.items():
        cd = table.customers[spec.customer_id]
        lf = label_free_features(cd)
        f, cp = fitted_features(cd, clim, lf)
        out[name] = ({**lf, **f}, cp)
    return out


def test_climatology_recovers_known_weekly_curve():
    weeks = np.arange(1, N_WEEKS + 1)
    curve = 0.4 * np.sin(2 * np.pi * weeks / N_WEEKS)
    dates = pd.date_range("2022-01-03", periods=7 * 52 * 2, freq="D")
    woy = np.minimum(dates.isocalendar().week.to_numpy().astype(int), 52)
    dailies = {}
    for k, offset in enumerate((0.0, 1.0, -0.5, 2.0)):
        d = pd.DataFrame(
            {"s": curve[woy - 1] + offset, "mid": 1.0, "night": 1.0, "eve": 1.0, "total": 1.0, "valid": True,
             "woy": woy, "month": dates.month},
            index=dates,
        )
        dailies[f"c{k}"] = d
    clim = fit_climatology(dailies, list(dailies), {}, smooth=False)
    np.testing.assert_allclose(clim.s, curve - np.median(curve[woy - 1]), atol=1e-9)


def test_recurring_seasonal_drop_is_rejected(world):
    """The raw step at the best candidate is large (P1 would fire), but it recurs every year."""
    f, cp = world["ac_seasonal"]
    assert cp.step > 0.10
    assert not cp.found or cp.evidence < 0.10


def test_winter_install_is_dated(world):
    f, cp = world["pv_winter"]
    assert cp.found
    assert abs((pd.Timestamp(cp.date) - pd.Timestamp("2023-12-05")).days) <= 15


def test_pv_before_series_start_is_not_an_event(world):
    f, cp = world["pv_before"]
    assert not cp.found
    assert f["early_contrast"] < world["ev"][0]["early_contrast"] - 1.0


@pytest.mark.parametrize("name", ["ev", "boiler_night", "heat_pump"])
def test_night_load_confounders_are_separable_from_pv(world, name):
    f, cp = world[name]
    pv = world["pv_summer"][0]
    assert f["cp_evidence"] < 0.10 or f["d_night"] >= 0.15
    assert pv["d_night"] < 0.15
    if np.isfinite(f["cloud_var"]):
        assert f["cloud_var"] > pv["cloud_var"]
