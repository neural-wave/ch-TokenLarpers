from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from pvdetect.synthetic import (
    EXT_CASES,
    EXT_POSITIVE_CASES,
    CustomerSpec,
    PVSpec,
    generate_customer,
    generate_extension_dataset,
)


@pytest.fixture(scope="module", params=["A", "B"])
def suite(request):
    return generate_extension_dataset(2 * len(EXT_CASES), seed=7, variant=request.param)


def test_extension_labels_follow_their_case(suite):
    ts, labels, ext, cases = suite
    cov = ts.groupby("customer_id")["timestamp_local"].agg(["min", "max"]).apply(pd.to_datetime)
    lab = labels.set_index("customer_id")
    for cid, case in cases.set_index("customer_id")["case"].items():
        ev = ext[ext["customer_id"] == cid]
        dates = pd.to_datetime(ev["extension_date"]).tolist()
        totals = ev["pv_capacity_after_extension_kwp"].astype(float).tolist()
        assert len(dates) == ({"double": 2}.get(case, 1) if case in EXT_POSITIVE_CASES else 0), (cid, case)
        assert (lab.at[cid, "pv_installed"] == "TRUE") == (case != "no_pv")
        if not dates:
            continue
        install, initial = pd.Timestamp(lab.at[cid, "pv_installation_date"]), float(lab.at[cid, "pv_capacity_kwp"])
        # totals, not added capacity: strictly above the initial kWp and increasing per event
        assert np.all(np.diff([initial, *totals]) > 0), (cid, totals)
        assert dates == sorted(dates) and install < dates[0]
        assert all(cov.at[cid, "min"] <= d <= cov.at[cid, "max"] for d in dates)
        if case == "double":
            assert (dates[1] - dates[0]).days >= 120
        if case == "early":
            assert (dates[0] - install).days <= 300
        if case == "late":
            assert (cov.at[cid, "max"] - dates[0]).days <= 121
        if case == "winter":
            assert dates[0].month in (11, 12, 1) and (dates[0].month != 11 or dates[0].day >= 15)


def _midday_net(spec: CustomerSpec, variant: Literal["A", "B"] = "A") -> pd.Series:
    """Daily mean signed withdrawal 11:00–14:00 local, indexed by local date."""
    df = generate_customer(spec, seed=3, variant=variant)
    t = pd.to_datetime(df["timestamp_local"])
    mid = df[(t.dt.hour >= 11) & (t.dt.hour < 14)]
    return mid.groupby(pd.to_datetime(mid["timestamp_local"]).dt.normalize())["energy_consumption_kwh"].mean()


def _spec(**changes) -> CustomerSpec:
    pv = replace(PVSpec("2022-01-15", 6.0, export=True), **changes)
    return CustomerSpec("000001", "2023-04-01", "2023-08-31", pv=pv, missing=False, weather_seed=4)


def _window(s: pd.Series, a: str, b: str) -> float:
    return float(s[(s.index >= a) & (s.index < b)].mean())


@pytest.mark.parametrize("variant", ["A", "B"])
def test_extension_raises_and_outage_removes_midday_production(variant):
    plain = _midday_net(_spec(), variant)
    extended = _midday_net(_spec(extensions=(("2023-06-01", 12.0),)), variant)
    outage = _midday_net(_spec(outages=(("2023-06-01", "2023-07-01"),)), variant)
    no_pv = _midday_net(CustomerSpec("000001", "2023-04-01", "2023-08-31", missing=False, weather_seed=4), variant)
    # same load and weather draws: before the event the series agree
    assert _window(extended, "2023-04-01", "2023-06-01") == pytest.approx(_window(plain, "2023-04-01", "2023-06-01"))
    # doubling kWp doubles the midday export, which dominates the load
    gain = _window(plain, "2023-06-01", "2023-09-01") - _window(extended, "2023-06-01", "2023-09-01")
    assert gain == pytest.approx(_window(no_pv, "2023-06-01", "2023-09-01") - _window(plain, "2023-06-01", "2023-09-01"), rel=0.05)
    # an outage returns the midday withdrawal to the no-PV load, then production comes back
    assert _window(outage, "2023-06-01", "2023-07-01") == pytest.approx(_window(no_pv, "2023-06-01", "2023-07-01"))
    assert _window(outage, "2023-07-01", "2023-09-01") == pytest.approx(_window(plain, "2023-07-01", "2023-09-01"))


def test_east_west_extension_broadens_the_daily_shape():
    def morning_share(spec: CustomerSpec) -> float:
        no_pv = generate_customer(CustomerSpec("000001", "2023-06-01", "2023-08-31", missing=False, weather_seed=4), 3)
        df = generate_customer(spec, 3)
        prod = no_pv["energy_consumption_kwh"].to_numpy() - df["energy_consumption_kwh"].to_numpy()
        h = pd.to_datetime(df["timestamp_local"]).dt.hour.to_numpy()
        return prod[(h >= 7) & (h < 10)].sum() / prod[(h >= 11) & (h < 14)].sum()

    def spec(orientation):
        pv = PVSpec("2022-01-15", 6.0, export=True, extensions=(("2023-01-01", 12.0),), extension_orientation=orientation)
        return CustomerSpec("000001", "2023-06-01", "2023-08-31", pv=pv, missing=False, weather_seed=4)

    assert morning_share(spec("east_west")) > 1.2 * morning_share(spec(None))


def test_regional_weather_is_shared_within_a_region_and_correlated_across():
    from pvdetect.synthetic import _grid, _shared_sun

    g = _grid("2022-01-01", "2023-12-31")
    day = g.std_day
    first = np.r_[True, np.diff(day) != 0]
    nat, r1, r2 = (_shared_sun(9, g, k)[first] for k in (0, 1, 2))
    np.testing.assert_array_equal(_shared_sun(9, g, 1)[first], r1)
    c12 = np.corrcoef(r1, r2)[0, 1]
    assert 0.4 < c12 < 0.85
    assert np.corrcoef(nat, r1)[0, 1] > c12
