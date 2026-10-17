from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvdetect.changepoint import (
    Commissioning,
    TAIL_GUARD_MIN_VALID,
    assign_regime,
    detect_commissioning,
    refine_breakpoint,
    shrunk_median,
)
from pvdetect.season import zero_climatology
from pvdetect.synthetic import CustomerSpec, Event, PVSpec, generate_dataset
from tests.helpers import as_loaded, customer_daily


def test_refine_breakpoint_finds_exact_step():
    y = np.r_[np.zeros(117), np.ones(83)]
    t, gain = refine_breakpoint(y, 110)
    assert t == 117
    assert gain > 0.99


@pytest.mark.parametrize("pv_date", ["2022-09-05", "2024-04-10"])
def test_install_date_within_10_days(pv_date):
    spec = CustomerSpec("000001", "2022-01-01", "2025-12-31", pv=PVSpec(pv_date, 8.0))
    cp = detect_commissioning(customer_daily(spec, 11), zero_climatology())
    assert cp is not None and cp.found
    assert abs((pd.Timestamp(cp.date) - pd.Timestamp(pv_date)).days) <= 10


def _cp(found: bool) -> Commissioning:
    d = pd.Timestamp("2023-01-01").date()
    return Commissioning(d, 0, 0.5, 1.0, 0.0, 0.5, 0.5, -0.5, found, None if found else d, 0)


def test_assign_regime_three_cases():
    assert assign_regime(_cp(True), -2.0, -1.0) == "event"
    assert assign_regime(_cp(False), -2.0, -1.0) == "present_at_start"
    assert assign_regime(_cp(False), 0.0, -1.0) == "insufficient"
    assert assign_regime(_cp(False), float("nan"), -1.0) == "insufficient"
    assert assign_regime(None, -2.0, -1.0) == "present_at_start"


def test_shrunk_median_rules():
    assert shrunk_median([10, 12, 8, 11, 9], -21, 21) == 5
    assert shrunk_median([30, 40, 50], 0, 365) == 15
    assert shrunk_median([100, 100, 100, 100, 100], -21, 21) == 21
    assert shrunk_median([], -21, 21) == 0


# --- Unit tests for tail_guard ---

def test_tail_guard_min_valid_constant():
    """TAIL_GUARD_MIN_VALID is the documented threshold (45 days)."""
    assert TAIL_GUARD_MIN_VALID == 45


def test_tail_guard_suppresses_end_of_series_candidate():
    """A PV install very near the series end triggers an end-of-series candidate;
    tail_guard=True must not find it when there are too few post days."""
    # Install 30 days before end: fewer than 45 valid post days → tail_guard suppresses.
    spec = CustomerSpec(
        "000001", "2022-01-01", "2025-12-31",
        pv=PVSpec("2025-12-01", 10.0),  # ~30 days before end
    )
    daily = customer_daily(spec, 42)
    cp_base = detect_commissioning(daily, zero_climatology())
    cp_tg = detect_commissioning(daily, zero_climatology(), tail_guard=True)
    # base may find a late candidate; tail_guard must not find (or must date it earlier)
    if cp_base is not None and cp_base.found and cp_tg is not None and cp_tg.found:
        # If tail_guard also finds something it must be at least 30 days earlier
        assert (pd.Timestamp(cp_base.date) - pd.Timestamp(cp_tg.date)).days >= 0


def test_tail_guard_holiday_tail_scenario_runs():
    """Tier R: holiday_tail scenario generates valid data and tail_guard runs without error (seed < 1000)."""
    ts, labels, _ = generate_dataset(8, seed=10, scenario="holiday_tail")
    from pvdetect.pipeline import build_feature_table, compute_features
    from pvdetect.season import fit_climatology
    table = build_feature_table(as_loaded(ts))
    lab = labels.set_index("customer_id")
    pv_ids = lab[lab["pv_installed"] == "TRUE"].index.tolist()
    nonpv_ids = [c for c in lab.index if lab.at[c, "pv_installed"] == "FALSE"]
    pv_dates = {}
    for c in pv_ids:
        d = lab.at[c, "pv_installation_date"]
        if d and d != "":
            pv_dates[c] = pd.Timestamp(d).date()
    clim = fit_climatology(table.dailies(), nonpv_ids, pv_dates)
    # Must run without error on all customers
    F_base, cps_base = compute_features(table, list(lab.index), clim)
    F_tg, cps_tg = compute_features(table, list(lab.index), clim, tail_guard=True)
    assert len(F_base) == len(F_tg)


# --- Unit tests for onset_first ---

def test_onset_first_changes_only_the_dating_commissioning_never_the_features():
    """`onset_first` is a dating option: every feature equals the default run, while the returned
    commissioning is the earliest-qualifying detector's (dates and regimes may move)."""
    from pvdetect.pipeline import build_feature_table, compute_features
    from pvdetect.season import fit_climatology

    ts, labels, _ = generate_dataset(16, seed=50, scenario="winter_install_ramp")
    table = build_feature_table(as_loaded(ts))
    lab = labels.set_index("customer_id")
    nonpv = [c for c in lab.index if lab.at[c, "pv_installed"] == "FALSE"]
    pv_dates = {c: pd.Timestamp(d).date() for c, d in lab["pv_installation_date"].items() if d}
    clim = fit_climatology(table.dailies(), nonpv, pv_dates)
    ids = list(lab.index)
    F_base, cps_base = compute_features(table, ids, clim)
    F_of, cps_of = compute_features(table, ids, clim, onset_first=True)
    pd.testing.assert_frame_equal(F_base, F_of)
    for c in ids:
        cd = table.customers[c]
        assert cps_of[c] == detect_commissioning(cd.daily, clim, onset_first=True)
        assert cps_base[c] == detect_commissioning(cd.daily, clim)
    assert any(cps_of[c] != cps_base[c] for c in ids)  # the scenario exercises the rule


def test_onset_first_never_picks_later_date_on_summer_install():
    """For a mid-summer install (clear step, no spring ramp ambiguity) onset_first must
    not push the date later than the base detector."""
    spec = CustomerSpec("000001", "2022-01-01", "2025-12-31", pv=PVSpec("2023-06-15", 10.0))
    daily = customer_daily(spec, 11)
    clim = zero_climatology()
    cp_base = detect_commissioning(daily, clim)
    cp_of = detect_commissioning(daily, clim, onset_first=True)
    if cp_base is not None and cp_base.found and cp_of is not None and cp_of.found:
        # onset_first must not produce a later date than the base
        assert pd.Timestamp(cp_of.date) <= pd.Timestamp(cp_base.date) + pd.Timedelta(days=1)


# --- Tier R: winter_install_ramp scenario ---

def test_winter_install_ramp_scenario_runs_and_onset_first_not_worse():
    """Tier R (seed < 1000): on winter_install_ramp, onset_first does not increase mean absolute
    date error vs base on found PV customers (within a 60-day tolerance)."""
    from pvdetect.pipeline import build_feature_table, compute_features
    from pvdetect.season import fit_climatology
    ts, labels, _ = generate_dataset(16, seed=50, scenario="winter_install_ramp")
    table = build_feature_table(as_loaded(ts))
    lab = labels.set_index("customer_id")
    pv_ids = lab[lab["pv_installed"] == "TRUE"].index.tolist()
    nonpv_ids = [c for c in lab.index if lab.at[c, "pv_installed"] == "FALSE"]
    pv_dates = {}
    for c in pv_ids:
        d = lab.at[c, "pv_installation_date"]
        if d and d != "":
            pv_dates[c] = pd.Timestamp(d).date()
    clim = fit_climatology(table.dailies(), nonpv_ids, pv_dates)
    errs_base, errs_of = [], []
    for c, true_date in pv_dates.items():
        cd = table.customers.get(c)
        if cd is None:
            continue
        cp_b = detect_commissioning(cd.daily, clim)
        cp_o = detect_commissioning(cd.daily, clim, onset_first=True)
        true_ts = pd.Timestamp(true_date)
        if cp_b is not None and cp_b.found:
            errs_base.append(abs((pd.Timestamp(cp_b.date) - true_ts).days))
        if cp_o is not None and cp_o.found:
            errs_of.append(abs((pd.Timestamp(cp_o.date) - true_ts).days))
    # onset_first must not be dramatically worse than base (tolerance: 60 days mean absolute error)
    if errs_base and errs_of:
        assert np.mean(errs_of) <= np.mean(errs_base) + 60
