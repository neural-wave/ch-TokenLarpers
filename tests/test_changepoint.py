from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvdetect.changepoint import Commissioning, assign_regime, detect_commissioning, refine_breakpoint, shrunk_median
from pvdetect.season import zero_climatology
from pvdetect.synthetic import CustomerSpec, PVSpec
from tests.helpers import customer_daily


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
