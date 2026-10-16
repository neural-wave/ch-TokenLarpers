from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvdetect.changepoint import detect_commissioning, refine_breakpoint
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
    cp = detect_commissioning(customer_daily(spec, 11))
    assert cp is not None and cp.found
    assert abs((pd.Timestamp(cp.date) - pd.Timestamp(pv_date)).days) <= 10
