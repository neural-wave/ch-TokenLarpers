import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pvdetect.changepoint import Commissioning
from pvdetect.procedure import fit_date_params

FIRST = dt.date(2021, 1, 1)
LABEL = pd.Timestamp("2022-06-15")


def _commissioning(date: dt.date) -> Commissioning:
    return Commissioning(date, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, None, None)


def _offset(errors: list[int]) -> int:
    ids = [f"PODSYN{i:03d}" for i in range(len(errors))]
    days = pd.date_range(FIRST, periods=30, freq="D")
    cd = SimpleNamespace(first_date=FIRST, n_valid=len(days), daily=pd.DataFrame({"valid": np.ones(len(days), bool)}, index=days))
    table = SimpleNamespace(customers={c: cd for c in ids})
    lab = pd.DataFrame(
        {"pv_installed": True, "pv_installation_date": LABEL}, index=pd.Index(ids, name="customer_id")
    )
    F = pd.DataFrame({"seasonal_contrast": 0.0}, index=ids)
    cps = {c: _commissioning((LABEL - pd.Timedelta(days=e)).date()) for c, e in zip(ids, errors)}
    return fit_date_params(table, lab, ids, F, cps)["offset_event"]


def test_offset_learns_months_long_label_lag():
    assert _offset([90] * 10) == 60  # round(90 * 10/15)


def test_offset_lag_is_shrunk_for_few_customers():
    assert _offset([90] * 2) == 26  # round(90 * 2/7)


def test_offset_median_ignores_minority_lag():
    assert _offset([0] * 8 + [150] * 2) == 0
