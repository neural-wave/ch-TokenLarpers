from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from pvdetect.io import day_matrix, load_timeseries
from pvdetect.synthetic import CustomerSpec, generate_customer

SPEC = CustomerSpec("000123", "2024-03-01", "2024-11-30", missing=False)


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return generate_customer(SPEC, 7)


def test_dst_days_and_utc1_matrix(tmp_path, clean):
    path = tmp_path / "e.csv"
    clean.to_csv(path, index=False)
    loaded = load_timeseries(path)
    n_days = (pd.Timestamp("2024-11-30") - pd.Timestamp("2024-03-01")).days + 1
    assert loaded.report["day_length_counts"] == {"92": 1, "96": n_days - 2, "100": 1, "other": 0}
    dates, M = day_matrix(loaded.frame)
    for d in ("2024-03-31", "2024-10-27"):
        row = M[np.flatnonzero(dates == np.datetime64(d))[0]]
        assert np.isfinite(row).sum() == 96


def test_separator_decimal_and_epoch_unit_variants_agree(tmp_path, clean):
    base = tmp_path / "comma.csv"
    clean.to_csv(base, index=False)
    semi = tmp_path / "semi.csv"
    clean.to_csv(semi, index=False, sep=";", decimal=",")
    ms = tmp_path / "ms.csv"
    clean.assign(timestamp_epoch=clean["timestamp_epoch"] * 1000).to_csv(ms, index=False)
    ref = load_timeseries(base)
    assert ref.customer_ids == ["000123"]
    for p in (semi, ms):
        other = load_timeseries(p)
        pdt.assert_frame_equal(ref.frame, other.frame)
        assert other.customer_ids == ["000123"]


def test_duplicate_row_counted_and_dropped(tmp_path):
    df = generate_customer(CustomerSpec("000123", "2024-03-01", "2024-04-30"), 7)
    path = tmp_path / "e.csv"
    df.to_csv(path, index=False)
    loaded = load_timeseries(path)
    assert loaded.report["rows_duplicate"] == 1
    assert len(loaded.frame) == len(df) - 1


def test_missing_required_column_raises(tmp_path, clean):
    path = tmp_path / "e.csv"
    clean.drop(columns=["timestamp_local"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_timeseries(path)
