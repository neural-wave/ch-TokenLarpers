from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from pvdetect.io import day_matrix, load_timeseries, split_by_customer
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


def test_split_by_customer_partitions_customers_and_rows(tmp_path):
    ids = [f"{i:05d}" for i in range(1, 9)]
    df = pd.concat(
        [generate_customer(CustomerSpec(c, "2024-03-01", "2024-03-10", missing=False), i) for i, c in enumerate(ids)],
        ignore_index=True,
    )
    src = tmp_path / "in.csv"
    df.to_csv(src, index=False, sep=";", decimal=",")
    raw = pd.read_csv(src, sep=";", dtype=str, keep_default_na=False)

    train, test = split_by_customer(src, tmp_path / "a", seed=3, chunksize=1000)
    assert (len(train), len(test)) == (6, 2)
    assert sorted(train + test) == ids
    tr = pd.read_csv(tmp_path / "a" / "training.csv", sep=";", dtype=str, keep_default_na=False)
    te = pd.read_csv(tmp_path / "a" / "test.csv", sep=";", dtype=str, keep_default_na=False)
    assert list(tr.columns) == list(raw.columns) == list(te.columns)
    assert sorted(set(tr["customer_id"])) == train and sorted(set(te["customer_id"])) == test
    key = list(raw.columns)
    both = pd.concat([tr, te]).sort_values(key, ignore_index=True)
    pdt.assert_frame_equal(both, raw.sort_values(key, ignore_index=True))

    assert split_by_customer(src, tmp_path / "b", seed=3) == (train, test)
    assert split_by_customer(src, tmp_path / "c", seed=4) != (train, test)
