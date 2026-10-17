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


@pytest.mark.parametrize("sep", [",", ";"])
def test_rows_without_customer_id_are_dropped_and_counted(tmp_path, clean, sep):
    df = clean.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df.loc[df.index[:3], "customer_id"] = ""
    df.loc[df.index[3:5], "customer_id"] = "  "
    path = tmp_path / "e.csv"
    df.to_csv(path, index=False, sep=sep)
    loaded = load_timeseries(path)
    assert loaded.customer_ids == ["000123"]
    assert loaded.report["rows_bad_id"] == 5
    assert len(loaded.frame) == len(df) - 5 and (loaded.frame["customer_id"] == "000123").all()


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


def _labelled_series(tmp_path, n_pv: int, n_non: int, unlabelled: int = 0):
    ids = [f"{i:05d}" for i in range(1, n_pv + n_non + unlabelled + 1)]
    df = pd.concat(
        [generate_customer(CustomerSpec(c, "2024-03-01", "2024-03-03", missing=False), i) for i, c in enumerate(ids)],
        ignore_index=True,
    )
    src = tmp_path / "in.csv"
    df.to_csv(src, index=False, sep=";")
    labels = pd.DataFrame({"customer_id": ids[: n_pv + n_non], "pv_installed": [True] * n_pv + [False] * n_non})
    return src, labels, ids


@pytest.mark.parametrize(
    ("sizes", "n_test", "expected"),
    [([20, 20], 10, [5, 5]), ([7, 5], 4, [2, 2]), ([9, 2], 2, [1, 1]), ([30, 10], 10, [8, 2]), ([1, 5], 2, [0, 2])],
)
def test_stratified_counts_largest_remainder_with_one_per_side(sizes, n_test, expected):
    from pvdetect.io import _stratified_test_counts

    assert _stratified_test_counts(sizes, n_test) == expected


def test_stratified_split_class_counts_disjoint_and_deterministic(tmp_path):
    src, labels, ids = _labelled_series(tmp_path, n_pv=5, n_non=7)
    pv = set(labels.loc[labels["pv_installed"], "customer_id"])
    train, test = split_by_customer(src, tmp_path / "a", seed=2026, labels=labels, n_test=4, chunksize=500)
    assert not set(train) & set(test) and sorted(train + test) == ids
    assert (len(set(test) & pv), len(set(test) - pv)) == (2, 2)
    assert (len(set(train) & pv), len(set(train) - pv)) == (3, 5)
    te = pd.read_csv(tmp_path / "a" / "test.csv", sep=";", dtype=str, keep_default_na=False)
    assert sorted(set(te["customer_id"])) == test
    assert split_by_customer(src, tmp_path / "b", seed=2026, labels=labels, n_test=4) == (train, test)
    draws = {tuple(split_by_customer(src, tmp_path / f"s{s}", seed=s, labels=labels, n_test=4)[1]) for s in range(5)}
    assert len(draws) > 1


def test_stratified_split_ignores_labels_without_rows_and_keeps_unlabelled_in_training(tmp_path):
    src, labels, ids = _labelled_series(tmp_path, n_pv=3, n_non=6, unlabelled=2)
    ghosts = pd.DataFrame({"customer_id": ["99991", "99992", "99993"], "pv_installed": [True, True, False]})
    base = split_by_customer(src, tmp_path / "a", seed=7, labels=labels, n_test=3)
    assert split_by_customer(src, tmp_path / "b", seed=7, labels=pd.concat([labels, ghosts]), n_test=3) == base
    assert set(ids[-2:]) <= set(base[0])
    pv = set(labels.loc[labels["pv_installed"], "customer_id"])
    assert (len(set(base[1]) & pv), len(set(base[1]) - pv)) == (1, 2)
    with pytest.raises(ValueError, match="n_test"):
        split_by_customer(src, tmp_path / "c", seed=7, labels=labels, n_test=9)
