from __future__ import annotations

import pandas as pd
import pytest

from pvdetect.scoring import extension_bonus, main_score


def _frame(rows):
    return pd.DataFrame(rows, columns=["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"])


TRUTH = _frame([("a", True, "2023-01-01", 10.0), ("b", True, "2023-06-01", 4.0), ("c", False, None, None)])


def test_date_and_capacity_partial_credit():
    pred = _frame([("a", True, "2023-01-16", 12.5), ("b", True, "2023-05-17", 3.0), ("c", False, None, None)])
    s = main_score(pred, TRUTH)
    assert s["f1"] == pytest.approx(1.0)
    assert s["D"] == pytest.approx(0.5)
    assert s["P"] == pytest.approx(0.5)
    assert s["score"] == pytest.approx(85.0)


def test_false_positive_scales_score_by_f1():
    pred = _frame([("a", True, "2023-01-16", 12.5), ("b", True, "2023-05-17", 3.0), ("c", True, "2023-01-01", 5.0)])
    s = main_score(pred, TRUTH)
    assert s["f1"] == pytest.approx(0.8)
    assert s["score"] == pytest.approx(68.0)


def test_missing_date_counts_as_zero_and_stays_in_mean():
    pred = _frame([("a", True, None, 10.0), ("b", True, "2023-06-01", 4.0), ("c", False, None, None)])
    s = main_score(pred, TRUTH)
    assert s["D"] == pytest.approx(0.5)
    assert s["P"] == pytest.approx(1.0)


def test_no_true_positive_scores_zero():
    pred = _frame([("a", False, None, None), ("b", False, None, None), ("c", True, "2023-01-01", 5.0)])
    assert main_score(pred, TRUTH)["score"] == 0.0


def _ext(rows):
    return pd.DataFrame(rows, columns=["customer_id", "extension_date", "pv_capacity_after_extension_kwp"])


def test_extension_matching_is_one_to_one_within_30_days():
    truth = _ext([("a", "2024-01-01", 15.0)])
    b = extension_bonus(_ext([("a", "2024-01-11", 15.0), ("a", "2024-01-13", 15.0)]), truth)
    assert (b["tp"], b["fp"], b["fn"]) == (1, 1, 0)
    assert b["f1_ext"] == pytest.approx(2 / 3)
    assert b["D_ext"] == pytest.approx(1 - 10 / 30)
    far = extension_bonus(_ext([("a", "2024-02-01", 15.0)]), truth)
    assert far["tp"] == 0 and far["bonus"] == 0.0
