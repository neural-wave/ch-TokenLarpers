"""Hand-built checks of the inner-CV scoring helpers that every automatic choice relies on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvdetect.scoring import main_score
from pvdetect.selection import (
    THRESHOLDS,
    bootstrap_prob_better,
    pick_threshold,
    score_arrays,
    threshold_scores,
)

NAN = np.nan


def _official(t_pv, p_pv, err, ape) -> float:
    """Same case through the official scorer: truth date 2024-01-01 and 10 kWp for every PV customer."""
    ids = [f"{i:03d}" for i in range(len(t_pv))]
    base = pd.Timestamp("2024-01-01")
    truth = pd.DataFrame(
        {
            "customer_id": ids,
            "pv_installed": t_pv,
            "pv_installation_date": [base if t else pd.NaT for t in t_pv],
            "pv_capacity_kwp": [10.0 if t else NAN for t in t_pv],
        }
    )
    pred = pd.DataFrame(
        {
            "customer_id": ids,
            "pv_installed": p_pv,
            "pv_installation_date": [
                base + pd.Timedelta(days=e) if p and np.isfinite(e) else pd.NaT for p, e in zip(p_pv, err)
            ],
            "pv_capacity_kwp": [10.0 * (1 + a) if p and np.isfinite(a) else NAN for p, a in zip(p_pv, ape)],
        }
    )
    return main_score(pred, truth)["score"]


def test_all_four_outcomes_and_partial_credit_match_the_official_score():
    t_pv = np.array([True, True, True, False, False])
    p_pv = np.array([True, True, False, True, False])  # 2 TP, 1 FN, 1 FP, 1 TN
    err = np.array([0.0, 15.0, NAN, 3.0, NAN])  # D: 1.0 and 0.5
    ape = np.array([0.0, 0.25, NAN, 0.1, NAN])  # P: 1.0 and 0.5
    f1 = 2 * 2 / (2 * 2 + 1 + 1)
    expected = f1 * (70 + 15 * 0.75 + 15 * 0.75)
    assert score_arrays(t_pv, p_pv, err, ape) == pytest.approx(expected)
    assert score_arrays(t_pv, p_pv, err, ape) == pytest.approx(_official(t_pv, p_pv, err, ape))


def test_missing_date_or_capacity_on_a_true_positive_scores_zero_for_that_part():
    t_pv = np.array([True, True])
    p_pv = np.array([True, True])
    err = np.array([0.0, NAN])
    ape = np.array([NAN, 0.0])
    assert score_arrays(t_pv, p_pv, err, ape) == pytest.approx(1.0 * (70 + 15 * 0.5 + 15 * 0.5))


def test_no_true_positive_scores_zero():
    t_pv = np.array([True, False])
    assert score_arrays(t_pv, np.array([False, True]), np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0
    assert score_arrays(np.array([False, False]), np.array([False, False]), np.zeros(2), np.zeros(2)) == 0.0


def test_errors_beyond_the_scoring_windows_floor_at_zero():
    one = np.array([True])
    assert score_arrays(one, one, np.array([-45.0]), np.array([0.9])) == pytest.approx(70.0)


def test_threshold_scores_follow_the_probability_cut():
    prob = np.array([0.9, 0.6, 0.3, 0.1])
    t_pv = np.array([True, True, False, False])
    err, ape = np.zeros(4), np.zeros(4)
    scores = threshold_scores(prob, t_pv, err, ape)
    assert scores.shape == THRESHOLDS.shape
    for t, s in zip(THRESHOLDS, scores):
        assert s == pytest.approx(score_arrays(t_pv, prob >= t, err, ape))
    assert scores[THRESHOLDS.tolist().index(0.5)] == pytest.approx(100.0)


def test_pick_threshold_breaks_ties_toward_half_then_higher():
    scores = np.zeros(THRESHOLDS.size)
    scores[:] = 50.0
    assert pick_threshold(scores) == (0.5, 50.0)
    scores = np.zeros(THRESHOLDS.size)
    lo, hi = THRESHOLDS.tolist().index(0.45), THRESHOLDS.tolist().index(0.55)
    scores[[lo, hi]] = 80.0
    assert pick_threshold(scores) == (0.55, 80.0)
    scores = np.zeros(THRESHOLDS.size)
    scores[0] = 90.0
    assert pick_threshold(scores) == (float(THRESHOLDS[0]), 90.0)


def test_bootstrap_prefers_the_clearly_better_classifier_and_is_reproducible():
    rng = np.random.default_rng(3)
    t_pv = rng.random(40) < 0.5
    good = np.where(t_pv, 0.9, 0.1)
    bad = np.where(rng.random(40) < 0.5, 0.9, 0.1)
    err, ape = np.zeros(40), np.zeros(40)
    p = bootstrap_prob_better(bad, 0.5, good, 0.5, t_pv, err, ape, B=200, seed=0)
    assert p > 0.95
    assert bootstrap_prob_better(good, 0.5, bad, 0.5, t_pv, err, ape, B=200, seed=0) < 0.05
    assert p == bootstrap_prob_better(bad, 0.5, good, 0.5, t_pv, err, ape, B=200, seed=0)
    assert bootstrap_prob_better(good, 0.5, good, 0.5, t_pv, err, ape, B=200, seed=0) == 0.0
