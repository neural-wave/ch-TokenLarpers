from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvdetect.features import (
    TOBIT_EPS_KW,
    TOBIT_GRID,
    tobit_bell,
    tobit_estimate,
    tobit_features,
    tobit_loglik,
)

DATES = pd.date_range("2022-01-01", "2024-12-31", freq="D")
ONSET = pd.Timestamp("2023-03-01")
FLAT_YIELD = np.ones(52)


def _matrix(C: float, seed: int = 0, load_kw: float = 20.0, sd: float = 0.3) -> np.ndarray:
    """Hand-built UTC+1 day matrix (kWh per slot): lognormal load, PV C · bell after ONSET, no export."""
    rng = np.random.default_rng(seed)
    load = np.exp(np.log(load_kw) + sd * rng.standard_normal((len(DATES), 96))) / 4
    hours = (np.arange(96) + 0.5) / 4
    pv = np.where(np.asarray(DATES >= ONSET)[:, None], C * tobit_bell(hours[None, :], DATES.month.to_numpy()[:, None]) / 4, 0)
    return np.maximum(0.0, load - pv)


def test_bell_is_timestamp_only_with_fixed_month_lengths():
    assert tobit_bell(np.array([12.5]), np.array([6]))[0] == 1.0
    # zero at 12.5 ± L/2: L = 14 h in June, 9 h in December, 11.5 h in March and September
    np.testing.assert_allclose(tobit_bell(np.array([5.5, 19.5]), np.array([6, 6])), 0.0, atol=1e-12)
    np.testing.assert_allclose(tobit_bell(np.array([8.0, 17.0]), np.array([12, 12])), 0.0, atol=1e-12)
    np.testing.assert_allclose(tobit_bell(np.array([12.5 + 11.5 / 4]), np.array([3])), np.cos(np.pi / 4))
    np.testing.assert_allclose(tobit_bell(np.array([10.0]), np.array([3])), tobit_bell(np.array([10.0]), np.array([9])))


def test_loglik_terms_exact():
    # one uncensored reading: lognormal log-pdf of r + C f; one censored: log Φ((log C f − μ)/σ)
    from scipy.stats import lognorm, norm

    r, f = np.array([2.0, 0.0]), np.array([0.5, 0.25])
    mu, sd = np.array([np.log(3.0), np.log(3.0)]), np.array([0.4, 0.4])
    C = 4.0
    want = lognorm.logpdf(2.0 + C * 0.5, 0.4, scale=3.0) + norm.logcdf((np.log(C * 0.25) - np.log(3.0)) / 0.4)
    np.testing.assert_allclose(tobit_loglik(np.array([np.log(C)]), r, f, mu, sd), [want])


def test_estimate_recovers_censored_capacity():
    M = _matrix(40.0)
    out = tobit_features(DATES.values, M, np.ones(len(DATES), bool), ONSET, FLAT_YIELD)
    assert abs(np.exp(out["log_kwp_tobit"]) / 40.0 - 1) < 0.10
    assert (4 * M[np.asarray(DATES >= ONSET)][:, 40:60] <= TOBIT_EPS_KW).mean() > 0.5  # mostly censored at midday


@pytest.mark.parametrize("C", [150.0, 600.0])
def test_estimate_recovers_capacity_far_above_the_load(C):
    """The midday deficit saturates at the load; the censored likelihood still sees the size."""
    out = tobit_features(DATES.values, _matrix(C, seed=1), np.ones(len(DATES), bool), ONSET, FLAT_YIELD)
    assert abs(np.exp(out["log_kwp_tobit"]) / C - 1) < 0.10


def test_estimate_stays_on_the_grid_range():
    r, f = np.zeros(50), np.full(50, 0.5)  # all censored: the likelihood rises with C
    mu, sd = np.full(50, np.log(10.0)), np.full(50, 0.3)
    assert tobit_estimate(r, f, mu, sd) == pytest.approx(TOBIT_GRID[-1])


def test_nan_without_enough_pre_onset_days():
    valid = np.ones(len(DATES), bool)
    early = pd.Timestamp("2022-02-15")  # < 60 valid days before the onset
    assert np.isnan(tobit_features(DATES.values, _matrix(40.0), valid, early, FLAT_YIELD)["log_kwp_tobit"])
    valid[(DATES < ONSET) & (DATES >= pd.Timestamp("2022-01-15"))] = False  # 14 valid pre days
    assert np.isnan(tobit_features(DATES.values, _matrix(40.0), valid, ONSET, FLAT_YIELD)["log_kwp_tobit"])


def test_nan_without_post_onset_summer_days():
    late = pd.Timestamp("2024-09-01")
    assert np.isnan(tobit_features(DATES.values, _matrix(40.0), np.ones(len(DATES), bool), late, FLAT_YIELD)["log_kwp_tobit"])


def test_nan_without_any_pre_onset_reading():
    M = _matrix(40.0)
    M[np.asarray(DATES < ONSET)] = np.nan
    assert np.isnan(tobit_features(DATES.values, M, np.ones(len(DATES), bool), ONSET, FLAT_YIELD)["log_kwp_tobit"])


def _inner_scores(scenario: str, seed: int) -> tuple[float, float]:
    """Full inner score without / with `log_kwp_tobit` in the capacity names, same inner splits."""
    from pvdetect import procedure as P
    from pvdetect.evaluation import inner_splits
    from pvdetect.features import capacity_feature_names, feature_names
    from pvdetect.pipeline import build_feature_table
    from pvdetect.synthetic import generate_dataset
    from tests.helpers import as_loaded
    from tests.test_evaluation import typed_labels

    ts, labels, _ = generate_dataset(32, seed=seed, scenario=scenario)
    table, lab = build_feature_table(as_loaded(ts)), typed_labels(labels)
    ids = list(lab["customer_id"])
    splits = inner_splits(lab, ids)
    names, cap = feature_names(False), capacity_feature_names(False)
    scores = []
    for cap_names in (cap, cap + ["log_kwp_tobit"]):
        oof = P._inner_oof(table, lab, lab.set_index("customer_id"), ids, splits, names, cap_names)
        scores.append(P._full_inner_score(oof, P._select(oof, False)))
    return scores[0], scores[1]


def test_tier_r_censored_large_gains_and_base_is_not_hurt():
    """Tier R (dev seeds, variant A): with large non-exporting commercial PV the midday deficit saturates at the
    load; `log_kwp_tobit` raises the inner score there and leaves `base` within a small tolerance."""
    off, on = _inner_scores("censored_large", 11)
    assert on - off >= 1.0
    off, on = _inner_scores("base", 13)
    assert on - off >= -0.2
