from datetime import date, datetime, timezone

import numpy as np

from neuralwave_experiment.temporal import summarize_trace, temporal_diagnostics
from neuralwave_experiment.windows import WindowBatch


CFG = {
    "persistence_days": 14,
    "persistence_min_coverage": 0.9,
    "persistence_max_gap_days": 2,
}


def _batch(centres, observed=None, length=1):
    centres = np.asarray(centres, dtype=np.int64)
    n = len(centres)
    if observed is None:
        observed = np.ones(n, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    values = np.zeros((n, length, 1), dtype=np.float32)
    mask = np.repeat(observed[:, None, None], length, axis=1)
    starts = centres - (length // 2) * 900
    return WindowBatch(values, mask, np.zeros((n, length, 4), np.float32),
                       np.zeros((n, 4), np.float32), ["c"] * n,
                       [datetime.fromtimestamp(int(x), timezone.utc).date() for x in centres], starts)


def _days(start, count):
    return [start + i * 86_400 for i in range(count)]


def test_elapsed_day_persistence_and_dst_epoch_date():
    # The Zurich DST transition must not turn fourteen UTC days into thirteen.
    centres = _days(int(datetime(2024, 3, 24, tzinfo=timezone.utc).timestamp()), 14)
    result = summarize_trace(_batch(centres), np.full(14, 0.8), 0.7, CFG)
    assert result["score"] == 0.8
    assert result["predicted_installation_date"] is not None
    assert result["confirmation_date"] is not None


def test_sparse_four_day_samples_do_not_bridge_persistence():
    base = 1_700_000_000
    # Four observations are separated by four days, not one-day support.
    sparse = [base + i * 4 * 86_400 for i in range(4)]
    result = summarize_trace(_batch(sparse), np.full(4, 0.9), 0.5, CFG)
    assert result["predicted_installation_date"] is None
    assert not result["coverage_eligible"]

def test_clustered_midnight_centers_do_not_manufacture_two_days():
    config = {
        "persistence_days": 2,
        "persistence_min_coverage": 0.9,
        "persistence_max_gap_days": 2,
    }
    first = int(datetime(2024, 1, 1, 23, 59, tzinfo=timezone.utc).timestamp())
    result = summarize_trace(_batch([first, first + 120]), np.full(2, 0.95), 0.5, config)
    assert result["predicted_installation_date"] is None
    assert not result["coverage_eligible"]


def test_short_record_has_no_max_fallback():
    result = summarize_trace(_batch(_days(1_700_000_000, 10)), np.full(10, 1.0), 0.5, CFG)
    assert result == {
        "score": 0.0,
        "predicted_installation_date": None,
        "coverage_eligible": False,
        "confirmation_date": None,
    }


def test_median_threshold_crossing_and_missing_support():
    centres = _days(1_700_000_000, 14)
    probabilities = np.array([0.9] * 7 + [0.1] * 7)
    result = summarize_trace(_batch(centres), probabilities, 0.5, CFG)
    assert result["score"] == 0.5
    assert result["predicted_installation_date"] is not None

    missing = [1.0] * 12 + [0.0, 0.0]
    result = summarize_trace(_batch(centres, missing), np.full(14, 0.9), 0.5, CFG)
    assert result["predicted_installation_date"] is None
    assert not result["coverage_eligible"]


def test_diagnostics_separate_preinstallation_and_onset():
    install = date(2024, 2, 1)
    start = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    centres = _days(start, 14)
    batch = _batch(centres)
    result = temporal_diagnostics(batch, np.full(14, 0.9),
                                 {"pv_installed": True, "pv_installation_date": install.isoformat()},
                                 0.5, CFG)
    assert result["pre_installation_false_alarm"] is True
    assert result["date_reward"] == 0.0
    assert result["onset_error_days"] < 0
    assert result["absolute_onset_error_days"] == abs(result["onset_error_days"])


def test_diagnostics_non_pv_and_late_installation():
    start = int(datetime(2024, 5, 1, tzinfo=timezone.utc).timestamp())
    batch = _batch(_days(start, 14))
    non_pv = temporal_diagnostics(batch, np.full(14, 0.9),
                                  {"pv_installed": False, "pv_installation_date": None},
                                  0.5, CFG)
    assert non_pv["non_pv_false_alarm"] is True
    assert non_pv["date_reward"] is None

    late = temporal_diagnostics(batch, np.full(14, 0.9),
                                {"pv_installed": True, "pv_installation_date": "2024-05-10"},
                                0.5, CFG)
    assert late["late_installation"] is True

def test_pv_missing_date_is_censored_not_invalid():
    batch = _batch(_days(1_700_000_000, 14))
    result = temporal_diagnostics(batch, np.full(14, 0.9),
                                  {"pv_installed": True, "pv_installation_date": np.nan},
                                  0.5, CFG)
    assert result["date_reward"] is None
    assert result["observable_onset"] is None
