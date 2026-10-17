from datetime import date

import pandas as pd
import pytest

from neuralwave_experiment.scoring import score_predictions
from neuralwave_experiment.features import candidate_change_point


def test_exact_conditional_score_and_missing_targets():
    labels = pd.DataFrame({"customer_id": ["a", "b"], "pv_installed": [True, False],
                           "pv_installation_date": [date(2024, 1, 1), None], "pv_capacity_kwp": [10.0, None]})
    predictions = pd.DataFrame({"customer_id": ["a", "b"], "pv_installed": [True, True],
                                "predicted_installation_date": [date(2024, 1, 16), None],
                                "predicted_capacity_kwp": [12.5, 4.0]})
    result = score_predictions(labels, predictions)
    assert result["precision"] == pytest.approx(.5)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["D"] == pytest.approx(.5)
    assert result["P"] == pytest.approx(.5)
    assert result["main_score"] == pytest.approx((2 / 3) * 85)


def test_no_true_positive_gives_zero_conditional_rewards():
    labels = pd.DataFrame({"customer_id": ["a"], "pv_installed": [True], "pv_installation_date": [date(2024, 1, 1)], "pv_capacity_kwp": [2.0]})
    predictions = pd.DataFrame({"customer_id": ["a"], "pv_installed": [False], "predicted_installation_date": [None], "predicted_capacity_kwp": [None]})
    result = score_predictions(labels, predictions)
    assert result["D"] == result["P"] == result["main_score"] == 0


def test_raw_change_point_fallback_handles_short_history():
    frame = pd.DataFrame({"customer_id": ["a"] * 4, "local_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
                          "local_slot": [44, 44, 44, 44], "energy_kwh": [1, 1, 5, 5], "value_status": ["observed"] * 4})
    assert candidate_change_point(frame, "a") in {date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)}
