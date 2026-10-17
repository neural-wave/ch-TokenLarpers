import pandas as pd
import pytest

from neuralwave_experiment.evaluation import select_threshold, _split


def test_threshold_uses_customer_predictions_not_window_count():
    labels = pd.DataFrame({"customer_id": ["positive", "negative"],
                           "pv_installed": [True, False]})
    threshold, grid = select_threshold({"positive": .6, "negative": .4}, labels,
                                      [.3, .5, .7])
    assert threshold == .5
    assert [row["f1"] for row in grid] == pytest.approx([2 / 3, 1., 0.])


def test_calibration_rejects_missing_or_unexpected_customer():
    labels = pd.DataFrame({"customer_id": ["p", "n"],
                           "pv_installed": [True, False]})
    with pytest.raises(ValueError, match="exactly one OOF"):
        select_threshold({"p": .6}, labels, [.5])
    with pytest.raises(ValueError, match="exactly one OOF"):
        select_threshold({"p": .6, "n": .4, "heldout": .9}, labels, [.5])


def test_customer_splits_change_and_remain_disjoint_with_full_coverage():
    labels = pd.DataFrame({"customer_id": [f"c{i:02}" for i in range(20)],
                           "pv_installed": [True] * 10 + [False] * 10})
    first, second = _split(labels, 5, 17), _split(labels, 5, 42)
    assert {frozenset(v) for _, v in first} != {frozenset(v) for _, v in second}
    for splits in (first, second):
        seen = []
        for train, validation in splits:
            assert set(train).isdisjoint(validation)
            assert set(train) | set(validation) == set(labels.customer_id)
            seen.extend(validation)
        assert sorted(seen) == sorted(labels.customer_id)


def test_nested_split_fails_instead_of_silently_lowering_fold_count():
    labels = pd.DataFrame({"customer_id": ["p", "n", "n2"],
                           "pv_installed": [True, False, False]})
    with pytest.raises(ValueError, match="each class"):
        _split(labels, 2, 17)
