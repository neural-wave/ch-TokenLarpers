from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from pvdetect.evaluation import outer_splits
from pvdetect.model import save_model
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.procedure import DEFAULT_OPTIONS, fit
from pvdetect.synthetic import generate_dataset
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels


@pytest.fixture(scope="module")
def data():
    ts, labels, _ = generate_dataset(12, seed=4)
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


def _fold_predictions(table, labels, train_ids, test_ids, options=DEFAULT_OPTIONS) -> pd.DataFrame:
    return predict(fit(table, labels, train_ids, options=options), table, test_ids)[0]


@pytest.mark.parametrize(
    "options",
    [DEFAULT_OPTIONS, DEFAULT_OPTIONS | {"schedule"}, frozenset()],
    ids=["default", "with-schedule", "no-e1e2"],
)
def test_flipping_held_out_labels_changes_nothing_in_their_fold(data, options):
    """Any cache, climatology or selection fitted on held-out labels would change these predictions."""
    table, labels = data
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    flipped = labels.copy()
    for x in test_ids:
        row = flipped["customer_id"] == x
        was_pv = bool(flipped.loc[row, "pv_installed"].iloc[0])
        flipped.loc[row, "pv_installed"] = not was_pv
        flipped.loc[row, "pv_installation_date"] = pd.NaT if was_pv else pd.Timestamp("2023-05-01")
        flipped.loc[row, "pv_capacity_kwp"] = np.nan if was_pv else 7.5
    a = _fold_predictions(table, labels, train_ids, test_ids, options)
    b = _fold_predictions(table, flipped, train_ids, test_ids, options)
    pdt.assert_frame_equal(a, b)


def test_fit_is_deterministic(data, tmp_path):
    table, labels = data
    ids = labels["customer_id"].tolist()
    save_model(fit(table, labels, ids), tmp_path / "a" / "model.json")
    save_model(fit(table, labels, ids), tmp_path / "b" / "model.json")
    assert json.loads((tmp_path / "a" / "model.json").read_text()) == json.loads((tmp_path / "b" / "model.json").read_text())
