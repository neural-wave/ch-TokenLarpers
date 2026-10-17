from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvdetect.evaluation import count_decisions, nested_cv, outer_splits
from pvdetect.io import parse_bool
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.procedure import fit
from pvdetect.synthetic import generate_dataset
from tests.helpers import as_loaded


def typed_labels(labels: pd.DataFrame) -> pd.DataFrame:
    return labels.assign(
        pv_installed=parse_bool(labels["pv_installed"]),
        pv_installation_date=pd.to_datetime(labels["pv_installation_date"], errors="coerce"),
        pv_capacity_kwp=pd.to_numeric(labels["pv_capacity_kwp"], errors="coerce"),
    )


def test_count_decisions(tmp_path):
    p = tmp_path / "DECISIONS.md"
    p.write_text("# Outer-informed decisions (Tier B)\n- 2026-01-01 | a | b\nnot a decision\n- 2026-01-02 | c | d\n")
    assert count_decisions(p) == 2


@pytest.fixture(scope="module")
def canary_data():
    ts, labels, _ = generate_dataset(24, seed=1, hard_negatives=False)
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


def test_leakage_canary(canary_data):
    """Nested CV scores high on true labels and near chance on customer-permuted labels."""
    table, labels = canary_data
    res = nested_cv(fit, predict, table, labels, outer_splits(labels, repeats=2), bootstrap=0)
    assert res.mean("score") >= 80
    perm = np.random.default_rng(0).permutation(len(labels))
    shuffled = labels.copy()
    for c in ("pv_installed", "pv_installation_date", "pv_capacity_kwp"):
        shuffled[c] = labels[c].to_numpy()[perm]
    res_p = nested_cv(fit, predict, table, shuffled, outer_splits(shuffled, repeats=2), bootstrap=0)
    assert res_p.mean("score") <= 60
