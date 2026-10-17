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


def _truth8() -> pd.DataFrame:
    ids = [f"{i:03d}" for i in range(8)]
    pv = [i < 4 for i in range(8)]
    return pd.DataFrame(
        {
            "customer_id": ids,
            "pv_installed": pv,
            "pv_installation_date": pd.to_datetime(["2023-03-01" if p else None for p in pv]),
            "pv_capacity_kwp": [6.0 if p else np.nan for p in pv],
        }
    )


def _repeats(truth: pd.DataFrame) -> list[pd.DataFrame]:
    """Repeat 0 perfect; repeat 1 misses two PV customers, raises one false alarm and is 20 days late."""
    good = truth.assign(first_date=pd.Timestamp("2022-01-01"), regime="event")
    bad = good.copy()
    bad.loc[[0, 1], "pv_installed"] = False
    bad.loc[5, ["pv_installed", "pv_installation_date", "pv_capacity_kwp"]] = [True, pd.Timestamp("2023-01-01"), 5.0]
    bad.loc[[2, 3], "pv_installation_date"] += pd.Timedelta(days=20)
    return [good, bad]


def test_ci90_pools_every_repeat_on_the_same_resampled_customers():
    from types import SimpleNamespace

    from pvdetect.evaluation import bootstrap_ci
    from pvdetect.scoring import main_score

    truth = _truth8()
    res = SimpleNamespace(oof=_repeats(truth))  # NestedResult-like: the per-repeat pooled predictions
    lo, hi = bootstrap_ci(res.oof, truth, B=200, seed=3)
    rng, draws = np.random.default_rng(3), []
    for _ in range(200):
        idx = rng.integers(0, 8, 8)
        ids = [f"b{i}" for i in range(8)]
        tt = truth.iloc[idx].assign(customer_id=ids)
        draws.append(np.mean([main_score(p.iloc[idx].assign(customer_id=ids), tt)["score"] for p in res.oof]))
    assert (lo, hi) == pytest.approx((np.quantile(draws, 0.05), np.quantile(draws, 0.95)))
    assert bootstrap_ci(res.oof[:1], truth, B=200, seed=3) != pytest.approx((lo, hi))


def test_nested_cv_ci90_uses_all_repeats():
    from pvdetect.evaluation import bootstrap_ci

    truth = _truth8()
    reps = _repeats(truth)
    ids = truth["customer_id"].tolist()
    splits = [[(ids[4:], ids[:4]), (ids[:4], ids[4:])]] * 2
    fits = []

    def fit_stub(table, labels, train_ids):
        fits.append(train_ids)
        return {"selection": {}, "repeat": (len(fits) - 1) // 2}

    def predict_stub(model, table, test_ids):
        p = reps[model["repeat"]]
        return p[p["customer_id"].isin(test_ids)], pd.DataFrame()

    res = nested_cv(fit_stub, predict_stub, None, truth, splits, bootstrap=300)
    assert res.ci90 == pytest.approx(bootstrap_ci(res.oof, truth, B=300))
    assert res.ci90 != pytest.approx(bootstrap_ci(res.oof[:1], truth, B=300))
