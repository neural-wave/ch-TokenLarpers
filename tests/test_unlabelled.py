from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from pvdetect import procedure
from pvdetect.changepoint import detect_commissioning
from pvdetect.evaluation import outer_splits
from pvdetect.io import load_timeseries
from pvdetect.model import load_model, save_model
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.procedure import BASE_SETTING, fit
from pvdetect.season import N_WEEKS, SIGNALS, Climatology, fit_climatology
from pvdetect.synthetic import generate_dataset
from pvdetect.unlabelled import (
    MIN_PRIOR_CUSTOMERS,
    QUANTILES,
    DailyAccumulator,
    UnlabelledPrior,
    accumulate_file,
    build_unlabelled_prior,
    check_sensitivity,
)
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels

CHUNK = 7_000


def _write_csv(ts: pd.DataFrame, path: Path) -> Path:
    ts.to_csv(path, index=False, float_format="%.4f")
    return path


@pytest.fixture(scope="module")
def stream_file(tmp_path_factory) -> Path:
    """3 customers, shuffled rows, a 10-day hole, random gaps, and late duplicates with other values."""
    ts, _, _ = generate_dataset(3, seed=5, start="2023-01-01", end="2024-12-31", hard_negatives=False)
    ts["customer_id"] = "PODSYN" + ts["customer_id"].astype(str)
    rng = np.random.default_rng(5)
    first = ts["customer_id"] == ts["customer_id"].iloc[0]
    day = (ts["timestamp_epoch"] + 3600) // 86400
    hole = first & (day >= day[first].min() + 200) & (day < day[first].min() + 210)
    ts = ts[~hole & (rng.random(len(ts)) >= 0.01)]
    ts = ts.iloc[rng.permutation(len(ts))].reset_index(drop=True)
    dups = ts.iloc[:40].copy()  # first chunk; re-appear in the last chunk with different values
    dups["energy_consumption_kwh"] = dups["energy_consumption_kwh"] + 5.0
    ts = pd.concat([ts, dups], ignore_index=True)
    return _write_csv(ts, tmp_path_factory.mktemp("stream") / "unl.csv")


def test_accumulator_matches_loader(stream_file):
    loaded = load_timeseries(stream_file)
    ref = build_feature_table(loaded).dailies()
    got, rep = accumulate_file(stream_file, chunksize=CHUNK, id_prefix="PODSYN")
    assert rep["rows_duplicate"] == loaded.report["rows_duplicate"] >= 40
    assert rep["customers"] == len(ref) == 3
    assert set(got) == set(ref)
    assert any(a["total"].isna().sum() >= 10 for a in ref.values())  # the hole produced empty days
    for cid, a in ref.items():
        b = got[cid]
        assert a.index.equals(b.index)
        assert list(a.columns) == list(b.columns)
        for col in a.columns:
            np.testing.assert_allclose(b[col].to_numpy(float), a[col].to_numpy(float), rtol=1e-12, atol=1e-12)


def test_exclusion_counts(tmp_path):
    epochs = 1_700_000_000 - 1_700_000_000 % 900 + 900 * np.arange(4)
    rows = []
    for cid, n in (("", 2), ("XSYN0001", 3), ("PODSYNLAB", 4), ("PODSYN0002", 4)):
        rows += [{"customer_id": cid, "timestamp_local": "x", "timestamp_epoch": e, "energy_consumption_kwh": 0.1}
                 for e in epochs[:n]]
    p = _write_csv(pd.DataFrame(rows), tmp_path / "u.csv")
    dailies, rep = accumulate_file(p, chunksize=5, id_prefix="PODSYN", exclude_ids={"PODSYNLAB"})
    assert (rep["rows_excluded_null"], rep["rows_excluded_prefix"], rep["rows_excluded_labelled"]) == (2, 3, 4)
    assert rep["rows_total"] == 13 and rep["rows_used"] == 4
    assert set(dailies) == {"PODSYN0002"}
    acc = DailyAccumulator(id_prefix="")
    acc.add(pd.DataFrame({"customer_id": ["XSYN0001"], "epoch_s": [int(epochs[0])], "energy_kwh": [0.1]}))
    assert set(acc.finalize()) == {"XSYN0001"}


def _curves(offset: float) -> dict[float, dict[str, np.ndarray]]:
    out = {q: {k: np.zeros(N_WEEKS) for k in SIGNALS} for q in QUANTILES}
    out[0.75]["s"][20] = offset
    return out


def test_sensitivity_pass_and_fail():
    s, reason = check_sensitivity(_curves(0.04), MIN_PRIOR_CUSTOMERS)
    assert reason is None and s == pytest.approx(0.04)
    s, reason = check_sensitivity(_curves(0.06), MIN_PRIOR_CUSTOMERS)
    assert reason is not None and s == pytest.approx(0.06)
    assert check_sensitivity(_curves(0.0), MIN_PRIOR_CUSTOMERS - 1)[1] is not None


@pytest.fixture(scope="module")
def data():
    ts, labels, _ = generate_dataset(12, seed=4)
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


def _hand_prior(table) -> UnlabelledPrior:
    curves = fit_climatology(table.dailies(), list(table.dailies()), {}).curves
    spread = 0.2 + 0.1 * np.cos(2 * np.pi * np.arange(N_WEEKS) / N_WEEKS)
    return UnlabelledPrior(
        curves_q={q: curves for q in QUANTILES},
        sensitivity=0.0,
        passed=True,
        reason=None,
        spread=spread,
        n_customers=40,
        n_contrast=40,
        n_q={q: 40 for q in QUANTILES},
    )


def _stub_inner_oof(good):
    """OOF frames whose quality depends only on the option setting (perfect vs dates/capacity off)."""

    def stub(table, labels, lab, train_ids, splits, names, cap_names, setting=BASE_SETTING):
        t = lab.loc[train_ids]
        t_pv = t["pv_installed"].to_numpy(bool)
        ok = good(setting)
        shift = pd.Timedelta(days=0 if ok else 90)
        kwp = t["pv_capacity_kwp"].to_numpy(float) * (1.0 if ok else 3.0)
        dates = pd.Series(t["pv_installation_date"].to_numpy(), index=t.index) + shift
        oof = pd.DataFrame(
            {
                "p_logreg": t_pv.astype(float),
                "p_catboost": t_pv.astype(float),
                "kwp_ridge": np.where(t_pv, kwp, 5.0),
                "kwp_catboost": np.where(t_pv, kwp, 5.0),
                "regime": "event",
                "date_weak": dates,
                "date_first": dates,
            },
            index=t.index,
        )
        oof["t_pv"] = t_pv
        oof["t_date"] = t["pv_installation_date"].to_numpy()
        oof["t_kwp"] = t["pv_capacity_kwp"].to_numpy(float)
        return oof

    return stub


def test_fit_keeps_baseline_when_inner_prefers_it(data, monkeypatch):
    table, labels = data
    table = dataclasses.replace(table, unlabelled=_hand_prior(table))
    monkeypatch.setattr(procedure, "_inner_oof", _stub_inner_oof(lambda s: s == BASE_SETTING))
    model = fit(table, labels, labels["customer_id"].tolist())
    sel = model["selection"]
    assert (sel["climatology"], sel["evidence_scale"]) == BASE_SETTING
    assert set(sel["unlabelled_options"]) == {"climatology", "evidence_scale"}
    assert model["climatology"]["source"] == "labelled_nonpv"
    assert "evidence_scale" not in model["climatology"]


@pytest.mark.parametrize(
    "winner, expect",
    [
        (lambda s: s[0] == "unlabelled_0.5", ("unlabelled_0.5", "none")),
        (lambda s: s[1] == "unlabelled_spread", ("labelled_nonpv", "unlabelled_spread")),
    ],
)
def test_fit_adopts_option_that_wins_by_margin(data, monkeypatch, tmp_path, winner, expect):
    table, labels = data
    table = dataclasses.replace(table, unlabelled=_hand_prior(table))
    monkeypatch.setattr(procedure, "_inner_oof", _stub_inner_oof(winner))
    ids = labels["customer_id"].tolist()
    model = fit(table, labels, ids)
    sel = model["selection"]
    assert (sel["climatology"], sel["evidence_scale"]) == expect
    boots = [o["bootstrap_p"] for o in sel["unlabelled_options"].values() if o["bootstrap_p"] is not None]
    assert boots and max(boots) >= procedure.BOOTSTRAP_MIN
    save_model(model, tmp_path / "model.json")
    loaded = load_model(tmp_path / "model.json")
    assert loaded["climatology"]["source"] == expect[0]
    assert ("evidence_scale" in loaded["climatology"]) == (expect[1] == "unlabelled_spread")
    pred, _ = predict(loaded, table, ids)
    assert len(pred) == len(ids)


def test_climatology_json_and_unit_scale(data):
    table, labels = data
    clim = fit_climatology(table.dailies(), labels.loc[~labels["pv_installed"], "customer_id"].tolist(), {})
    old = clim.to_json()
    assert "evidence_scale" not in old
    assert Climatology.from_json(old).evidence_scale is None
    ones = dataclasses.replace(clim, evidence_scale=np.ones(N_WEEKS))
    back = Climatology.from_json(ones.to_json())
    np.testing.assert_array_equal(back.evidence_scale, np.ones(N_WEEKS))
    for cid in labels["customer_id"]:
        a = detect_commissioning(table.customers[cid].daily, clim)
        b = detect_commissioning(table.customers[cid].daily, back)
        np.testing.assert_equal(dataclasses.astuple(a), dataclasses.astuple(b))


def test_label_flip_invariant_with_prior(data):
    table, labels = data
    ts_u, _, _ = generate_dataset(8, seed=7)
    prior = build_unlabelled_prior(build_feature_table(as_loaded(ts_u)).dailies())
    prior = dataclasses.replace(prior, passed=True, reason=None)  # force every option into the inner pool
    table = dataclasses.replace(table, unlabelled=prior)
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    flipped = labels.copy()
    for x in test_ids:
        row = flipped["customer_id"] == x
        was_pv = bool(flipped.loc[row, "pv_installed"].iloc[0])
        flipped.loc[row, "pv_installed"] = not was_pv
        flipped.loc[row, "pv_installation_date"] = pd.NaT if was_pv else pd.Timestamp("2023-05-01")
        flipped.loc[row, "pv_capacity_kwp"] = np.nan if was_pv else 7.5
    a = predict(fit(table, labels, train_ids), table, test_ids)[0]
    b = predict(fit(table, flipped, train_ids), table, test_ids)[0]
    pdt.assert_frame_equal(a, b)
