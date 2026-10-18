from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import train
from pvdetect import procedure, pseudolabel
from pvdetect.evaluation import outer_splits
from pvdetect.io import load_timeseries
from pvdetect.model import load_model, save_model
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.procedure import BASE_SETTING, fit
from pvdetect.synthetic import generate_dataset
from pvdetect.unlabelled import POOL_MIN_VALID_DAYS, accumulate_file, build_pool, select_pool
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels

CHUNK = 7_000


@pytest.fixture(scope="module")
def unl_csv(tmp_path_factory) -> Path:
    """4 PODSYN customers over 2 years, one cut to 300 days; shuffled rows, gaps, late duplicates."""
    ts, _, _ = generate_dataset(4, seed=5, start="2023-01-01", end="2024-12-31", hard_negatives=False)
    ts["customer_id"] = "PODSYN" + ts["customer_id"].astype(str)
    rng = np.random.default_rng(5)
    short = ts["customer_id"] == ts["customer_id"].iloc[-1]
    day = (ts["timestamp_epoch"] + 3600) // 86400
    ts = ts[~(short & (day >= day[short].min() + 300)) & (rng.random(len(ts)) >= 0.01)]
    ts = ts.iloc[rng.permutation(len(ts))].reset_index(drop=True)
    dups = ts.iloc[:40].copy()  # first chunk; re-appear in the last chunk with different values
    dups["energy_consumption_kwh"] = dups["energy_consumption_kwh"] - 5.0
    ts = pd.concat([ts, dups], ignore_index=True)
    path = tmp_path_factory.mktemp("pool") / "unl.csv"
    ts.to_csv(path, index=False, float_format="%.4f")
    return path


def test_pool_matches_feature_table(unl_csv):
    ref = build_feature_table(load_timeseries(unl_csv)).customers
    dailies, _ = accumulate_file(unl_csv, chunksize=CHUNK, id_prefix="PODSYN")
    pool = build_pool(unl_csv, dailies, 10, chunksize=CHUNK)
    eligible = {c for c, cd in ref.items() if cd.n_valid >= POOL_MIN_VALID_DAYS}
    assert set(pool) == eligible and len(eligible) == 3
    for cid, cd in pool.items():
        r = ref[cid]
        np.testing.assert_array_equal(cd.dates, r.dates)
        assert cd.M.dtype == np.float32
        np.testing.assert_allclose(cd.M, r.M.astype(np.float32), rtol=0, atol=0, equal_nan=True)
        assert list(cd.daily.columns) == list(r.daily.columns) and cd.daily.index.equals(r.daily.index)
        for col in r.daily.columns:
            np.testing.assert_allclose(cd.daily[col].to_numpy(float), r.daily[col].to_numpy(float), rtol=1e-12)
        assert (cd.neg_frac, cd.q001, cd.first_date, cd.n_valid) == (r.neg_frac, r.q001, r.first_date, r.n_valid)


def test_pool_selection_is_deterministic_and_order_free(unl_csv):
    n_valid = {f"PODSYN{i:04d}": 365 + i for i in range(40)} | {"PODSYNSHORT": POOL_MIN_VALID_DAYS - 1}
    a = select_pool(n_valid, 7)
    b = select_pool(dict(reversed(list(n_valid.items()))), 7)
    assert a == b == sorted(a) and len(a) == 7 and "PODSYNSHORT" not in a
    assert set(a) < set(select_pool(n_valid, 20)) and len(select_pool(n_valid, 100)) == 40
    assert select_pool(n_valid, 0) == []
    # the second pass sees each row; any row order gives the same pool and the same data
    shuffled = unl_csv.with_name("unl_rev.csv")
    raw = pd.read_csv(unl_csv, dtype={"customer_id": str})
    raw = raw.drop_duplicates(["customer_id", "timestamp_epoch"])  # without duplicates order cannot matter
    raw.to_csv(unl_csv.with_name("unl_dedup.csv"), index=False)
    raw.iloc[::-1].to_csv(shuffled, index=False)
    pools = []
    for p in (unl_csv.with_name("unl_dedup.csv"), shuffled):
        dailies, _ = accumulate_file(p, chunksize=CHUNK, id_prefix="PODSYN")
        pools.append(build_pool(p, dailies, 2, chunksize=CHUNK))
    assert list(pools[0]) == list(pools[1]) and len(pools[0]) == 2
    for cid in pools[0]:
        np.testing.assert_array_equal(pools[0][cid].M, pools[1][cid].M)


def test_train_respects_pseudo_pool(unl_csv, tmp_path, monkeypatch):
    table = build_feature_table(as_loaded(generate_dataset(2, seed=4)[0]))
    args = SimpleNamespace(
        unlabelled=str(unl_csv),
        unlabelled_chunksize=CHUNK,
        unlabelled_id_prefix="PODSYN",
        out=str(tmp_path / "model.json"),
        pseudo_pool=2,
    )
    t2 = train._attach_unlabelled(table, table.ids, args)
    assert len(t2.unlabelled_pool) == 2 and t2.unlabelled is not None
    monkeypatch.setattr(train, "build_pool", lambda *a, **k: pytest.fail("no second pass with --pseudo-pool 0"))
    t0 = train._attach_unlabelled(table, table.ids, SimpleNamespace(**{**vars(args), "pseudo_pool": 0}))
    assert t0.unlabelled_pool is None and t0.unlabelled is not None


def test_pick_pseudo_thresholds_and_caps():
    prob = np.array([0.99, 0.95, 0.94, 0.5, 0.05, 0.051, 0.0, 0.97, 0.01])
    pos, neg = pseudolabel.pick_pseudo(prob, 2, 2)
    assert pos.tolist() == [0, 7] and neg.tolist() == [6, 8]
    pos, neg = pseudolabel.pick_pseudo(prob, 10, 10)
    assert pos.tolist() == [0, 1, 7] and neg.tolist() == [4, 6, 8]
    pos, neg = pseudolabel.pick_pseudo(np.full(4, 0.96), 2, 0)
    assert pos.tolist() == [0, 1] and neg.size == 0


def test_with_pseudo_builds_capped_weighted_set(monkeypatch):
    ids = ["L1", "L2", "L3", "L4", "L5"]
    lab = pd.DataFrame({"pv_installed": [True, True, False, False, False]}, index=ids)
    names = ["a", "b"]
    F = pd.DataFrame({"a": np.arange(5.0), "b": -np.arange(5.0)}, index=ids)
    # 8 confident positives (cap 3×2=6), 10 confident negatives (cap 3×3=9), 2 unsure
    prob = np.r_[np.linspace(0.95, 0.99, 8), np.linspace(0.0, 0.05, 10), [0.5, 0.94]]
    pool_ids = [f"PODSYN{i:02d}" for i in range(prob.size)]
    Fp = pd.DataFrame({"a": 100.0 + np.arange(prob.size), "b": 0.0}, index=pool_ids)
    seen = {}
    monkeypatch.setattr(pseudolabel, "pool_features", lambda table, clim, recur_strict=False: Fp)

    def proba(clf, objects, F_, names_, kind, weekly=None):
        seen.update(kind=kind, weekly=weekly)
        return prob

    def fake_fit(X, y, kinds, weight, C):
        seen.update(X=X, y=y, kinds=kinds, w=weight, C=C)
        return {"refit": True}, {"catboost_clf": "refit"}

    monkeypatch.setattr(pseudolabel, "classifier_proba", proba)
    monkeypatch.setattr(procedure, "_fit_classifiers", fake_fit)
    comp = procedure.Components(
        None, F, {}, {}, {"refit": False}, {"r": 1}, None, 5.0, {"catboost_cap": "cap"}, logreg_C=2.0
    )
    out, counts = pseudolabel.with_pseudo(None, lab, ids, names, comp, "catboost")
    assert seen["kind"] == "logreg"  # no CatBoost in comp -> logreg scores the pool, as in `_inner_oof`
    assert seen["weekly"] is None and seen["C"] == 2.0  # refit at the fit's logreg C
    assert counts == (6, 9) and seen["kinds"] == ("logreg", "catboost")
    pos_rows = list(range(2, 8))  # the 6 highest of the 8 increasing positives
    neg_rows = list(range(8, 17))  # the 9 lowest of the 10 increasing negatives
    np.testing.assert_array_equal(seen["X"][:5], F.to_numpy())
    np.testing.assert_array_equal(seen["X"][5:, 0], 100.0 + np.array(pos_rows + neg_rows))
    assert seen["y"].tolist() == [True, True, False, False, False] + [True] * 6 + [False] * 9
    assert seen["w"].tolist() == [1.0] * 5 + [pseudolabel.WEIGHT] * 15
    assert out.logreg == {"refit": True}
    assert out.objects == {"catboost_cap": "cap", "catboost_clf": "refit"}
    assert (out.ridge, out.fallback, out.F) == (comp.ridge, comp.fallback, comp.F)
    # nothing confident -> unchanged
    monkeypatch.setattr(pseudolabel, "classifier_proba", lambda *a, **k: np.full(prob.size, 0.5))
    assert pseudolabel.with_pseudo(None, lab, ids, names, comp, "logreg") == (comp, (0, 0))


# ------------------------------------------------------------------------------------- fit level


@pytest.fixture(scope="module")
def data():
    ts, labels, _ = generate_dataset(12, seed=4)
    table = build_feature_table(as_loaded(ts))
    ts_u, _, _ = generate_dataset(10, seed=7)
    ts_u["customer_id"] = "PODSYN" + ts_u["customer_id"].astype(str)
    pool = {
        c: dataclasses.replace(cd, M=cd.M.astype(np.float32))
        for c, cd in build_feature_table(as_loaded(ts_u)).customers.items()
    }
    return table, dataclasses.replace(table, unlabelled_pool=pool), typed_labels(labels)


# the stubbed inner OOF scores no weekly head / C / date options
STUB_OPTIONS = procedure.DEFAULT_OPTIONS - {"weekly", "ctune", "offset_tri", "recur_strict"}


def _stub_inner_oof(good):
    """OOF frames whose quality depends only on whether pseudo-labelling refits the classifier."""

    def stub(table, labels, lab, train_ids, splits, names, cap_names, setting=BASE_SETTING, refit=None, **_):
        t = lab.loc[train_ids]
        t_pv = t["pv_installed"].to_numpy(bool)
        ok = good(refit is not None)
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


def _json(model: dict, path: Path) -> dict:
    save_model(model, path)
    return json.loads(path.read_text())


def test_fit_keeps_off_when_inner_prefers_it_and_model_is_unchanged(data, monkeypatch, tmp_path):
    plain, pooled, labels = data
    ids = labels["customer_id"].tolist()
    monkeypatch.setattr(procedure, "_inner_oof", _stub_inner_oof(lambda pseudo: not pseudo))
    with_pool = _json(fit(pooled, labels, ids, options=STUB_OPTIONS), tmp_path / "a" / "model.json")
    without = _json(fit(plain, labels, ids, options=STUB_OPTIONS), tmp_path / "b" / "model.json")
    sel = with_pool["selection"]
    assert sel["pseudo"] == "off" and "pseudo_labeller" not in sel
    assert sel["pseudo_options"]["on"] < sel["pseudo_options"]["off"]
    assert not any(k.startswith("pseudo") for k in without["selection"])
    for k in [k for k in sel if k.startswith("pseudo")]:
        del sel[k]
    assert with_pool == without


def test_fit_adopts_pseudo_that_wins_by_margin(data, monkeypatch, tmp_path):
    plain, pooled, labels = data
    ids = labels["customer_id"].tolist()
    monkeypatch.setattr(procedure, "_inner_oof", _stub_inner_oof(lambda pseudo: pseudo))
    model = fit(pooled, labels, ids, options=STUB_OPTIONS)
    sel = model["selection"]
    assert sel["pseudo"] == "on" and sel["pseudo_labeller"] == "logreg"
    assert sel["pseudo_options"]["on"] - sel["pseudo_options"]["off"] >= procedure.CLASSIFIER_MARGIN
    assert sel["pseudo_options"]["bootstrap_p"] >= procedure.BOOTSTRAP_MIN
    counts = sel["pseudo_counts"]
    assert counts["pool"] == len(pooled.unlabelled_pool)
    assert counts["final_pos"] + counts["final_neg"] > 0
    assert counts["final_pos"] <= 3 * labels["pv_installed"].sum()
    assert counts["final_neg"] <= 3 * (~labels["pv_installed"]).sum()
    save_model(model, tmp_path / "model.json")
    loaded = load_model(tmp_path / "model.json")
    assert loaded["selection"]["pseudo"] == "on"
    pred, _ = predict(loaded, plain, ids)
    assert len(pred) == len(ids)
    # pseudo-labels reach only the classifier: climatology, capacity and date fits are the labelled ones
    # (κ is a selection on the inner OOF, so it may differ)
    base = fit(plain, labels, ids, options=STUB_OPTIONS)
    assert model["climatology"] == base["climatology"]
    for k in ("kind", "ridge", "ridge_export", "fallback_kwp"):
        assert model["capacity"][k] == base["capacity"][k]
    assert {k: model["date"][k] for k in ("offset_event", "start_lag", "contrast_cut")} == {
        k: base["date"][k] for k in ("offset_event", "start_lag", "contrast_cut")
    }
    assert model["classifier"]["logreg"] != base["classifier"]["logreg"]


def _flip(labels: pd.DataFrame, test_ids: list[str]) -> pd.DataFrame:
    flipped = labels.copy()
    for x in test_ids:
        row = flipped["customer_id"] == x
        was_pv = bool(flipped.loc[row, "pv_installed"].iloc[0])
        flipped.loc[row, "pv_installed"] = not was_pv
        flipped.loc[row, "pv_installation_date"] = pd.NaT if was_pv else pd.Timestamp("2023-05-01")
        flipped.loc[row, "pv_capacity_kwp"] = np.nan if was_pv else 7.5
    return flipped


@pytest.mark.parametrize("force", [False, True])
def test_label_flip_invariant_with_pool(data, monkeypatch, force):
    _, pooled, labels = data
    if force:  # every option adopted, so the pseudo path runs in the inner and the final fit
        monkeypatch.setattr(procedure, "_adopt", lambda base, cand: (True, 1.0))
    train_ids, test_ids = outer_splits(labels, repeats=1, folds=3)[0][0]
    # forced tail_guard / onset_first re-select the classifier (weekly here), which pseudo-labels never refit
    opts = procedure.DEFAULT_OPTIONS - {"tail_guard", "onset_first", "tobit"}
    m_a = fit(pooled, labels, train_ids, options=opts)
    m_b = fit(pooled, _flip(labels, test_ids), train_ids, options=opts)
    if force:
        assert m_a["selection"]["pseudo"] == "on" and m_a["selection"]["pseudo_counts"]["inner_pos"] > 0
    pdt.assert_frame_equal(predict(m_a, pooled, test_ids)[0], predict(m_b, pooled, test_ids)[0])


def test_selection_line_counts_pseudo():
    base = {"threshold": 0.5, "kappa": 1.0, "offset_event": 0}
    line = train._selection_line([{**base, "pseudo": p} for p in ("on", "off", "off")])
    assert line.endswith("pseudo off=2/3 on=1/3")
    assert "pseudo" not in train._selection_line([base])
