"""Customer-level nested cross-validation harness and decision log helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pvdetect.scoring import extension_bonus, main_score

REGIMES = ("event", "present_at_start", "insufficient")


def _min_class(y: np.ndarray) -> int:
    return int(min(y.sum(), (~y).sum()))


def outer_splits(labels: pd.DataFrame, repeats: int = 4, folds: int = 5, seed: int = 0) -> list[list[tuple[list[str], list[str]]]]:
    """Per repeat, a list of (train_ids, test_ids) customer splits."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    ids = labels["customer_id"].astype(str).to_numpy()
    y = labels["pv_installed"].to_numpy(bool)
    mc = _min_class(y)
    if mc < 2:
        raise ValueError("fewer than 2 customers in a class")
    k = min(folds, mc)
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed)
    out: list[list[tuple[list[str], list[str]]]] = [[] for _ in range(repeats)]
    for j, (tr, te) in enumerate(rskf.split(ids, y)):
        out[j // k].append((ids[tr].tolist(), ids[te].tolist()))
    return out


def inner_splits(labels: pd.DataFrame, train_ids: list[str], groups: list[str] | None = None) -> list[tuple[list[str], list[str]]]:
    """Common customer folds for every Tier A option of one fit (grouped by source customer)."""
    from sklearn.model_selection import StratifiedGroupKFold

    lab = labels.set_index("customer_id").loc[train_ids]
    y = lab["pv_installed"].to_numpy(bool)
    mc = _min_class(y)
    if mc < 2:
        return []
    ids = np.asarray(train_ids)
    g = np.asarray(groups if groups is not None else train_ids)
    sgkf = StratifiedGroupKFold(n_splits=min(5, mc), shuffle=True, random_state=0)
    return [(ids[tr].tolist(), ids[te].tolist()) for tr, te in sgkf.split(ids, y, g)]


def true_regime(truth: pd.DataFrame, first_date: pd.Series) -> pd.Series:
    td = pd.to_datetime(truth["pv_installation_date"])
    return pd.Series(np.where(td < first_date, "present_at_start", "event"), index=truth.index)


def regime_stats(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """D per predicted regime over TPs, and (true regime -> predicted regime) counts over TPs."""
    t = truth.set_index("customer_id")
    p = pred.set_index("customer_id").loc[t.index]
    tp = t["pv_installed"].to_numpy(bool) & p["pv_installed"].to_numpy(bool)
    tt, pp = t[tp], p[tp]
    err = (pp["pv_installation_date"] - tt["pv_installation_date"]).dt.days.to_numpy(float)
    d_i = np.where(np.isfinite(err), np.maximum(0, 1 - np.abs(err) / 30), 0.0)
    tr = np.where(tt["pv_installation_date"].to_numpy() < pp["first_date"].to_numpy(), "present_at_start", "event")
    stats = {}
    for r in REGIMES:
        m = pp["regime"].to_numpy() == r
        stats[r] = {"n": int(m.sum()), "D": float(d_i[m].mean()) if m.any() else float("nan")}
    confusion = {}
    for a in ("event", "present_at_start"):
        for b in REGIMES:
            confusion[f"{a}->{b}"] = int(((tr == a) & (pp["regime"].to_numpy() == b)).sum())
    return {"by_regime": stats, "confusion": confusion}


def bootstrap_ci(pred: pd.DataFrame, truth: pd.DataFrame, B: int = 1000, seed: int = 0) -> tuple[float, float]:
    """90% customer-bootstrap interval of the main score (uncertainty description only)."""
    rng = np.random.default_rng(seed)
    t = truth.reset_index(drop=True)
    p = pred.set_index("customer_id").loc[t["customer_id"]].reset_index()
    n = len(t)
    scores = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        tt = t.iloc[idx].assign(customer_id=[f"b{i}" for i in range(n)])
        pp = p.iloc[idx].assign(customer_id=[f"b{i}" for i in range(n)])
        scores.append(main_score(pp, tt)["score"])
    return float(np.quantile(scores, 0.05)), float(np.quantile(scores, 0.95))


@dataclass
class NestedResult:
    repeat_scores: list[dict]
    oof: list[pd.DataFrame]
    fit_infos: list[dict]
    regimes: list[dict]
    ext_scores: list[dict] = field(default_factory=list)
    ci90: tuple[float, float] = (float("nan"), float("nan"))

    def mean(self, key: str) -> float:
        return float(np.mean([s[key] for s in self.repeat_scores]))

    def sd(self, key: str) -> float:
        return float(np.std([s[key] for s in self.repeat_scores]))


def nested_cv(
    fit: Callable,
    predict: Callable,
    table,
    labels: pd.DataFrame,
    splits: list[list[tuple[list[str], list[str]]]],
    ext_labels: pd.DataFrame | None = None,
    bootstrap: int = 1000,
) -> NestedResult:
    """Outer CV around the full procedure; scores are computed on pooled repeat predictions."""
    res = NestedResult(repeat_scores=[], oof=[], fit_infos=[], regimes=[])
    for rep in splits:
        preds, exts = [], []
        for train_ids, test_ids in rep:
            model = fit(table, labels, train_ids)
            p, e = predict(model, table, test_ids)
            preds.append(p)
            exts.append(e)
            res.fit_infos.append(model.get("selection", {}))
        pooled = pd.concat(preds, ignore_index=True)
        truth = labels[labels["customer_id"].isin(pooled["customer_id"])]
        res.repeat_scores.append(main_score(pooled, truth))
        res.regimes.append(regime_stats(pooled, truth))
        res.oof.append(pooled)
        if ext_labels is not None:
            pooled_ext = pd.concat([e for e in exts if len(e)], ignore_index=True) if any(len(e) for e in exts) else exts[0]
            res.ext_scores.append(extension_bonus(pooled_ext, ext_labels))
    if bootstrap and res.oof:
        truth0 = labels[labels["customer_id"].isin(res.oof[0]["customer_id"])]
        res.ci90 = bootstrap_ci(res.oof[0], truth0, B=bootstrap)
    return res


def count_decisions(path: str | Path = "DECISIONS.md") -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.startswith("- "))
