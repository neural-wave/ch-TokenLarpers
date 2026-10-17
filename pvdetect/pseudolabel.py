"""Pseudo-labelling (one round of self-training) on the unlabelled pool: a Tier A detection option.

Per fit, the classifier chosen so far scores the pool under that fit's climatology; confident pool
customers become weighted pseudo-examples and ONLY the classifiers are refit. Climatology, capacity and
date parameters keep coming from the labelled customers alone (pseudo-labels carry no dates or kWp).
Pool customers that are not confident are left out: unlabelled is never treated as negative.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from pvdetect import procedure
from pvdetect.pipeline import FeatureTable, classifier_proba, compute_features

POS_PROB = 0.95
NEG_PROB = 0.05
CLASS_CAP = 3  # pseudo-examples per class ≤ CLASS_CAP × the fit's labelled count of that class
WEIGHT = 0.5  # sample weight of a pseudo-example (labelled customers weigh 1)


def pool_features(table: FeatureTable, clim) -> pd.DataFrame:
    """Pool features under one fit's climatology; only the label-free part is cached (separately)."""
    pool = table.unlabelled_pool
    ids = list(pool)
    view = FeatureTable(ids=ids, customers=pool, cache=table.cache.setdefault("unlabelled_pool", {}))
    return compute_features(view, ids, clim)[0]


def pick_pseudo(prob: np.ndarray, max_pos: int, max_neg: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of pseudo-positives (prob ≥ 0.95) and pseudo-negatives (prob ≤ 0.05), most confident first
    up to the caps (ties keep pool order), each returned in ascending pool order."""
    down = np.argsort(-prob, kind="stable")
    up = np.argsort(prob, kind="stable")
    pos = down[prob[down] >= POS_PROB][: max(max_pos, 0)]
    neg = up[prob[up] <= NEG_PROB][: max(max_neg, 0)]
    return np.sort(pos), np.sort(neg)


def with_pseudo(table: FeatureTable, lab: pd.DataFrame, ids: list[str], names: list[str], comp, kind: str):
    """`comp` with logreg and CatBoost refit on its labelled `ids` plus pseudo-labels from its `kind`
    classifier, and the (positive, negative) pseudo counts. Unchanged when nothing qualifies."""
    y = lab.loc[ids, "pv_installed"].to_numpy(bool)
    if y.all() or not y.any():
        return comp, (0, 0)
    Fp = pool_features(table, comp.clim)
    if kind != "logreg" and "catboost_clf" not in comp.objects:  # as `_inner_oof`: CatBoost needs both classes
        kind = "logreg"
    prob = classifier_proba({"logreg": comp.logreg}, comp.objects, Fp, names, kind)
    pos, neg = pick_pseudo(prob, CLASS_CAP * int(y.sum()), CLASS_CAP * int((~y).sum()))
    if pos.size + neg.size == 0:
        return comp, (0, 0)
    Xp = Fp[names].to_numpy(float)
    X = np.vstack([comp.F[names].to_numpy(float), Xp[pos], Xp[neg]])
    yy = np.concatenate([y, np.ones(pos.size, bool), np.zeros(neg.size, bool)])
    w = np.concatenate([np.ones(y.size), np.full(pos.size + neg.size, WEIGHT)])
    logreg, objects = procedure._fit_classifiers(X, yy, ("logreg", "catboost"), w)
    return replace(comp, logreg=logreg, objects={**comp.objects, **objects}), (int(pos.size), int(neg.size))


def choose_pseudo(table, labels, lab, train_ids, splits, names, cap_names, setting, oof, sel, avg=True):
    """pseudo ∈ {off, on} on the same inner splits, after every other selection; adopted by the Step 11 rule.

    Returns the selection to use and the record for `selection` (aggregate counts only)."""
    counts: list[tuple[int, int]] = []

    def refit(comp, tr):
        comp, n = with_pseudo(table, lab, tr, names, comp, sel["classifier"])
        counts.append(n)
        return comp

    c_oof = procedure._inner_oof(table, labels, lab, train_ids, splits, names, cap_names, setting, refit=refit)
    cand = procedure._select(c_oof, avg)
    adopt, boot = procedure._adopt((oof, sel), (c_oof, cand))
    info = {
        "pseudo": "on" if adopt else "off",
        "pseudo_options": {"off": sel["inner_score"], "on": cand["inner_score"], "bootstrap_p": boot},
        "pseudo_counts": {
            "pool": len(table.unlabelled_pool),
            "inner_pos": sum(p for p, _ in counts),
            "inner_neg": sum(n for _, n in counts),
        },
    }
    if adopt:
        info["pseudo_labeller"] = sel["classifier"]  # the pre-pseudo classifier labels the pool in the final fit
    return (cand if adopt else sel), info


def final_pseudo(table: FeatureTable, lab: pd.DataFrame, train_ids: list[str], names: list[str], comp, info: dict):
    """The adopted option in the final fit: pseudo-labels from this fit's pre-pseudo classifier."""
    comp, (n_pos, n_neg) = with_pseudo(table, lab, train_ids, names, comp, info["pseudo_labeller"])
    info["pseudo_counts"].update(final_pos=n_pos, final_neg=n_neg)
    return comp
