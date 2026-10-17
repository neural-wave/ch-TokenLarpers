"""Training procedure: every data-driven choice happens inside `fit`, chosen by inner CV (Tier A)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from pvdetect.changepoint import shrunk_median
from pvdetect.evaluation import inner_splits
from pvdetect.features import EXPORT_CAPACITY_FEATURES, SCHEDULE_FEATURES, capacity_feature_names, feature_names
from pvdetect.model import CURRENT_VERSION, fit_logreg, fit_ridge, standardize_params
from pvdetect.pipeline import (
    SIGNED_SHARE,
    FeatureTable,
    capacity_log,
    capacity_matrix,
    classifier_proba,
    compute_features,
    regimes_and_dates,
    to_kwp,
)
from pvdetect.season import Climatology, climatology_from_curves, fit_climatology
from pvdetect.selection import (
    ape_array,
    bootstrap_prob_better,
    bootstrap_prob_better_outputs,
    choose_kappa,
    pick_threshold,
    threshold_scores,
)

CLASSIFIER_MARGIN = 2.0
BOOTSTRAP_MIN = 0.8
CAPACITY_MARGIN = 0.03
NO_CUT = -1e9  # contrast_cut when the present_at_start regime is disabled
BASE_SETTING = ("labelled_nonpv", "none")  # (climatology, evidence_scale); the conservative baseline
# Labelled commissioning dates may lag the visible onset by months.
OFFSET_ERROR_WINDOW = 180
OFFSET_CAP = 120
# Optional inner-CV options: E1 persistence block, E2 `avg` classifier, schedule-robust feature block.
# Without e1/e2 the procedure is exactly the pre-E1/E2 one (`train.py --no-e1e2`); schedule is opt-in.
ALL_OPTIONS = frozenset({"e1", "e2", "schedule"})
DEFAULT_OPTIONS = frozenset({"e1", "e2"})

CB_PARAMS = dict(
    iterations=300,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=5,
    random_seed=0,
    thread_count=1,
    verbose=False,
    allow_writing_files=False,
)


def export_enabled_for(table: FeatureTable, ids: list[str]) -> bool:
    """Signed-data test: export features only when enough customers have ≥1% negative readings."""
    signed = sum(1 for c in ids if table.customers.get(c) is not None and table.customers[c].neg_frac >= SIGNED_SHARE)
    return signed >= max(2, 0.1 * len(ids))


def _constant_linear(X: np.ndarray, intercept: float) -> dict:
    p = standardize_params(X)
    p.update(coef=[0.0] * X.shape[1], intercept=float(intercept))
    return p


@dataclass
class Components:
    clim: Climatology
    F: pd.DataFrame
    cps: dict
    date_params: dict
    logreg: dict | None
    ridge: dict
    ridge_export: dict | None
    fallback: float
    objects: dict


def _last_valid(table: FeatureTable, cid: str) -> pd.Timestamp | None:
    cd = table.customers.get(cid)
    if cd is None or cd.n_valid == 0:
        return None
    return cd.daily.index[cd.daily["valid"].to_numpy()][-1]


def fit_date_params(table: FeatureTable, lab: pd.DataFrame, ids: list[str], F: pd.DataFrame, cps: dict) -> dict:
    y = lab.loc[ids, "pv_installed"].to_numpy(bool)
    true_dates = lab.loc[ids, "pv_installation_date"]
    pv_ids = [c for c, on in zip(ids, y) if on and pd.notna(true_dates[c])]
    nonpv = [c for c, on in zip(ids, y) if not on]
    pv_post = []
    for c in pv_ids:
        last = _last_valid(table, c)
        val = F.at[c, "seasonal_contrast"]
        if last is not None and true_dates[c] <= last - pd.Timedelta(days=365) and np.isfinite(val):
            pv_post.append(val)
    np_c = [F.at[c, "seasonal_contrast"] for c in nonpv if np.isfinite(F.at[c, "seasonal_contrast"])]
    cut = (float(np.median(pv_post)) + float(np.median(np_c))) / 2 if len(pv_post) >= 3 and len(np_c) >= 3 else NO_CUT
    errs, gaps = [], []
    for c in pv_ids:
        cp = cps.get(c)
        cd = table.customers.get(c)
        if cp is not None and cp.found:
            e = (true_dates[c] - pd.Timestamp(cp.date)).days
            if abs(e) <= OFFSET_ERROR_WINDOW:
                errs.append(e)
        if cd is not None and true_dates[c] < pd.Timestamp(cd.first_date):
            gaps.append((pd.Timestamp(cd.first_date) - true_dates[c]).days)
    return {
        "offset_event": shrunk_median(errs, -OFFSET_CAP, OFFSET_CAP),
        "start_lag": shrunk_median(gaps, 0, 365) if len(gaps) >= 3 else 0,
        "contrast_cut": cut,
    }


def _fit_classifiers(
    X: np.ndarray, y: np.ndarray, clf_kinds: tuple[str, ...], weight: np.ndarray | None = None
) -> tuple[dict | None, dict]:
    """(logreg params or None, {"catboost_clf": model} when fitted); `weight` is a per-row sample weight."""
    from catboost import CatBoostClassifier

    both = y.any() and not y.all()
    objects: dict = {}
    logreg = None
    if "logreg" in clf_kinds:
        logreg = (
            fit_logreg(X, y, C=0.5, sample_weight=weight) if both else _constant_linear(X, 50.0 if y.all() else -50.0)
        )
    if "catboost" in clf_kinds and both:
        cb = CatBoostClassifier(auto_class_weights="Balanced", **CB_PARAMS)
        cb.fit(X, y.astype(int), sample_weight=weight)
        objects["catboost_clf"] = cb
    return logreg, objects


def _fit_components(
    table: FeatureTable,
    lab: pd.DataFrame,
    ids: list[str],
    names: list[str],
    cap_names: list[str],
    clf_kinds: tuple[str, ...],
    cap_kinds: tuple[str, ...],
    setting: tuple[str, str] = BASE_SETTING,
) -> Components:
    from catboost import CatBoostRegressor

    sub = lab.loc[ids]
    y = sub["pv_installed"].to_numpy(bool)
    dates = {c: d.date() for c, d in sub.loc[y, "pv_installation_date"].items() if pd.notna(d)}
    nonpv = [c for c, on in zip(ids, y) if not on]
    clim = _climatology(table, nonpv, dates, setting)
    F, cps = compute_features(table, ids, clim)
    date_params = fit_date_params(table, lab, ids, F, cps)
    logreg, objects = _fit_classifiers(F[names].to_numpy(float), y, clf_kinds)
    kwp = sub["pv_capacity_kwp"].to_numpy(float)
    pos = y & np.isfinite(kwp) & (kwp > 0)
    fallback = float(np.median(kwp[pos])) if pos.any() else 5.0
    Xc = capacity_matrix(F, cap_names)
    ridge = fit_ridge(Xc[pos], np.log(kwp[pos])) if pos.sum() >= 3 else _constant_linear(Xc, np.log(fallback))
    ridge_export = None
    if "export_kw" in cap_names:
        signed = pos & (F["neg_frac"].to_numpy(float) >= SIGNED_SHARE)
        if signed.sum() >= 3:
            Xe = capacity_matrix(F, list(EXPORT_CAPACITY_FEATURES))
            ridge_export = fit_ridge(Xe[signed], np.log(kwp[signed]))
    if "catboost" in cap_kinds and pos.sum() >= 3:
        cb_names = cap_names + (["neg_frac"] if "export_kw" in cap_names else [])
        reg = CatBoostRegressor(loss_function="RMSE", **CB_PARAMS)
        reg.fit(F.loc[pos, cb_names].to_numpy(float), np.log(kwp[pos]))
        objects["catboost_cap"] = reg
    return Components(clim, F, cps, date_params, logreg, ridge, ridge_export, fallback, objects)


def _climatology(table: FeatureTable, nonpv: list[str], pv_dates: dict, setting: tuple[str, str]) -> Climatology:
    """Climatology for one fit; unlabelled options use the label-free prior's curves/spread."""
    clim_kind, scale_kind = setting
    dailies = table.dailies()
    if clim_kind == "unlabelled_0.5":
        clim = climatology_from_curves(table.unlabelled.curves_q[0.5], dailies, pv_dates, "unlabelled_0.5")
    else:
        clim = fit_climatology(dailies, nonpv, pv_dates)
    if scale_kind == "unlabelled_spread":
        clim = replace(clim, evidence_scale=table.unlabelled.evidence_scale)
    return clim


def _inner_oof(
    table, labels, lab, train_ids, splits, names, cap_names, setting=BASE_SETTING, refit=None
) -> pd.DataFrame:
    """Pooled inner OOF outputs; `refit(comp, tr)` may replace the classifiers only (pseudo-labelling)."""
    rows = []
    for tr, va in splits:
        comp = _fit_components(
            table, lab, tr, names, cap_names, ("logreg", "catboost"), ("ridge", "catboost"), setting
        )
        if refit is not None:
            comp = refit(comp, tr)
        Fva, cps_va = compute_features(table, va, comp.clim)
        p_lr = classifier_proba({"logreg": comp.logreg}, {}, Fva, names, "logreg")
        p_cb = (
            classifier_proba({}, comp.objects, Fva, names, "catboost") if "catboost_clf" in comp.objects else p_lr
        )
        cap = {"ridge": comp.ridge, "ridge_export": comp.ridge_export}
        k_r = to_kwp(capacity_log(cap, comp.objects, Fva, cap_names, "ridge"), 1.0, comp.fallback)
        k_c = (
            to_kwp(capacity_log(cap, comp.objects, Fva, cap_names, "catboost"), 1.0, comp.fallback)
            if "catboost_cap" in comp.objects
            else np.full(len(va), np.nan)
        )
        regimes, d_weak = regimes_and_dates(table, va, Fva, cps_va, comp.date_params, "weak_candidate")
        _, d_first = regimes_and_dates(table, va, Fva, cps_va, comp.date_params, "first_date")
        for i, cid in enumerate(va):
            rows.append(
                {
                    "customer_id": cid,
                    "p_logreg": p_lr[i],
                    "p_catboost": p_cb[i],
                    "kwp_ridge": k_r[i],
                    "kwp_catboost": k_c[i],
                    "regime": regimes[i],
                    "date_weak": d_weak[i],
                    "date_first": d_first[i],
                }
            )
    oof = pd.DataFrame(rows).set_index("customer_id").loc[train_ids]
    t = lab.loc[train_ids]
    oof["t_pv"] = t["pv_installed"].to_numpy(bool)
    oof["t_date"] = t["pv_installation_date"].to_numpy()
    oof["t_kwp"] = t["pv_capacity_kwp"].to_numpy(float)
    return oof


def _err_days(pred_dates: pd.Series, true_dates: np.ndarray) -> np.ndarray:
    pdates = pd.to_datetime(pred_dates).to_numpy()
    e = (pdates - true_dates).astype("timedelta64[D]").astype(float)
    e[np.isnat(pdates) | np.isnat(true_dates)] = np.nan
    return e


def _d_mean(err: np.ndarray) -> float:
    return float(np.where(np.isfinite(err), np.maximum(0, 1 - np.abs(err) / 30), 0.0).mean()) if err.size else np.nan


def _p_mean(ape: np.ndarray) -> float:
    return float(np.where(np.isfinite(ape), np.maximum(0, 1 - ape / 0.5), 0.0).mean()) if ape.size else np.nan


def _proba(oof: pd.DataFrame, kind: str) -> np.ndarray:
    """Inner OOF probabilities of a classifier option; `avg` (E2) is the mean of logreg and CatBoost."""
    if kind == "avg":
        return (oof["p_logreg"].to_numpy(float) + oof["p_catboost"].to_numpy(float)) / 2
    return oof[f"p_{kind}"].to_numpy(float)


def _select(oof: pd.DataFrame, avg: bool = True) -> dict:
    """Tier A choices from pooled inner OOF predictions."""
    t_pv = oof["t_pv"].to_numpy(bool)
    t_dates = oof["t_date"].to_numpy("datetime64[ns]")
    err = _err_days(oof["date_weak"], t_dates)
    ape_r = ape_array(oof["kwp_ridge"].to_numpy(float), oof["t_kwp"].to_numpy(float))
    p_lr, p_cb = _proba(oof, "logreg"), _proba(oof, "catboost")
    t_lr, s_lr = pick_threshold(threshold_scores(p_lr, t_pv, err, ape_r))
    t_cb, s_cb = pick_threshold(threshold_scores(p_cb, t_pv, err, ape_r))
    clf_kind, thr, boot = "logreg", t_lr, None
    if s_cb - s_lr >= CLASSIFIER_MARGIN:
        boot = bootstrap_prob_better(p_lr, t_lr, p_cb, t_cb, t_pv, err, ape_r)
        if boot >= BOOTSTRAP_MIN:
            clf_kind, thr = "catboost", t_cb
    # E2: probability averaging is a third candidate, compared against the winner of the rule above.
    scores = {"logreg": s_lr, "catboost": s_cb, "avg": np.nan}
    boot_avg = None
    if avg and np.isfinite(p_cb).all():
        p_avg = _proba(oof, "avg")
        t_avg, scores["avg"] = pick_threshold(threshold_scores(p_avg, t_pv, err, ape_r))
        if scores["avg"] - scores[clf_kind] >= CLASSIFIER_MARGIN:
            boot_avg = bootstrap_prob_better(_proba(oof, clf_kind), thr, p_avg, t_avg, t_pv, err, ape_r)
            if boot_avg >= BOOTSTRAP_MIN:
                clf_kind, thr = "avg", t_avg
    prob = _proba(oof, clf_kind)
    tp = t_pv & (prob >= thr)
    true_k = oof["t_kwp"].to_numpy(float)
    P_r = _p_mean(ape_r[tp])
    cap_kind = "ridge"
    P_c = np.nan
    if oof["kwp_catboost"].notna().all():
        ape_c = ape_array(oof["kwp_catboost"].to_numpy(float), true_k)
        P_c = _p_mean(ape_c[tp])
        if np.isfinite(P_c) and np.isfinite(P_r) and P_c - P_r >= CAPACITY_MARGIN:
            cap_kind = "catboost"
    kappa = choose_kappa(oof[f"kwp_{cap_kind}"].to_numpy(float)[tp], true_k[tp])
    insuff = tp & (oof["regime"].to_numpy() == "insufficient")
    rule = "weak_candidate"
    if insuff.sum() >= 3:
        d_weak = _d_mean(err[insuff])
        d_first = _d_mean(_err_days(oof["date_first"], t_dates)[insuff])
        if d_first > d_weak:
            rule = "first_date"
    # B4: rescore the same threshold grid on the outputs that are actually submitted
    # (chosen capacity kind × κ, chosen date rule). One pass; same tie-breaking.
    final_dates = oof["date_first"] if rule == "first_date" else oof["date_weak"]
    err_final = _err_days(final_dates, t_dates)
    ape_final = ape_array(kappa * oof[f"kwp_{cap_kind}"].to_numpy(float), true_k)
    thr_final, s_final = pick_threshold(threshold_scores(prob, t_pv, err_final, ape_final))
    out = {
        "classifier": clf_kind,
        "threshold": thr_final,
        "threshold_initial": thr,
        "inner_score_logreg": s_lr,
        "inner_score_catboost": s_cb,
        "inner_score_avg": scores["avg"],
        "inner_score": s_final,
        "inner_score_initial": scores[clf_kind],
        "bootstrap_p": boot,
        "bootstrap_p_avg": boot_avg,
        "capacity": cap_kind,
        "inner_P_ridge": P_r,
        "inner_P_catboost": P_c,
        "kappa": kappa,
        "insufficient_rule": rule,
    }
    if not avg:  # the pre-E2 selection record, key for key
        del out["inner_score_avg"], out["bootstrap_p_avg"]
    return out


def _final_outputs(oof: pd.DataFrame, sel: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The outputs a configuration would submit: (prediction, date error days, capacity APE)."""
    pred = _proba(oof, sel["classifier"]) >= sel["threshold"]
    dates = oof["date_first"] if sel["insufficient_rule"] == "first_date" else oof["date_weak"]
    err = _err_days(dates, oof["t_date"].to_numpy("datetime64[ns]"))
    ape = ape_array(sel["kappa"] * oof[f"kwp_{sel['capacity']}"].to_numpy(float), oof["t_kwp"].to_numpy(float))
    return pred, err, ape


def _adopt(base: tuple[pd.DataFrame, dict], cand: tuple[pd.DataFrame, dict]) -> tuple[bool, float | None]:
    """Step 11 margin rule on final inner scores plus a paired bootstrap of the submitted outputs."""
    (oof_b, sel_b), (oof_c, sel_c) = base, cand
    if not sel_c["inner_score"] - sel_b["inner_score"] >= CLASSIFIER_MARGIN:
        return False, None
    t_pv = oof_b["t_pv"].to_numpy(bool)
    boot = bootstrap_prob_better_outputs(t_pv, *_final_outputs(oof_b, sel_b), *_final_outputs(oof_c, sel_c))
    return boot >= BOOTSTRAP_MIN, boot


def _choose_unlabelled(table, labels, lab, train_ids, splits, names, cap_names, oof, sel, avg=True):
    """Climatology first, then evidence_scale on top; every option on the same inner splits."""
    prior = table.unlabelled
    setting, cur = BASE_SETTING, (oof, sel)
    info: dict = {}
    options: dict = {}

    def evaluate(cand_setting):
        c_oof = _inner_oof(table, labels, lab, train_ids, splits, names, cap_names, cand_setting)
        return c_oof, _select(c_oof, avg)

    if prior.passed:
        cand_setting = ("unlabelled_0.5", "none")
        cand = evaluate(cand_setting)
        adopt, boot = _adopt(cur, cand)
        options["climatology"] = {
            "labelled_nonpv": cur[1]["inner_score"],
            "unlabelled_0.5": cand[1]["inner_score"],
            "bootstrap_p": boot,
        }
        if adopt:
            setting, cur = cand_setting, cand
    else:
        info["unlabelled_disabled_reason"] = prior.reason
    if prior.evidence_scale is not None:
        cand_setting = (setting[0], "unlabelled_spread")
        cand = evaluate(cand_setting)
        adopt, boot = _adopt(cur, cand)
        options["evidence_scale"] = {
            "none": cur[1]["inner_score"],
            "unlabelled_spread": cand[1]["inner_score"],
            "bootstrap_p": boot,
        }
        if adopt:
            setting, cur = cand_setting, cand
    else:
        info["evidence_scale_disabled_reason"] = "unlabelled spread unusable"
    settings = {"climatology": setting[0], "evidence_scale": setting[1]}
    return setting, cur[0], cur[1], {**settings, **info, "unlabelled_options": options}


def _choose_feature_block(table, labels, lab, train_ids, splits, export, cap_names, setting, oof, sel, avg=True):
    """E1 persistence block (classifier only) on the same inner splits, adopted by the Step 11 rule."""
    names, cand_names = feature_names(export), feature_names(export, persistence=True)
    c_oof = _inner_oof(table, labels, lab, train_ids, splits, cand_names, cap_names, setting)
    cand = (c_oof, _select(c_oof, avg))
    adopt, boot = _adopt((oof, sel), cand)
    info = {
        "feature_block": "persistence" if adopt else "base",
        "feature_options": {"base": sel["inner_score"], "persistence": cand[1]["inner_score"], "bootstrap_p": boot},
    }
    return (cand_names, *cand, info) if adopt else (names, oof, sel, info)


def _choose_schedule(table, labels, lab, train_ids, splits, names, cap_names, setting, oof, sel, avg=True):
    """Schedule-robust block (classifier only) on top of the current names, same inner splits and rule."""
    cand_names = names + list(SCHEDULE_FEATURES)
    c_oof = _inner_oof(table, labels, lab, train_ids, splits, cand_names, cap_names, setting)
    cand = (c_oof, _select(c_oof, avg))
    adopt, boot = _adopt((oof, sel), cand)
    info = {
        "schedule_block": "on" if adopt else "off",
        "schedule_options": {"off": sel["inner_score"], "on": cand[1]["inner_score"], "bootstrap_p": boot},
    }
    return (cand_names, *cand, info) if adopt else (names, oof, sel, info)


def fit(
    table: FeatureTable,
    labels: pd.DataFrame,
    train_ids: list[str],
    *,
    inner: bool = True,
    options: frozenset = DEFAULT_OPTIONS,
) -> dict:
    if not options <= ALL_OPTIONS:
        raise ValueError(f"unknown options: {sorted(options - ALL_OPTIONS)}")
    avg = "e2" in options
    lab = labels.set_index("customer_id")
    export = export_enabled_for(table, train_ids)
    names = feature_names(export)
    cap_names = capacity_feature_names(export)
    splits = inner_splits(labels, train_ids) if inner else []
    setting, unl_info = BASE_SETTING, {}
    feat_info = {"feature_block": "base"} if "e1" in options else {}
    sched_info = {"schedule_block": "off"} if "schedule" in options else {}
    pseudo_info = {"pseudo": "off"} if table.unlabelled_pool else {}
    if splits:
        oof = _inner_oof(table, labels, lab, train_ids, splits, names, cap_names)
        sel = _select(oof, avg)
        if table.unlabelled is not None:
            setting, oof, sel, unl_info = _choose_unlabelled(
                table, labels, lab, train_ids, splits, names, cap_names, oof, sel, avg
            )
        if "e1" in options:
            names, oof, sel, feat_info = _choose_feature_block(
                table, labels, lab, train_ids, splits, export, cap_names, setting, oof, sel, avg
            )
        if "schedule" in options:
            names, oof, sel, sched_info = _choose_schedule(
                table, labels, lab, train_ids, splits, names, cap_names, setting, oof, sel, avg
            )
        if table.unlabelled_pool:
            from pvdetect.pseudolabel import choose_pseudo  # pseudolabel imports this module

            sel, pseudo_info = choose_pseudo(
                table, labels, lab, train_ids, splits, names, cap_names, setting, oof, sel, avg
            )
    else:
        sel = {
            "classifier": "logreg",
            "threshold": 0.5,
            "threshold_initial": 0.5,
            "capacity": "ridge",
            "kappa": 1.0,
            "insufficient_rule": "weak_candidate",
            "inner_score": float("nan"),
            "inner_score_initial": float("nan"),
        }
        if table.unlabelled is not None:
            unl_info = {"climatology": setting[0], "evidence_scale": setting[1]}
    labeller = pseudo_info.get("pseudo_labeller")
    clf_kinds = ("logreg", "catboost") if sel["classifier"] == "avg" or labeller else ("logreg", sel["classifier"])
    comp = _fit_components(table, lab, train_ids, names, cap_names, clf_kinds, (sel["capacity"],), setting)
    if labeller:
        from pvdetect.pseudolabel import final_pseudo

        comp = final_pseudo(table, lab, train_ids, names, comp, pseudo_info)
    if sel["classifier"] in ("catboost", "avg") and "catboost_clf" not in comp.objects:
        sel["classifier"] = "logreg"
        sel["inner_score"] = sel["inner_score_logreg"]
    if sel["capacity"] == "catboost" and "catboost_cap" not in comp.objects:
        sel["capacity"] = "ridge"
    date = {**comp.date_params, "insufficient_rule": sel["insufficient_rule"]}
    sel.update(offset_event=date["offset_event"], start_lag=date["start_lag"], contrast_cut=date["contrast_cut"])
    sel.update(unl_info)
    sel.update(feat_info)
    sel.update(sched_info)
    sel.update(pseudo_info)
    uses_cb = sel["classifier"] in ("catboost", "avg")
    objects = {}
    if uses_cb:
        objects["catboost_clf"] = comp.objects["catboost_clf"]
    if sel["capacity"] == "catboost":
        objects["catboost_cap"] = comp.objects["catboost_cap"]
    return {
        "version": CURRENT_VERSION,
        "feature_names": names,
        "capacity_feature_names": cap_names,
        "export_enabled": export,
        "climatology": comp.clim.to_json(),
        "classifier": {
            "kind": sel["classifier"],
            "threshold": sel["threshold"],
            "logreg": comp.logreg if sel["classifier"] in ("logreg", "avg") else None,
            "catboost_file": "catboost_clf.cbm" if uses_cb else None,
        },
        "capacity": {
            "kind": sel["capacity"],
            "ridge": comp.ridge,
            "ridge_export": comp.ridge_export,
            "catboost_file": "catboost_cap.cbm" if sel["capacity"] == "catboost" else None,
            "kappa": sel["kappa"],
            "fallback_kwp": comp.fallback,
        },
        "date": date,
        "selection": sel,
        "extensions": {"enabled": False},
        "_objects": objects,
    }
