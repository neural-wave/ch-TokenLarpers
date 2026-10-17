"""Training procedure: every data-driven choice happens inside `fit`, chosen by inner CV (Tier A)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from pvdetect.changepoint import shrunk_median, triangle_offset
from pvdetect.evaluation import inner_splits
from pvdetect.features import (
    EXPORT_CAPACITY_FEATURES,
    RECENT_FEATURES,
    SCHEDULE_FEATURES,
    SMALLPV_FEATURES,
    SUN_FEATURES,
    capacity_feature_names,
    feature_names,
)
from pvdetect.model import CURRENT_VERSION, fit_logreg, fit_ridge, standardize_params
from pvdetect.pipeline import (
    SIGNED_SHARE,
    FeatureTable,
    apply_cap_band,
    capacity_log,
    capacity_matrix,
    classifier_proba,
    compute_features,
    regimes_and_dates,
    sun_anomaly,
    to_kwp,
)
from pvdetect.season import Climatology, climatology_from_curves, fit_climatology, fit_sun_index
from pvdetect.selection import (
    ape_array,
    bootstrap_prob_better,
    bootstrap_prob_better_outputs,
    choose_kappa,
    pick_threshold,
    score_arrays,
    threshold_scores,
)
from pvdetect.weekly import customer_scores, fit_head

CLASSIFIER_MARGIN = 2.0
BOOTSTRAP_MIN = 0.8
CAPACITY_MARGIN = 0.03
NO_CUT = -1e9  # contrast_cut when the present_at_start regime is disabled
BASE_SETTING = ("labelled_nonpv", "none")  # (climatology, evidence_scale); the conservative baseline
# Labelled commissioning dates may lag the visible onset by months.
OFFSET_ERROR_WINDOW = 180
OFFSET_CAP = 120
START_LAG_CAP = 365
START_LAG_MIN_N = 3  # fewer present-at-start gaps: 0 (today) or, with `offset_tri`, a shrunk estimate
# Optional inner-CV options: E1 persistence block, E2 `avg` classifier, schedule-robust feature block,
# the weekly-evidence head (`weekly`/`avg_weekly` classifiers), the logreg C choice (`ctune`), the
# score-matched date offsets (`offset_tri`), the strict short-series detection rule (`recur_strict`) and
# the sunshine-coupling feature block (`sun`), the residual date-bias correction (`date_debias`) and the
# learned capacity band (`cap_ceiling`).
# `onset_first`: among qualifying candidates choose the earliest with evidence ≥ ONSET_FIRST_FRAC × best,
# and use only earlier-year recurrence (k < 0) to avoid post-install PV seasonality penalising the onset.
# `tail_guard`: require ≥ 45 valid days after τ and ≥ 2 post months for _persist (suppresses end-of-series
# holiday-shutdown artifacts).
# Without e1/e2/weekly/ctune/offset_tri/recur_strict/date_debias/cap_ceiling/onset_first/tail_guard the
# procedure is exactly the pre-E1/E2 one (`train.py --no-e1e2 --no-weekly --no-ctune --no-offset-tri
# --no-recur-strict --no-date-debias --no-cap-ceiling --no-onset-first --no-tail-guard`, plus
# `--no-recent --no-smallpv`); schedule and sun are opt-in.
# Recall-targeted feature blocks (missed PV): `recent` (tail evidence against the same weeks of earlier years and
# the customer's own earlier level) and `smallpv` (scale-free midday deficit against night/evening load).
# Both are in the default pool; `train.py --no-recent --no-smallpv` removes them.
ALL_OPTIONS = frozenset(
    {
        "e1", "e2", "schedule", "weekly", "ctune", "offset_tri", "recur_strict", "sun", "recent", "smallpv",
        "date_debias", "cap_ceiling", "onset_first", "tail_guard",
    }
)
DEFAULT_OPTIONS = frozenset(
    {
        "e1", "e2", "weekly", "ctune", "offset_tri", "recur_strict", "recent", "smallpv",
        "date_debias", "cap_ceiling", "onset_first", "tail_guard",
    }
)
REGIMES = ("event", "present_at_start", "insufficient")
DEBIAS_MIN_N = 3  # inner OOF true positives of a regime needed for its bias; fewer: no shift
DEBIAS_CAP = 365
# `cap_ceiling` band from the fit's labelled kWp: [q05 / margin, q95 × margin]; outside it, the training median.
CAP_BAND_Q = 0.05
CAP_BAND_MARGIN = 1.5
CAP_BAND_MIN_N = 3
LOGREG_C = 0.5  # the default; another C of C_OPTIONS only by the margin rule on the logreg path
C_OPTIONS = (0.1, 0.5, 2.0)
WEEKLY_KINDS = ("weekly", "avg_weekly")

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
    logreg_C: float = LOGREG_C
    recur_strict: bool = False
    cap_band: dict | None = None
    onset_first: bool = False
    tail_guard: bool = False


def _last_valid(table: FeatureTable, cid: str) -> pd.Timestamp | None:
    cd = table.customers.get(cid)
    if cd is None or cd.n_valid == 0:
        return None
    return cd.daily.index[cd.daily["valid"].to_numpy()][-1]


def fit_cap_band(kwp: np.ndarray) -> dict | None:
    """`cap_ceiling` band from labelled kWp: a learned ceiling q95 × margin (and floor q05 / margin); the
    robust fallback is the median. None with fewer than CAP_BAND_MIN_N positive values."""
    k = np.asarray(kwp, float)
    k = k[np.isfinite(k) & (k > 0)]
    if k.size < CAP_BAND_MIN_N:
        return None
    return {
        "lo": float(np.quantile(k, CAP_BAND_Q) / CAP_BAND_MARGIN),
        "hi": float(np.quantile(k, 1 - CAP_BAND_Q) * CAP_BAND_MARGIN),
        "fallback": float(np.median(k)),
    }


def regime_bias(err: np.ndarray, regimes: np.ndarray, mask: np.ndarray) -> dict:
    """Per-regime median signed date error (days, predicted − labelled) over `mask`; 0 for a regime with
    fewer than DEBIAS_MIN_N finite errors; rounded and clipped to ±DEBIAS_CAP."""
    out = {}
    for r in REGIMES:
        e = err[mask & (regimes == r) & np.isfinite(err)]
        b = float(np.median(e)) if e.size >= DEBIAS_MIN_N else 0.0
        out[r] = int(round(float(np.clip(b, -DEBIAS_CAP, DEBIAS_CAP))))
    return out


def _start_lag_tri(gaps: list[int]) -> int:
    """`offset_tri` start lag: the triangle optimum; with fewer than START_LAG_MIN_N gaps it is shrunk
    toward 0 by n/(n+5), so one or two examples move the lag partially."""
    lag = triangle_offset(gaps, 0, START_LAG_CAP)
    n = len(gaps)
    return lag if n >= START_LAG_MIN_N else int(round(lag * n / (n + 5)))


def fit_date_params(
    table: FeatureTable, lab: pd.DataFrame, ids: list[str], F: pd.DataFrame, cps: dict, *, tri: bool = False
) -> dict:
    """Date-regime parameters; `tri` (option `offset_tri`) replaces the shrunk medians by the
    score-matched `triangle_offset` on the same error lists, with the small-sample start-lag fallback."""
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
    if tri:
        offset, lag = triangle_offset(errs, -OFFSET_CAP, OFFSET_CAP), _start_lag_tri(gaps)
    else:
        offset = shrunk_median(errs, -OFFSET_CAP, OFFSET_CAP)
        lag = shrunk_median(gaps, 0, START_LAG_CAP) if len(gaps) >= START_LAG_MIN_N else 0
    return {"offset_event": offset, "start_lag": lag, "contrast_cut": cut}


def _fit_classifiers(
    X: np.ndarray, y: np.ndarray, clf_kinds: tuple[str, ...], weight: np.ndarray | None = None, C: float = LOGREG_C
) -> tuple[dict | None, dict]:
    """(logreg params or None, {"catboost_clf": model} when fitted); `weight` is a per-row sample weight."""
    from catboost import CatBoostClassifier

    both = y.any() and not y.all()
    objects: dict = {}
    logreg = None
    if "logreg" in clf_kinds:
        logreg = (
            fit_logreg(X, y, C=C, sample_weight=weight) if both else _constant_linear(X, 50.0 if y.all() else -50.0)
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
    C: float = LOGREG_C,
    *,
    embedding=None,
    projection: dict | None = None,
    offset_tri: bool = False,
    recur_strict: bool = False,
    onset_first: bool = False,
    tail_guard: bool = False,
) -> Components:
    from catboost import CatBoostRegressor

    sub = lab.loc[ids]
    y = sub["pv_installed"].to_numpy(bool)
    dates = {c: d.date() for c, d in sub.loc[y, "pv_installation_date"].items() if pd.notna(d)}
    nonpv = [c for c, on in zip(ids, y) if not on]
    clim = _climatology(table, nonpv, dates, setting, sun="sun_coupling" in names)
    F, cps = compute_features(
        table, ids, clim,
        recur_strict=recur_strict, onset_first=onset_first, tail_guard=tail_guard,
        embedding=embedding, projection=projection,
    )
    date_params = fit_date_params(table, lab, ids, F, cps, tri=offset_tri)
    logreg, objects = _fit_classifiers(F[names].to_numpy(float), y, clf_kinds, C=C)
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
    band = fit_cap_band(kwp[pos])
    return Components(clim, F, cps, date_params, logreg, ridge, ridge_export, fallback, objects, C, recur_strict, band, onset_first, tail_guard)


def _climatology(
    table: FeatureTable, nonpv: list[str], pv_dates: dict, setting: tuple[str, str], *, sun: bool = False
) -> Climatology:
    """Climatology for one fit; unlabelled options use the label-free prior's curves/spread; `sun` adds the
    sunshine index of the fit's labelled PV customers (only when the classifier names include its feature)."""
    clim_kind, scale_kind = setting
    dailies = table.dailies()
    if clim_kind == "unlabelled_0.5":
        clim = climatology_from_curves(table.unlabelled.curves_q[0.5], dailies, pv_dates, "unlabelled_0.5")
    else:
        clim = fit_climatology(dailies, nonpv, pv_dates)
    if scale_kind == "unlabelled_spread":
        clim = replace(clim, evidence_scale=table.unlabelled.evidence_scale)
    if sun:
        clim = replace(clim, sun=fit_sun_index({c: sun_anomaly(table, c) for c in pv_dates}, pv_dates))
    return clim


def _inner_oof(
    table,
    labels,
    lab,
    train_ids,
    splits,
    names,
    cap_names,
    setting=BASE_SETTING,
    refit=None,
    *,
    C=LOGREG_C,
    alt_cs=(),
    weekly=False,
    offset_tri=False,
    recur_strict=False,
    onset_first=False,
    tail_guard=False,
    tri_columns=False,
    embedding=None,
    raw=None,
) -> pd.DataFrame:
    """Pooled inner OOF outputs; `refit(comp, tr)` may replace the classifiers only (pseudo-labelling).

    `C` is the logreg C; `alt_cs` adds `p_logreg_C{c}` columns (logreg refit at other C on the same
    features, without `refit`); `weekly` adds `p_weekly`, the customer score of a weekly head trained on
    the inner-train customers' windows under that split's climatology. `offset_tri` and `recur_strict`
    are the date options of every fit; `tri_columns` adds `date_weak_tri`/`date_first_tri`, the dates
    under that split's `offset_tri` parameters (same features, regimes and contrast cut). `embedding`
    (a frozen encoder) with its label-free `raw` customer features adds the embedding block, its
    projection fitted on each split's inner-train customers only."""
    from pvdetect.embedding import fit_projection

    rows = []
    for fold, (tr, va) in enumerate(splits):
        projection = fit_projection(raw, tr) if embedding is not None else None
        comp = _fit_components(
            table,
            lab,
            tr,
            names,
            cap_names,
            ("logreg", "catboost"),
            ("ridge", "catboost"),
            setting,
            C,
            embedding=embedding,
            projection=projection,
            offset_tri=offset_tri,
            recur_strict=recur_strict,
            onset_first=onset_first,
            tail_guard=tail_guard,
        )
        if refit is not None:
            comp = refit(comp, tr)
        Fva, cps_va = compute_features(
            table, va, comp.clim,
            recur_strict=recur_strict, onset_first=onset_first, tail_guard=tail_guard,
            embedding=embedding, projection=projection,
        )
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
        extra = {}
        if tri_columns:
            tri_params = {**comp.date_params, **fit_date_params(table, lab, tr, comp.F, comp.cps, tri=True)}
            _, extra["date_weak_tri"] = regimes_and_dates(table, va, Fva, cps_va, tri_params, "weak_candidate")
            _, extra["date_first_tri"] = regimes_and_dates(table, va, Fva, cps_va, tri_params, "first_date")
        if alt_cs:
            X_tr, y_tr = comp.F[names].to_numpy(float), lab.loc[tr, "pv_installed"].to_numpy(bool)
            for c in alt_cs:
                lr, _ = _fit_classifiers(X_tr, y_tr, ("logreg",), C=c)
                extra[f"p_logreg_C{c}"] = classifier_proba({"logreg": lr}, {}, Fva, names, "logreg")
        if weekly:
            extra["p_weekly"] = customer_scores(fit_head(table, lab, tr, comp.clim), table, va, comp.clim)
        band = comp.cap_band or {"lo": np.nan, "hi": np.nan, "fallback": np.nan}
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
                    **{k: v[i] for k, v in extra.items()},
                    "fold": fold,
                    "cap_lo": band["lo"],
                    "cap_hi": band["hi"],
                    "cap_fb": band["fallback"],
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
    """Inner OOF probabilities of a classifier option; `avg` (E2) is the mean of logreg and CatBoost,
    `avg_weekly` the mean of logreg and the weekly head's customer score."""
    if kind == "avg":
        return (oof["p_logreg"].to_numpy(float) + oof["p_catboost"].to_numpy(float)) / 2
    if kind == "avg_weekly":
        return (oof["p_logreg"].to_numpy(float) + oof["p_weekly"].to_numpy(float)) / 2
    return oof[f"p_{kind}"].to_numpy(float)


def _select(oof: pd.DataFrame, avg: bool = True) -> dict:
    """Tier A choices from pooled inner OOF predictions. The weekly candidates take part when the inner
    OOF carries weekly-head scores (`p_weekly`, option `weekly`)."""
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
    # Weekly-evidence head: `weekly`, then `avg_weekly`, each against the current winner by the same rule.
    boot_weekly: dict = {}
    if "p_weekly" in oof.columns:
        for kind in WEEKLY_KINDS:
            p_k = _proba(oof, kind)
            t_k, scores[kind] = pick_threshold(threshold_scores(p_k, t_pv, err, ape_r))
            boot_weekly[kind] = None
            if scores[kind] - scores[clf_kind] >= CLASSIFIER_MARGIN:
                boot_weekly[kind] = bootstrap_prob_better(_proba(oof, clf_kind), thr, p_k, t_k, t_pv, err, ape_r)
                if boot_weekly[kind] >= BOOTSTRAP_MIN:
                    clf_kind, thr = kind, t_k
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
    kappa_initial = choose_kappa(oof[f"kwp_{cap_kind}"].to_numpy(float)[tp], true_k[tp])
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
    kwp_cap = oof[f"kwp_{cap_kind}"].to_numpy(float)
    # Submitted capacities are clipped to [0.5, 500] kWp (`to_kwp`); score the same outputs.
    ape_initial = ape_array(np.clip(kappa_initial * kwp_cap, 0.5, 500.0), true_k)
    thr_final, s_final = pick_threshold(threshold_scores(prob, t_pv, err_final, ape_initial))
    # κ once more on the TPs at the final threshold (same grid and tie rule), and the final score with it.
    tp_final = t_pv & (prob >= thr_final)
    kappa = choose_kappa(kwp_cap[tp_final], true_k[tp_final])
    if kappa != kappa_initial:
        s_final = score_arrays(
            t_pv, prob >= thr_final, err_final, ape_array(np.clip(kappa * kwp_cap, 0.5, 500.0), true_k)
        )
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
        "kappa_initial": kappa_initial,
        "insufficient_rule": rule,
    }
    if not avg:
        del out["inner_score_avg"], out["bootstrap_p_avg"]
    for kind in boot_weekly:
        out[f"inner_score_{kind}"] = scores[kind]
        out[f"bootstrap_p_{kind}"] = boot_weekly[kind]
    return out
def _full_inner_predictions(oof: pd.DataFrame, sel: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    prob = _proba(oof, sel["classifier"])
    raw_kwp = oof[f"kwp_{sel['capacity']}"].to_numpy(float)
    kwp = np.clip(float(sel.get("kappa", 1.0)) * raw_kwp, 0.5, 500.0)
    regime = oof["regime"].to_numpy()
    weak = oof["date_weak"].to_numpy()
    first = oof["date_first"].to_numpy()
    dates = weak.copy()
    if sel.get("insufficient_rule") == "first_date":
        dates[regime == "insufficient"] = first[regime == "insufficient"]
    pred = pd.DataFrame(
        {
            "customer_id": oof.index.astype(str),
            "pv_installed": prob >= float(sel["threshold"]),
            "pv_installation_date": dates,
            "pv_capacity_kwp": kwp,
        }
    )
    truth = pd.DataFrame(
        {
            "customer_id": oof.index.astype(str),
            "pv_installed": oof["t_pv"].to_numpy(bool),
            "pv_installation_date": oof["t_date"].to_numpy(),
            "pv_capacity_kwp": oof["t_kwp"].to_numpy(float),
        }
    )
    return pred, truth


def _full_inner_score(oof: pd.DataFrame, sel: dict) -> float:
    from pvdetect.scoring import main_score

    pred, truth = _full_inner_predictions(oof, sel)
    return float(main_score(pred, truth)["score"])


def _score_precomputed(pred: pd.DataFrame, truth: pd.DataFrame):
    t_pv = truth["pv_installed"].to_numpy(bool)
    p_pv = pred["pv_installed"].to_numpy(bool)
    t_dates = pd.to_datetime(truth["pv_installation_date"]).reset_index(drop=True)
    p_dates = pd.to_datetime(pred["pv_installation_date"]).reset_index(drop=True)
    days = np.abs((p_dates - t_dates).dt.days.to_numpy(float))
    d_i = np.where(np.isfinite(days), np.maximum(0.0, 1.0 - days / 30.0), 0.0)
    t_cap = truth["pv_capacity_kwp"].to_numpy(float)
    p_cap = pred["pv_capacity_kwp"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(p_cap - t_cap) / t_cap
    p_i = np.where(np.isfinite(ape) & (p_cap > 0) & (t_cap > 0), np.maximum(0.0, 1.0 - ape / 0.5), 0.0)
    return t_pv, p_pv, d_i, p_i


def _score_from_arrays(t_pv, p_pv, d_i, p_i, idx):
    t_idx = t_pv[idx]
    p_idx = p_pv[idx]
    tp_mask = t_idx & p_idx
    tp = int(tp_mask.sum())
    if tp == 0:
        return 0.0
    fp = int((~t_idx & p_idx).sum())
    fn = int((t_idx & ~p_idx).sum())
    f1 = 2.0 * tp / (2.0 * tp + fp + fn)
    D = float(d_i[idx][tp_mask].mean())
    P = float(p_i[idx][tp_mask].mean())
    return float(f1 * (70.0 + 15.0 * D + 15.0 * P))


def _paired_bootstrap_stability(
    base_oof: pd.DataFrame, base_sel: dict, emb_oof: pd.DataFrame, emb_sel: dict, B: int = 1000
) -> float:
    """Fast paired bootstrap equivalent to the official score for embedding adoption."""
    bp, truth = _full_inner_predictions(base_oof, base_sel)
    ep, _ = _full_inner_predictions(emb_oof, emb_sel)
    tb, pb, db, capb = _score_precomputed(bp, truth)
    te, pe, de, cape = _score_precomputed(ep, truth)
    rng = np.random.default_rng(0)
    n = len(truth)
    better = 0
    for _ in range(B):
        idx = rng.integers(0, n, n)
        sb = _score_from_arrays(tb, pb, db, capb, idx)
        se = _score_from_arrays(te, pe, de, cape, idx)
        better += se > sb
    return better / B


def _choose_logreg_c(oof: pd.DataFrame, sel: dict, avg: bool = True) -> tuple[float, pd.DataFrame, dict, dict]:
    """Logreg C on the first inner OOF (it carries `p_logreg_C{c}` for every other C of C_OPTIONS).

    The default C stays unless another C's logreg path (threshold-picked inner score of the logreg
    probabilities) beats it by CLASSIFIER_MARGIN with bootstrap ≥ BOOTSTRAP_MIN; among such Cs the best
    score wins (ties: C_OPTIONS order). Dates, capacities and the other classifiers do not depend on the
    logreg C, so the chosen C's inner OOF is this one with `p_logreg` replaced; the selection is rerun on it.
    Returns (C, inner OOF without the extra columns, selection, record)."""
    t_pv = oof["t_pv"].to_numpy(bool)
    err = _err_days(oof["date_weak"], oof["t_date"].to_numpy("datetime64[ns]"))
    ape_r = ape_array(oof["kwp_ridge"].to_numpy(float), oof["t_kwp"].to_numpy(float))
    cols = {c: "p_logreg" if c == LOGREG_C else f"p_logreg_C{c}" for c in C_OPTIONS}
    picks = {c: pick_threshold(threshold_scores(oof[col].to_numpy(float), t_pv, err, ape_r)) for c, col in cols.items()}
    t_base, s_base = picks[LOGREG_C]
    best, boots = LOGREG_C, {}
    for c in C_OPTIONS:
        if c == LOGREG_C:
            continue
        t_c, s_c = picks[c]
        boots[c] = None
        if s_c - s_base >= CLASSIFIER_MARGIN:
            p_base, p_c = oof["p_logreg"].to_numpy(float), oof[cols[c]].to_numpy(float)
            boots[c] = bootstrap_prob_better(p_base, t_base, p_c, t_c, t_pv, err, ape_r)
            if boots[c] >= BOOTSTRAP_MIN and (best == LOGREG_C or s_c > picks[best][1]):
                best = c
    extra = [cols[c] for c in C_OPTIONS if c != LOGREG_C]
    oof = oof.assign(p_logreg=oof[cols[best]]).drop(columns=extra)
    if best != LOGREG_C:
        sel = _select(oof, avg)
    info = {
        "logreg_C": best,
        "logreg_C_options": {
            **{str(c): picks[c][1] for c in C_OPTIONS},
            "bootstrap_p": {str(c): b for c, b in boots.items()},
        },
    }
    return best, oof, sel, info


def _select_outputs(oof: pd.DataFrame, sel: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outputs a configuration would submit: prediction, date error, capacity APE."""
    pred = _proba(oof, sel["classifier"]) >= sel["threshold"]
    dates = oof["date_first"] if sel["insufficient_rule"] == "first_date" else oof["date_weak"]
    err = _err_days(dates, oof["t_date"].to_numpy("datetime64[ns]"))
    ape = ape_array(sel["kappa"] * oof[f"kwp_{sel['capacity']}"].to_numpy(float), oof["t_kwp"].to_numpy(float))
    return pred, err, ape


def _adopt(base: tuple[pd.DataFrame, dict], cand: tuple[pd.DataFrame, dict]) -> tuple[bool, float | None]:
    """Adopt a candidate only after score margin and paired official-output bootstrap."""
    (oof_b, sel_b), (oof_c, sel_c) = base, cand
    if not sel_c["inner_score"] - sel_b["inner_score"] >= CLASSIFIER_MARGIN:
        return False, None
    t_pv = oof_b["t_pv"].to_numpy(bool)
    boot = bootstrap_prob_better_outputs(t_pv, *_select_outputs(oof_b, sel_b), *_select_outputs(oof_c, sel_c))
    return boot >= BOOTSTRAP_MIN, boot


def _choose_offset_tri(oof: pd.DataFrame, sel: dict, avg: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    """`offset_tri` on the first inner OOF, which carries the `_tri` dates of the same inner fits; the
    Step 11 margin rule decides. Returns (inner OOF without the `_tri` columns, selection, record)."""
    tri = ["date_weak_tri", "date_first_tri"]
    base = oof.drop(columns=tri)
    c_oof = base.assign(date_weak=oof["date_weak_tri"], date_first=oof["date_first_tri"])
    cand = (c_oof, _select(c_oof, avg))
    adopt, boot = _adopt((base, sel), cand)
    info = {
        "offset_tri": "on" if adopt else "off",
        "offset_tri_options": {"off": sel["inner_score"], "on": cand[1]["inner_score"], "bootstrap_p": boot},
    }
    return (*cand, info) if adopt else (base, sel, info)


def _debias_oof(oof: pd.DataFrame, sel: dict) -> tuple[pd.DataFrame, dict]:
    """Dates of the inner OOF shifted by the per-regime bias of the true positives (at the selection's
    classifier and threshold, final date rule). Cross-fitted: each inner validation fold is shifted by the
    bias of the other folds' rows, so a customer's own label never moves its date. Also returns the bias
    over all rows (the fit's training customers), used by the final model."""
    t_pv = oof["t_pv"].to_numpy(bool)
    tp = t_pv & (_proba(oof, sel["classifier"]) >= sel["threshold"])
    dates = oof["date_first"] if sel["insufficient_rule"] == "first_date" else oof["date_weak"]
    err = _err_days(dates, oof["t_date"].to_numpy("datetime64[ns]"))
    regimes = oof["regime"].to_numpy()
    folds = oof["fold"].to_numpy()
    shift = np.zeros(len(oof))
    for k in np.unique(folds):
        bias = regime_bias(err, regimes, tp & (folds != k))
        rows = folds == k
        shift[rows] = [bias[r] for r in regimes[rows]]
    delta = pd.to_timedelta(shift, unit="D")
    out = oof.assign(
        date_weak=pd.to_datetime(oof["date_weak"]) - delta, date_first=pd.to_datetime(oof["date_first"]) - delta
    )
    return out, regime_bias(err, regimes, tp)


def _choose_date_debias(oof: pd.DataFrame, sel: dict, avg: bool = True) -> tuple[pd.DataFrame, dict, dict, dict | None]:
    """`date_debias` on the selected inner OOF, adopted by the Step 11 rule. Returns (OOF, selection,
    record, per-regime bias for the final model or None)."""
    if "fold" not in oof.columns:  # an inner OOF built without fold ids (test stubs)
        return oof, sel, {"date_debias": "off", "date_debias_disabled_reason": "inner OOF has no fold ids"}, None
    c_oof, bias = _debias_oof(oof, sel)
    c_sel = _select(c_oof, avg)
    c_sel["inner_score"] = _full_inner_score(c_oof, c_sel)
    adopt, boot = _adopt((oof, sel), (c_oof, c_sel))
    info = {
        "date_debias": "on" if adopt else "off",
        "date_debias_options": {"off": sel["inner_score"], "on": c_sel["inner_score"], "bootstrap_p": boot, "bias": bias},
    }
    return (c_oof, c_sel, info, bias) if adopt else (oof, sel, info, None)


def _band_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """Inner OOF capacities with each fold's learned band applied (`cap_lo`/`cap_hi`/`cap_fb`, fitted on
    that fold's inner-train labels)."""
    band = {"lo": oof["cap_lo"].to_numpy(float), "hi": oof["cap_hi"].to_numpy(float), "fallback": oof["cap_fb"].to_numpy(float)}
    # folds without a band carry NaN bounds: both comparisons are False there, the capacity stays
    return oof.assign(**{c: apply_cap_band(oof[c].to_numpy(float), band) for c in ("kwp_ridge", "kwp_catboost")})


def _choose_cap_ceiling(oof: pd.DataFrame, sel: dict, avg: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    """`cap_ceiling` on the selected inner OOF, adopted by the Step 11 rule."""
    if not {"cap_lo", "cap_hi", "cap_fb"} <= set(oof.columns):  # an inner OOF built without fold bands
        return oof, sel, {"cap_ceiling": "off", "cap_ceiling_disabled_reason": "inner OOF has no fold bands"}
    c_oof = _band_oof(oof)
    c_sel = _select(c_oof, avg)
    c_sel["inner_score"] = _full_inner_score(c_oof, c_sel)
    adopt, boot = _adopt((oof, sel), (c_oof, c_sel))
    info = {
        "cap_ceiling": "on" if adopt else "off",
        "cap_ceiling_options": {"off": sel["inner_score"], "on": c_sel["inner_score"], "bootstrap_p": boot},
    }
    return (c_oof, c_sel, info) if adopt else (oof, sel, info)


def _choose_recur_strict(
    table, labels, lab, train_ids, splits, names, cap_names, oof, sel, avg=True, C=LOGREG_C, weekly=False, dkw=None
):
    """`recur_strict` (short-series detection rule) on the same inner splits, adopted by the Step 11 rule."""
    c_oof = _inner_oof(
        table, labels, lab, train_ids, splits, names, cap_names, C=C, weekly=weekly, **(dkw or {}), recur_strict=True
    )
    cand = (c_oof, _select(c_oof, avg))
    adopt, boot = _adopt((oof, sel), cand)
    info = {
        "recur_strict": "on" if adopt else "off",
        "recur_strict_options": {"off": sel["inner_score"], "on": cand[1]["inner_score"], "bootstrap_p": boot},
    }
    return (*cand, info) if adopt else (oof, sel, info)


def _choose_unlabelled(
    table, labels, lab, train_ids, splits, names, cap_names, oof, sel, avg=True, C=LOGREG_C, weekly=False, dkw=None
):
    """Climatology first, then evidence_scale on top; every option uses the same splits."""
    prior = table.unlabelled
    setting, cur = BASE_SETTING, (oof, sel)
    info: dict = {}
    options: dict = {}

    def evaluate(cand_setting):
        c_oof = _inner_oof(
            table, labels, lab, train_ids, splits, names, cap_names, cand_setting, C=C, weekly=weekly, **(dkw or {})
        )
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


def _choose_feature_block(
    table, labels, lab, train_ids, splits, export, cap_names, setting, oof, sel, avg=True, C=LOGREG_C, weekly=False,
    dkw=None,
):
    """E1 persistence block on the same inner splits, adopted by the Step 11 rule."""
    names, cand_names = feature_names(export), feature_names(export, persistence=True)
    c_oof = _inner_oof(
        table, labels, lab, train_ids, splits, cand_names, cap_names, setting, C=C, weekly=weekly, **(dkw or {})
    )
    cand = (c_oof, _select(c_oof, avg))
    adopt, boot = _adopt((oof, sel), cand)
    info = {
        "feature_block": "persistence" if adopt else "base",
        "feature_options": {"base": sel["inner_score"], "persistence": cand[1]["inner_score"], "bootstrap_p": boot},
    }
    return (cand_names, *cand, info) if adopt else (names, oof, sel, info)


def _choose_block(
    block, extra, table, labels, lab, train_ids, splits, names, cap_names, setting, oof, sel, avg=True, C=LOGREG_C,
    weekly=False, dkw=None,
):
    """Optional feature block `extra` (classifier only) on top of the current names, same inner splits and rule;
    reported as `{block}_block` and `{block}_options`."""
    cand_names = names + list(extra)
    c_oof = _inner_oof(
        table, labels, lab, train_ids, splits, cand_names, cap_names, setting, C=C, weekly=weekly, **(dkw or {})
    )
    cand = (c_oof, _select(c_oof, avg))
    adopt, boot = _adopt((oof, sel), cand)
    info = {
        f"{block}_block": "on" if adopt else "off",
        f"{block}_options": {"off": sel["inner_score"], "on": cand[1]["inner_score"], "bootstrap_p": boot},
    }
    return (cand_names, *cand, info) if adopt else (names, oof, sel, info)


def _choose_onset_first(
    table, labels, lab, train_ids, splits, names, cap_names, oof, sel, avg=True, C=LOGREG_C, weekly=False, dkw=None
):
    """`onset_first` (earliest-qualifying-candidate date rule + earlier-year-only recurrence) is the fixed
    dating rule while the option is in the pool: the round-2 targeted comparison adopted it, and re-deciding it
    per fit by the margin rule let inner-CV noise switch the final fit back (DECISIONS.md). It changes dates and
    regimes only, never the features; both inner scores are still recorded."""
    c_oof = _inner_oof(
        table, labels, lab, train_ids, splits, names, cap_names, C=C, weekly=weekly, **(dkw or {}), onset_first=True
    )
    cand = (c_oof, _select(c_oof, avg))
    info = {
        "onset_first": "on",
        "onset_first_options": {"off": sel["inner_score"], "on": cand[1]["inner_score"], "bootstrap_p": None},
    }
    return (*cand, info)


def _choose_tail_guard(
    table, labels, lab, train_ids, splits, names, cap_names, oof, sel, avg=True, C=LOGREG_C, weekly=False, dkw=None
):
    """`tail_guard` (≥ 45 valid days after τ, ≥ 2 post months for _persist) on the same inner splits,
    adopted by the Step 11 rule."""
    c_oof = _inner_oof(
        table, labels, lab, train_ids, splits, names, cap_names, C=C, weekly=weekly, **(dkw or {}), tail_guard=True
    )
    cand = (c_oof, _select(c_oof, avg))
    adopt, boot = _adopt((oof, sel), cand)
    info = {
        "tail_guard": "on" if adopt else "off",
        "tail_guard_options": {"off": sel["inner_score"], "on": cand[1]["inner_score"], "bootstrap_p": boot},
    }
    return (*cand, info) if adopt else (oof, sel, info)


def _final_outputs(oof: pd.DataFrame, sel: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _select_outputs(oof, sel)

def fit(
    table: FeatureTable,
    labels: pd.DataFrame,
    train_ids: list[str],
    *,
    inner: bool = True,
    encoder=None,
    feature_block: str = "auto",
    options: frozenset = DEFAULT_OPTIONS,
) -> dict:
    """Fit option-selected baseline features, optionally augmented by a frozen encoder."""
    if feature_block not in {"auto", "baseline", "embedding"}:
        raise ValueError("feature_block must be auto, baseline, or embedding")
    if not options <= ALL_OPTIONS:
        raise ValueError(f"unknown options: {sorted(options - ALL_OPTIONS)}")
    avg = "e2" in options
    weekly = "weekly" in options
    if encoder is not None:
        from pvdetect.embedding import FrozenEncoder

        if not isinstance(encoder, FrozenEncoder):
            encoder = FrozenEncoder.load(encoder)
    if feature_block == "embedding" and encoder is None:
        raise ValueError("feature_block=embedding requires an encoder checkpoint")
    lab = labels.set_index("customer_id")
    export = export_enabled_for(table, train_ids)
    base_names = feature_names(export)
    cap_names = capacity_feature_names(export)
    splits = inner_splits(labels, train_ids) if inner else []
    setting, unl_info = BASE_SETTING, {}
    names = base_names
    C, c_info = LOGREG_C, ({"logreg_C": LOGREG_C} if "ctune" in options else {})
    feat_info = {"feature_block": "base"} if "e1" in options else {}
    sched_info = {"schedule_block": "off"} if "schedule" in options else {}
    sun_info = {"sun_block": "off"} if "sun" in options else {}
    recent_info = {"recent_block": "off"} if "recent" in options else {}
    smallpv_info = {"smallpv_block": "off"} if "smallpv" in options else {}
    pseudo_info = {"pseudo": "off"} if table.unlabelled_pool else {}
    tri_info = {"offset_tri": "off"} if "offset_tri" in options else {}
    strict_info = {"recur_strict": "off"} if "recur_strict" in options else {}
    onset_first_info = {"onset_first": "off"} if "onset_first" in options else {}
    tail_guard_info = {"tail_guard": "off"} if "tail_guard" in options else {}
    dkw: dict = {}  # adopted date options, passed to every later inner OOF and the final fit
    base_oof = None
    base_sel = None
    if splits:
        alt_cs = tuple(c for c in C_OPTIONS if c != LOGREG_C) if "ctune" in options else ()
        tri_kw = {"tri_columns": True} if "offset_tri" in options else {}
        oof = _inner_oof(
            table, labels, lab, train_ids, splits, names, cap_names, setting, alt_cs=alt_cs, weekly=weekly, **tri_kw
        )
        sel = _select(oof, avg)
        sel["inner_score"] = _full_inner_score(oof, sel)
        if alt_cs:
            C, oof, sel, c_info = _choose_logreg_c(oof, sel, avg)
            sel["inner_score"] = _full_inner_score(oof, sel)
        if "offset_tri" in options:
            oof, sel, tri_info = _choose_offset_tri(oof, sel, avg)
            sel["inner_score"] = _full_inner_score(oof, sel)
            if tri_info["offset_tri"] == "on":
                dkw["offset_tri"] = True
        if "recur_strict" in options:
            oof, sel, strict_info = _choose_recur_strict(
                table, labels, lab, train_ids, splits, names, cap_names, oof, sel, avg, C, weekly, dkw
            )
            sel["inner_score"] = _full_inner_score(oof, sel)
            if strict_info["recur_strict"] == "on":
                dkw["recur_strict"] = True
        if "onset_first" in options:
            oof, sel, onset_first_info = _choose_onset_first(
                table, labels, lab, train_ids, splits, names, cap_names, oof, sel, avg, C, weekly, dkw
            )
            sel["inner_score"] = _full_inner_score(oof, sel)
            if onset_first_info["onset_first"] == "on":
                dkw["onset_first"] = True
        if "tail_guard" in options:
            oof, sel, tail_guard_info = _choose_tail_guard(
                table, labels, lab, train_ids, splits, names, cap_names, oof, sel, avg, C, weekly, dkw
            )
            sel["inner_score"] = _full_inner_score(oof, sel)
            if tail_guard_info["tail_guard"] == "on":
                dkw["tail_guard"] = True
        if table.unlabelled is not None:
            setting, oof, sel, unl_info = _choose_unlabelled(
                table, labels, lab, train_ids, splits, names, cap_names, oof, sel, avg, C, weekly, dkw
            )
            sel["inner_score"] = _full_inner_score(oof, sel)
        if "e1" in options:
            names, oof, sel, feat_info = _choose_feature_block(
                table, labels, lab, train_ids, splits, export, cap_names, setting, oof, sel, avg, C, weekly, dkw
            )
            sel["inner_score"] = _full_inner_score(oof, sel)
        block_args = (table, labels, lab, train_ids, splits)
        if "schedule" in options:
            names, oof, sel, sched_info = _choose_block(
                "schedule", SCHEDULE_FEATURES, *block_args, names, cap_names, setting, oof, sel, avg, C, weekly, dkw
            )
            sel["inner_score"] = _full_inner_score(oof, sel)
        if "sun" in options:
            names, oof, sel, sun_info = _choose_block(
                "sun", SUN_FEATURES, *block_args, names, cap_names, setting, oof, sel, avg, C, weekly, dkw
            )
            sel["inner_score"] = _full_inner_score(oof, sel)
        for key, extra in (("recent", RECENT_FEATURES), ("smallpv", SMALLPV_FEATURES)):
            if key in options:
                names, oof, sel, info = _choose_block(
                    key, extra, *block_args, names, cap_names, setting, oof, sel, avg, C, weekly, dkw
                )
                sel["inner_score"] = _full_inner_score(oof, sel)
                if key == "recent":
                    recent_info = info
                else:
                    smallpv_info = info
        base_oof, base_sel = oof, sel
        option_oof, option_sel = oof, sel
    else:
        sel = {
            "classifier": "logreg",
            "threshold": 0.5,
            "threshold_initial": 0.5,
            "capacity": "ridge",
            "kappa": 1.0,
            "kappa_initial": 1.0,
            "insufficient_rule": "weak_candidate",
            "inner_score": float("nan"),
            "inner_score_initial": float("nan"),
        }
        if table.unlabelled is not None:
            unl_info = {"climatology": setting[0], "evidence_scale": setting[1]}
        option_oof, option_sel = None, sel
        if "onset_first" in options:  # the fixed dating rule needs no inner CV
            dkw["onset_first"] = True
            onset_first_info = {"onset_first": "on"}
    labeller = None
    if table.unlabelled_pool and splits:
        from pvdetect.pseudolabel import choose_pseudo  # pseudolabel imports this module

        sel, pseudo_info, selected_oof = choose_pseudo(
            table, labels, lab, train_ids, splits, names, cap_names, setting, base_oof, sel, avg, C, weekly, dkw
        )
        sel["inner_score"] = _full_inner_score(selected_oof, sel)
        option_oof, option_sel = selected_oof, sel
        labeller = pseudo_info.get("pseudo_labeller")
    elif not table.unlabelled_pool:
        labeller = None
    else:
        pseudo_info["pseudo_disabled_reason"] = "inner CV is disabled"
    # Output corrections on the selected inner OOF, after every option that refits on new inner OOFs.
    debias_info = {"date_debias": "off"} if "date_debias" in options else {}
    band_info = {"cap_ceiling": "off"} if "cap_ceiling" in options else {}
    date_bias = None
    if option_oof is not None:
        if "date_debias" in options:
            option_oof, option_sel, debias_info, date_bias = _choose_date_debias(option_oof, option_sel, avg)
        if "cap_ceiling" in options:
            option_oof, option_sel, band_info = _choose_cap_ceiling(option_oof, option_sel, avg)
        sel = option_sel
    comparison_sel = option_sel
    raw = None
    embedding_projection = None
    block = "baseline"
    unavailable_reason = None
    if encoder is not None and feature_block in {"auto", "embedding"}:
        from pvdetect.embedding import EMBEDDING_FEATURES, ProjectionUnavailable, fit_projection

        raw = encoder.raw_customer_features(table, table.ids)
        emb_names = names + list(EMBEDDING_FEATURES)
        try:
            if splits:
                emb_oof = _inner_oof(
                    table,
                    labels,
                    lab,
                    train_ids,
                    splits,
                    emb_names,
                    cap_names,
                    setting,
                    C=C,
                    weekly=weekly,
                    **dkw,
                    embedding=encoder,
                    raw=raw,
                )
                emb_sel = _select(emb_oof, avg)
                if date_bias is not None:  # the adopted corrections, cross-fitted on the candidate's OOF
                    emb_oof, emb_bias = _debias_oof(emb_oof, emb_sel)
                if band_info.get("cap_ceiling") == "on":
                    emb_oof = _band_oof(emb_oof)
                if date_bias is not None or band_info.get("cap_ceiling") == "on":
                    emb_sel = _select(emb_oof, avg)
                emb_sel["inner_score"] = _full_inner_score(emb_oof, emb_sel)
                emb_bootstrap = _paired_bootstrap_stability(option_oof, option_sel, emb_oof, emb_sel)
            else:
                emb_oof = None
                emb_sel = dict(sel)
                emb_bootstrap = float("nan")
            embedding_projection = fit_projection(raw, train_ids)
        except ProjectionUnavailable as exc:
            unavailable_reason = str(exc)
            if feature_block == "embedding":
                raise ValueError(f"embedding candidate unavailable: {exc}") from exc
            emb_sel = None
        if emb_sel is not None and (
            feature_block == "embedding"
            or (
                np.isfinite(comparison_sel.get("inner_score", np.nan))
                and emb_sel.get("inner_score", -np.inf) - comparison_sel.get("inner_score", -np.inf) >= CLASSIFIER_MARGIN
                and emb_bootstrap >= BOOTSTRAP_MIN
            )
        ):
            block = "embedding"
            sel = emb_sel
            names = emb_names
            if date_bias is not None:
                date_bias = emb_bias
        else:
            embedding_projection = None
    else:
        emb_bootstrap = float("nan")
    if block == "embedding" and labeller:
        labeller = None
        pseudo_info = {k: v for k, v in pseudo_info.items() if k != "pseudo_labeller"}
        pseudo_info.update(
            pseudo="off",
            pseudo_disabled_reason="embedding candidate selected; pool has no epoch frame for embeddings",
        )
    if block == "embedding":
        sel["feature_block"] = "embedding"
        sel["embedding_feature_block"] = feat_info.get("feature_block", "base")
        sel.update({k: v for k, v in feat_info.items() if k != "feature_block"})
    elif "e1" in options:
        sel.update(feat_info)
    sel.update(unl_info)
    sel.update(sched_info)
    sel.update(sun_info)
    sel.update(recent_info)
    sel.update(smallpv_info)
    sel.update(pseudo_info)
    clf_kinds = ("logreg", "catboost") if sel["classifier"] in ("avg", "catboost") or labeller else ("logreg", "logreg")
    comp = _fit_components(
        table,
        lab,
        train_ids,
        names,
        cap_names,
        clf_kinds,
        (sel["capacity"],),
        setting,
        C,
        embedding=encoder if block == "embedding" else None,
        projection=embedding_projection,
        **dkw,  # includes onset_first=True / tail_guard=True when adopted
    )
    if labeller:
        from pvdetect.pseudolabel import final_pseudo

        comp = final_pseudo(table, lab, train_ids, names, comp, pseudo_info)
    if sel["classifier"] in ("catboost", "avg") and "catboost_clf" not in comp.objects:
        sel["classifier"] = "logreg"
        sel["inner_score"] = sel.get("inner_score_logreg", sel.get("inner_score", np.nan))
    if sel["capacity"] == "catboost" and "catboost_cap" not in comp.objects:
        sel["capacity"] = "ridge"
    date = {**comp.date_params, "insufficient_rule": sel["insufficient_rule"]}
    if "recur_strict" in options:  # inference detects with the same rule as this fit
        date["recur_strict"] = comp.recur_strict
    if "onset_first" in options:
        date["onset_first"] = comp.onset_first
    if "tail_guard" in options:
        date["tail_guard"] = comp.tail_guard
    sel.update(offset_event=date["offset_event"], start_lag=date["start_lag"], contrast_cut=date["contrast_cut"])
    if date_bias is not None:  # inference shifts each regime's date by the fit's inner OOF bias
        date["date_bias"] = date_bias
    sel.update(c_info)
    sel.update(tri_info)
    sel.update(strict_info)
    sel.update(onset_first_info)
    sel.update(tail_guard_info)
    sel.update(debias_info)
    sel.update(band_info)
    sel["embedding_bootstrap_p"] = emb_bootstrap
    if block == "embedding":
        sel["feature_block"] = "embedding"
    if unavailable_reason is not None:
        sel["embedding_unavailable_reason"] = unavailable_reason
    uses_cb = sel["classifier"] in ("catboost", "avg")
    objects = {}
    if uses_cb:
        objects["catboost_clf"] = comp.objects["catboost_clf"]
    if sel["capacity"] == "catboost":
        objects["catboost_cap"] = comp.objects["catboost_cap"]
    # the weekly head is fitted on the labelled customers only (pseudo-labels refit the logreg/CatBoost only)
    head = {"weekly": fit_head(table, lab, train_ids, comp.clim)} if sel["classifier"] in WEEKLY_KINDS else {}
    model = {
        "version": CURRENT_VERSION,
        "feature_names": names,
        "capacity_feature_names": cap_names,
        "export_enabled": export,
        "climatology": comp.clim.to_json(),
        "classifier": {
            "kind": sel["classifier"],
            "threshold": sel["threshold"],
            "logreg": comp.logreg if sel["classifier"] in ("logreg", "avg", "avg_weekly") else None,
            "catboost_file": "catboost_clf.cbm" if uses_cb else None,
            **head,
        },
        "capacity": {
            "kind": sel["capacity"],
            "ridge": comp.ridge,
            "ridge_export": comp.ridge_export,
            "catboost_file": "catboost_cap.cbm" if sel["capacity"] == "catboost" else None,
            "kappa": sel["kappa"],
            "fallback_kwp": comp.fallback,
            **({"band": comp.cap_band} if band_info.get("cap_ceiling") == "on" and comp.cap_band else {}),
        },
        "date": date,
        "selection": sel,
        "extensions": {"enabled": False},
        "_objects": objects,
    }
    if block == "embedding":
        model["embedding"] = {
            "feature_names": list(EMBEDDING_FEATURES),
            "projection": embedding_projection,
            "encoder_files": {"weights": "encoder/encoder.pt", "metadata": "encoder/encoder.json"},
            "encoder_hashes": encoder.hashes,
            "encoder_config": encoder.config,
            "window_config": encoder.window_config,
        }
        model["_encoder_source"] = str(encoder.directory)
        model["_embedding_encoder"] = encoder
    return model
