"""Training procedure: every data-driven choice happens inside `fit`, chosen by inner CV (Tier A)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pvdetect.changepoint import shrunk_median
from pvdetect.evaluation import inner_splits
from pvdetect.features import EXPORT_CAPACITY_FEATURES, capacity_feature_names, feature_names
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
from pvdetect.season import Climatology, fit_climatology
from pvdetect.selection import (
    ape_array,
    bootstrap_prob_better,
    choose_kappa,
    pick_threshold,
    threshold_scores,
)

CLASSIFIER_MARGIN = 2.0
BOOTSTRAP_MIN = 0.8
CAPACITY_MARGIN = 0.03
NO_CUT = -1e9  # contrast_cut when the present_at_start regime is disabled
# Labelled commissioning dates may lag the visible onset by months.
OFFSET_ERROR_WINDOW = 180
OFFSET_CAP = 120

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


def _fit_components(
    table: FeatureTable,
    lab: pd.DataFrame,
    ids: list[str],
    names: list[str],
    cap_names: list[str],
    clf_kinds: tuple[str, ...],
    cap_kinds: tuple[str, ...],
    *,
    embedding=None,
    projection: dict | None = None,
) -> Components:
    from catboost import CatBoostClassifier, CatBoostRegressor

    sub = lab.loc[ids]
    y = sub["pv_installed"].to_numpy(bool)
    dates = {c: d.date() for c, d in sub.loc[y, "pv_installation_date"].items() if pd.notna(d)}
    nonpv = [c for c, on in zip(ids, y) if not on]
    clim = fit_climatology(table.dailies(), nonpv, dates)
    F, cps = compute_features(table, ids, clim, embedding=embedding, projection=projection)
    date_params = fit_date_params(table, lab, ids, F, cps)
    X = F[names].to_numpy(float)
    both = y.any() and not y.all()
    objects: dict = {}
    logreg = None
    if "logreg" in clf_kinds:
        logreg = fit_logreg(X, y, C=0.5) if both else _constant_linear(X, 50.0 if y.all() else -50.0)
    if "catboost" in clf_kinds and both:
        cb = CatBoostClassifier(auto_class_weights="Balanced", **CB_PARAMS)
        cb.fit(X, y.astype(int))
        objects["catboost_clf"] = cb
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


def _inner_oof(
    table,
    labels,
    lab,
    train_ids,
    splits,
    names,
    cap_names,
    *,
    embedding=None,
    raw=None,
) -> pd.DataFrame:
    from pvdetect.embedding import fit_projection

    rows = []
    for tr, va in splits:
        projection = fit_projection(raw, tr) if embedding is not None else None
        comp = _fit_components(
            table,
            lab,
            tr,
            names,
            cap_names,
            ("logreg", "catboost"),
            ("ridge", "catboost"),
            embedding=embedding,
            projection=projection,
        )
        Fva, cps_va = compute_features(table, va, comp.clim, embedding=embedding, projection=projection)
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


def _select(oof: pd.DataFrame) -> dict:
    """Tier A choices from pooled inner OOF predictions."""
    t_pv = oof["t_pv"].to_numpy(bool)
    t_dates = oof["t_date"].to_numpy("datetime64[ns]")
    err = _err_days(oof["date_weak"], t_dates)
    ape_r = ape_array(oof["kwp_ridge"].to_numpy(float), oof["t_kwp"].to_numpy(float))
    p_lr, p_cb = oof["p_logreg"].to_numpy(float), oof["p_catboost"].to_numpy(float)
    t_lr, s_lr = pick_threshold(threshold_scores(p_lr, t_pv, err, ape_r))
    t_cb, s_cb = pick_threshold(threshold_scores(p_cb, t_pv, err, ape_r))
    clf_kind, thr, boot = "logreg", t_lr, None
    if s_cb - s_lr >= CLASSIFIER_MARGIN:
        boot = bootstrap_prob_better(p_lr, t_lr, p_cb, t_cb, t_pv, err, ape_r)
        if boot >= BOOTSTRAP_MIN:
            clf_kind, thr = "catboost", t_cb
    prob = p_cb if clf_kind == "catboost" else p_lr
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
    final_kwp = np.clip(kappa * oof[f"kwp_{cap_kind}"].to_numpy(float), 0.5, 500.0)
    ape_final = ape_array(final_kwp, true_k)
    thr_final, s_final = pick_threshold(threshold_scores(prob, t_pv, err_final, ape_final))
    return {
        "classifier": clf_kind,
        "threshold": thr_final,
        "threshold_initial": thr,
        "inner_score_logreg": s_lr,
        "inner_score_catboost": s_cb,
        "inner_score": s_final,
        "inner_score_initial": s_cb if clf_kind == "catboost" else s_lr,
        "bootstrap_p": boot,
        "capacity": cap_kind,
        "inner_P_ridge": P_r,
        "inner_P_catboost": P_c,
        "kappa": kappa,
        "insufficient_rule": rule,
    }
def _full_inner_predictions(oof: pd.DataFrame, sel: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    prob = oof["p_catboost"] if sel["classifier"] == "catboost" else oof["p_logreg"]
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
            "pv_installed": prob.to_numpy(float) >= float(sel["threshold"]),
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


def _paired_bootstrap_stability(
    base_oof: pd.DataFrame, base_sel: dict, emb_oof: pd.DataFrame, emb_sel: dict, B: int = 1000
) -> float:
    from pvdetect.scoring import main_score

    bp, truth = _full_inner_predictions(base_oof, base_sel)
    ep, _ = _full_inner_predictions(emb_oof, emb_sel)
    rng = np.random.default_rng(0)
    better = 0
    for _ in range(B):
        idx = rng.integers(0, len(truth), len(truth))
        sb = main_score(bp.iloc[idx], truth.iloc[idx])["score"]
        se = main_score(ep.iloc[idx], truth.iloc[idx])["score"]
        better += se > sb
    return better / B


def fit(
    table: FeatureTable,
    labels: pd.DataFrame,
    train_ids: list[str],
    *,
    inner: bool = True,
    encoder=None,
    feature_block: str = "auto",
) -> dict:
    """Fit baseline or the opt-in frozen-embedding candidate.

    ``encoder`` may be a :class:`FrozenEncoder` or an artifact directory.  A
    supplied bad artifact is intentionally allowed to raise; it never falls
    back to baseline.
    """
    if feature_block not in {"auto", "baseline", "embedding"}:
        raise ValueError("feature_block must be auto, baseline, or embedding")
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
    raw = None
    embedding_projection = None
    block = "baseline"
    unavailable_reason = None
    base_oof = None
    emb_oof = None
    base_score = float("nan")
    emb_score = float("nan")
    emb_bootstrap = float("nan")
    if splits:
        base_oof = _inner_oof(table, labels, lab, train_ids, splits, base_names, cap_names)
        base_sel = _select(base_oof)
        base_score = _full_inner_score(base_oof, base_sel)
        base_sel["inner_score"] = base_score
    else:
        base_sel = {
            "classifier": "logreg",
            "threshold": 0.5,
            "threshold_initial": 0.5,
            "capacity": "ridge",
            "kappa": 1.0,
            "insufficient_rule": "weak_candidate",
            "inner_score": float("nan"),
            "inner_score_initial": float("nan"),
        }
    emb_sel = None
    if encoder is not None and feature_block in {"auto", "embedding"}:
        from pvdetect.embedding import EMBEDDING_FEATURES, ProjectionUnavailable, fit_projection

        raw = encoder.raw_customer_features(table, table.ids)
        emb_names = base_names + list(EMBEDDING_FEATURES)
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
                    embedding=encoder,
                    raw=raw,
                )
                emb_sel = _select(emb_oof)
                emb_score = _full_inner_score(emb_oof, emb_sel)
                emb_bootstrap = _paired_bootstrap_stability(base_oof, base_sel, emb_oof, emb_sel)
            if not splits:
                emb_sel = dict(base_sel)
            # This is the outer fit projection, never a globally cached artifact.
            embedding_projection = fit_projection(raw, train_ids)
        except ProjectionUnavailable as exc:
            unavailable_reason = str(exc)
            if feature_block == "embedding":
                raise ValueError(f"embedding candidate unavailable: {exc}") from exc
            emb_sel = None
        if emb_sel is not None and (
            feature_block == "embedding"
            or (
                np.isfinite(base_sel.get("inner_score", np.nan))
                and emb_sel.get("inner_score", -np.inf) - base_sel.get("inner_score", -np.inf) >= CLASSIFIER_MARGIN
                and emb_bootstrap >= BOOTSTRAP_MIN
            )
        ):
            block = "embedding"
            sel = emb_sel
            names = emb_names
        else:
            sel = base_sel
            names = base_names
    else:
        sel = base_sel
        names = base_names
    if block != "embedding":
        embedding_projection = None
    comp = _fit_components(
        table,
        lab,
        train_ids,
        names,
        cap_names,
        ("logreg", sel["classifier"]),
        (sel["capacity"],),
        embedding=encoder if block == "embedding" else None,
        projection=embedding_projection,
    )
    if sel["classifier"] == "catboost" and "catboost_clf" not in comp.objects:
        sel["classifier"] = "logreg"
        sel["inner_score"] = sel.get("inner_score_logreg", sel.get("inner_score", np.nan))
    if sel["capacity"] == "catboost" and "catboost_cap" not in comp.objects:
        sel["capacity"] = "ridge"
    date = {**comp.date_params, "insufficient_rule": sel["insufficient_rule"]}
    sel.update(offset_event=date["offset_event"], start_lag=date["start_lag"], contrast_cut=date["contrast_cut"])
    sel["embedding_bootstrap_p"] = emb_bootstrap
    sel["feature_block"] = block
    if unavailable_reason is not None:
        sel["embedding_unavailable_reason"] = unavailable_reason
    objects = {}
    if sel["classifier"] == "catboost":
        objects["catboost_clf"] = comp.objects["catboost_clf"]
    if sel["capacity"] == "catboost":
        objects["catboost_cap"] = comp.objects["catboost_cap"]
    model = {
        "version": CURRENT_VERSION,
        "feature_names": names,
        "capacity_feature_names": cap_names,
        "export_enabled": export,
        "climatology": comp.clim.to_json(),
        "classifier": {
            "kind": sel["classifier"],
            "threshold": sel["threshold"],
            "logreg": comp.logreg if sel["classifier"] == "logreg" else None,
            "catboost_file": "catboost_clf.cbm" if sel["classifier"] == "catboost" else None,
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
    return model
