"""Label-free feature table shared by training and inference, and model prediction."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pvdetect.changepoint import Commissioning, assign_regime
from pvdetect.daily import daily_signal
from pvdetect.features import (
    EXPORT_CAPACITY_FEATURES,
    LOG1P_CAPACITY,
    fitted_features,
    label_free_features,
    sun_coupling,
)
from pvdetect.io import LoadedTimeseries, day_matrix
from pvdetect.model import linear_score, sigmoid
from pvdetect.season import Climatology, sun_residual
from pvdetect.weekly import customer_scores

if TYPE_CHECKING:
    from pvdetect.unlabelled import UnlabelledPrior

EXT_COLUMNS = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]
SIGNED_SHARE = 0.01


@dataclass(frozen=True)
class CustomerData:
    daily: pd.DataFrame
    dates: np.ndarray
    M: np.ndarray
    neg_frac: float
    q001: float
    first_date: dt.date
    n_valid: int


@dataclass(frozen=True)
class FeatureTable:
    """Label-free per-customer data. `cache` holds label-free derived values only (§2 cache rule);
    `unlabelled` is the optional label-free, fold-independent prior from the unlabelled series;
    `unlabelled_pool` is the optional label-free pool of unlabelled customers for pseudo-labelling."""

    ids: list[str]
    customers: dict[str, CustomerData | None]
    cache: dict = field(default_factory=dict)
    # Retain the parsed epoch rows for optional frozen-window inference.  This
    # is the same frame produced by io.load_timeseries; no labels are present.
    frame: pd.DataFrame | None = None
    unlabelled: UnlabelledPrior | None = None
    unlabelled_pool: dict[str, CustomerData] | None = None

    def dailies(self) -> dict[str, pd.DataFrame]:
        return {c: d.daily for c, d in self.customers.items() if d is not None}


def _embedding_features(table: FeatureTable, ids: list[str], embedding, projection: dict) -> pd.DataFrame:
    from pvdetect.embedding import transform_projection

    key = (
        embedding.hashes.get("encoder.pt"),
        embedding.hashes.get("encoder.json"),
        tuple(sorted(embedding.window_config.items())),
        id(table.frame),
        len(table.frame) if table.frame is not None else 0,
    )
    raw_cache = table.cache.setdefault("raw_embeddings", {})
    raw = raw_cache.get(key)
    if raw is None:
        raw = embedding.raw_customer_features(table, table.ids)
        raw_cache[key] = raw
    return transform_projection(raw, ids, projection)


def customer_data(dates: np.ndarray, M: np.ndarray, energy: np.ndarray) -> CustomerData:
    """One customer's data from its UTC+1 day matrix and its (deduplicated) readings."""
    daily = daily_signal(dates, M)
    valid = daily.index[daily["valid"].to_numpy()]
    first = (valid[0] if len(valid) else daily.index[0]).date()
    return CustomerData(
        daily=daily,
        dates=dates,
        M=M,
        neg_frac=float(np.mean(energy < 0)),
        q001=float(np.quantile(energy, 0.001)),
        first_date=first,
        n_valid=int(daily["valid"].sum()),
    )


def build_feature_table(loaded: LoadedTimeseries) -> FeatureTable:
    customers: dict[str, CustomerData | None] = dict.fromkeys(loaded.customer_ids)
    for cid, g in loaded.frame.groupby("customer_id", sort=False):
        customers[str(cid)] = customer_data(*day_matrix(g), g["energy_kwh"].to_numpy())
    return FeatureTable(ids=list(loaded.customer_ids), customers=customers, frame=loaded.frame.copy())


def sun_anomaly(table: FeatureTable, cid: str) -> pd.Series:
    """Cached label-free `sun_residual` of one customer (§2 cache rule)."""
    cache = table.cache.setdefault("sun_residual", {})
    if cid not in cache:
        cd = table.customers.get(cid)
        cache[cid] = sun_residual(cd.daily) if cd is not None and cd.n_valid else pd.Series(dtype=float)
    return cache[cid]


def compute_features(
    table: FeatureTable,
    ids: list[str],
    clim: Climatology,
    *,
    recur_strict: bool = False,
    embedding=None,
    projection: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Features for ``ids`` under ``clim`` (and the fit's ``recur_strict`` detection option); only the
    label-free part is cached on the table. With a sunshine index, a contributor's ``sun_coupling`` uses the
    index without itself. The embedding block is opt-in and persisted."""
    lf_cache = table.cache.setdefault("label_free", {})
    rows, cps = {}, {}
    for cid in ids:
        cd = table.customers.get(cid)
        if cid not in lf_cache:
            lf_cache[cid] = label_free_features(cd)
        lf = lf_cache[cid]
        fitted, cp = fitted_features(cd, clim, lf, recur_strict=recur_strict)
        if clim.sun is not None:
            fitted["sun_coupling"] = sun_coupling(sun_anomaly(table, cid), clim.sun.series(cid))
        rows[cid] = {**lf, **fitted}
        cps[cid] = cp
    F = pd.DataFrame.from_dict(rows, orient="index").reindex(ids)
    if embedding is not None or projection is not None:
        if embedding is None or projection is None:
            raise ValueError("embedding and projection must be supplied together")
        F = pd.concat([F, _embedding_features(table, ids, embedding, projection)], axis=1)
    return F, cps


def capacity_matrix(F: pd.DataFrame, names: list[str]) -> np.ndarray:
    X = F[names].to_numpy(float).copy()
    for j, n in enumerate(names):
        if n in LOG1P_CAPACITY:
            X[:, j] = np.log1p(np.maximum(X[:, j], 0))
    return X


def classifier_proba(
    clf: dict,
    objects: dict,
    F: pd.DataFrame,
    names: list[str],
    kind: str | None = None,
    weekly: np.ndarray | None = None,
) -> np.ndarray:
    """P(PV); `avg` (E2) is the mean of the logreg and CatBoost probabilities; `weekly` kinds need the
    weekly head's customer scores (`weekly`): `weekly` is that score, `avg_weekly` its mean with logreg."""
    kind = kind or clf["kind"]
    if kind == "weekly":
        return weekly
    X = F[names].to_numpy(float)
    if kind == "logreg":
        return sigmoid(linear_score(clf["logreg"], X))
    if kind == "avg":
        return (sigmoid(linear_score(clf["logreg"], X)) + objects["catboost_clf"].predict_proba(X)[:, 1]) / 2
    if kind == "avg_weekly":
        return (sigmoid(linear_score(clf["logreg"], X)) + weekly) / 2
    return objects["catboost_clf"].predict_proba(X)[:, 1]


def capacity_log(cap: dict, objects: dict, F: pd.DataFrame, names: list[str], kind: str | None = None) -> np.ndarray:
    """log(kWp) before κ; ridge uses the export model for signed customers when available."""
    kind = kind or cap["kind"]
    if kind == "catboost":
        cb_names = names + (["neg_frac"] if "export_kw" in names else [])
        return objects["catboost_cap"].predict(F[cb_names].to_numpy(float))
    out = linear_score(cap["ridge"], capacity_matrix(F, names))
    if cap.get("ridge_export") is not None:
        signed = F["neg_frac"].to_numpy(float) >= SIGNED_SHARE
        if signed.any():
            ex = linear_score(cap["ridge_export"], capacity_matrix(F, list(EXPORT_CAPACITY_FEATURES)))
            out = np.where(signed, ex, out)
    return out


def apply_cap_band(kwp: np.ndarray, band: dict | None) -> np.ndarray:
    """`cap_ceiling`: capacities (before κ) outside the fit's learned [lo, hi] band become its fallback."""
    if not band:
        return kwp
    return np.where((kwp < band["lo"]) | (kwp > band["hi"]), band["fallback"], kwp)


def to_kwp(log_kwp: np.ndarray, kappa: float, fallback: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        k = np.clip(kappa * np.exp(log_kwp), 0.5, 500)
    return np.where(np.isfinite(k), k, fallback)


def regime_date(
    cp: Commissioning | None, regime: str, cd: CustomerData | None, date_params: dict, rule: str
) -> dt.date | None:
    if cd is None:
        return None
    shift = dt.timedelta(days=(date_params.get("date_bias") or {}).get(regime, 0))  # `date_debias`
    if regime == "event":
        return cp.date + dt.timedelta(days=date_params["offset_event"]) - shift
    if regime == "present_at_start":
        return cd.first_date - dt.timedelta(days=date_params["start_lag"]) - shift
    if rule == "weak_candidate" and cp is not None and cp.weak_date is not None:
        return cp.weak_date - shift
    return cd.first_date - shift


def regimes_and_dates(table: FeatureTable, ids: list[str], F: pd.DataFrame, cps: dict, date_params: dict, rule: str):
    regimes, dates = [], []
    for cid in ids:
        r = assign_regime(cps[cid], F.at[cid, "early_contrast"], date_params["contrast_cut"])
        regimes.append(r)
        dates.append(regime_date(cps[cid], r, table.customers.get(cid), date_params, rule))
    return regimes, dates


def predict(model: dict, table: FeatureTable, ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictions for `ids`. TRUE rows always carry a date and a positive capacity."""
    clim = Climatology.from_json(model["climatology"])
    objects = model.get("_objects", {})
    embedding = model.get("_embedding_encoder")
    projection = (model.get("embedding") or {}).get("projection") if embedding is not None else None
    F, cps = compute_features(
        table,
        ids,
        clim,
        recur_strict=model["date"].get("recur_strict", False),
        embedding=embedding,
        projection=projection,
    )
    clf = model["classifier"]
    weekly = customer_scores(clf["weekly"], table, ids, clim) if "weekly" in clf else None
    prob = classifier_proba(clf, objects, F, model["feature_names"], weekly=weekly)
    cap = model["capacity"]
    log_kwp = capacity_log(cap, objects, F, model["capacity_feature_names"])
    if cap.get("band"):  # as scored in inner CV: band on the clipped κ=1 capacity, then κ and the clip
        log_kwp = np.log(apply_cap_band(to_kwp(log_kwp, 1.0, cap["fallback_kwp"]), cap["band"]))
    kwp = to_kwp(log_kwp, cap["kappa"], cap["fallback_kwp"])
    regimes, dates = regimes_and_dates(table, ids, F, cps, model["date"], model["date"]["insufficient_rule"])
    rows = []
    for i, cid in enumerate(ids):
        cd = table.customers.get(cid)
        on = bool(prob[i] >= clf["threshold"]) and cd is not None and cd.n_valid > 0
        cp = cps[cid]
        rows.append(
            {
                "customer_id": cid,
                "pv_installed": on,
                "pv_installation_date": pd.Timestamp(dates[i]) if on else pd.NaT,
                "pv_capacity_kwp": float(kwp[i]) if on else np.nan,
                "prob": float(prob[i]),
                "regime": regimes[i],
                "cp_found": bool(cp is not None and cp.found),
                "first_date": pd.Timestamp(cd.first_date) if cd is not None else pd.NaT,
                "raw_date": pd.Timestamp(dates[i]) if dates[i] is not None else pd.NaT,
                "raw_kwp": float(kwp[i]),
            }
        )
    pred = pd.concat([pd.DataFrame(rows), F.reset_index(drop=True)], axis=1)
    return pred, pd.DataFrame(columns=EXT_COLUMNS)
