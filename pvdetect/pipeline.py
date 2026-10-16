"""Label-free feature table shared by training and inference, and model prediction."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pvdetect.changepoint import Commissioning, assign_regime
from pvdetect.daily import daily_signal
from pvdetect.features import (
    EXPORT_CAPACITY_FEATURES,
    LOG1P_CAPACITY,
    fitted_features,
    label_free_features,
)
from pvdetect.io import LoadedTimeseries, day_matrix
from pvdetect.model import linear_score, sigmoid
from pvdetect.season import Climatology

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
    """Label-free per-customer data. `cache` holds label-free derived values only (§2 cache rule)."""

    ids: list[str]
    customers: dict[str, CustomerData | None]
    cache: dict = field(default_factory=dict)

    def dailies(self) -> dict[str, pd.DataFrame]:
        return {c: d.daily for c, d in self.customers.items() if d is not None}


def build_feature_table(loaded: LoadedTimeseries) -> FeatureTable:
    customers: dict[str, CustomerData | None] = dict.fromkeys(loaded.customer_ids)
    for cid, g in loaded.frame.groupby("customer_id", sort=False):
        dates, M = day_matrix(g)
        daily = daily_signal(dates, M)
        e = g["energy_kwh"].to_numpy()
        valid = daily.index[daily["valid"].to_numpy()]
        first = (valid[0] if len(valid) else daily.index[0]).date()
        customers[str(cid)] = CustomerData(
            daily=daily,
            dates=dates,
            M=M,
            neg_frac=float(np.mean(e < 0)),
            q001=float(np.quantile(e, 0.001)),
            first_date=first,
            n_valid=int(daily["valid"].sum()),
        )
    return FeatureTable(ids=list(loaded.customer_ids), customers=customers)


def compute_features(table: FeatureTable, ids: list[str], clim: Climatology) -> tuple[pd.DataFrame, dict]:
    """All features for `ids` under `clim`; only the label-free part is cached on the table."""
    lf_cache = table.cache.setdefault("label_free", {})
    rows, cps = {}, {}
    for cid in ids:
        cd = table.customers.get(cid)
        if cid not in lf_cache:
            lf_cache[cid] = label_free_features(cd)
        lf = lf_cache[cid]
        fitted, cp = fitted_features(cd, clim, lf)
        rows[cid] = {**lf, **fitted}
        cps[cid] = cp
    return pd.DataFrame.from_dict(rows, orient="index").reindex(ids), cps


def capacity_matrix(F: pd.DataFrame, names: list[str]) -> np.ndarray:
    X = F[names].to_numpy(float).copy()
    for j, n in enumerate(names):
        if n in LOG1P_CAPACITY:
            X[:, j] = np.log1p(np.maximum(X[:, j], 0))
    return X


def classifier_proba(clf: dict, objects: dict, F: pd.DataFrame, names: list[str], kind: str | None = None) -> np.ndarray:
    kind = kind or clf["kind"]
    X = F[names].to_numpy(float)
    if kind == "logreg":
        return sigmoid(linear_score(clf["logreg"], X))
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


def to_kwp(log_kwp: np.ndarray, kappa: float, fallback: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        k = np.clip(kappa * np.exp(log_kwp), 0.5, 500)
    return np.where(np.isfinite(k), k, fallback)


def regime_date(
    cp: Commissioning | None, regime: str, cd: CustomerData | None, date_params: dict, rule: str
) -> dt.date | None:
    if cd is None:
        return None
    if regime == "event":
        return cp.date + dt.timedelta(days=date_params["offset_event"])
    if regime == "present_at_start":
        return cd.first_date - dt.timedelta(days=date_params["start_lag"])
    if rule == "weak_candidate" and cp is not None and cp.weak_date is not None:
        return cp.weak_date
    return cd.first_date


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
    F, cps = compute_features(table, ids, clim)
    clf = model["classifier"]
    prob = classifier_proba(clf, objects, F, model["feature_names"])
    cap = model["capacity"]
    kwp = to_kwp(capacity_log(cap, objects, F, model["capacity_feature_names"]), cap["kappa"], cap["fallback_kwp"])
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
