"""Label-free feature table shared by training and inference, and model prediction."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pvdetect.daily import daily_signal
from pvdetect.features import CAPACITY_FEATURES, FEATURE_NAMES, LOG1P_CAPACITY, baseline_features
from pvdetect.io import LoadedTimeseries, day_matrix
from pvdetect.model import linear_score, sigmoid


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


def baseline_feature_frame(table: FeatureTable, ids: list[str]) -> tuple[pd.DataFrame, dict]:
    cache = table.cache.setdefault("baseline", {})
    rows, cps = {}, {}
    for cid in ids:
        if cid not in cache:
            cd = table.customers.get(cid)
            cache[cid] = baseline_features(cd.daily) if cd is not None else (dict.fromkeys(FEATURE_NAMES, np.nan), None)
        rows[cid], cps[cid] = cache[cid]
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=ids, columns=list(FEATURE_NAMES)), cps


def capacity_matrix(F: pd.DataFrame, names: list[str]) -> np.ndarray:
    X = F[names].to_numpy(float).copy()
    for j, n in enumerate(names):
        if n in LOG1P_CAPACITY:
            X[:, j] = np.log1p(np.maximum(X[:, j], 0))
    return X


def predict(model: dict, table: FeatureTable, ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictions for `ids`. TRUE rows always carry a date and a positive capacity."""
    F, cps = baseline_feature_frame(table, ids)
    clf = model["classifier"]
    prob = sigmoid(linear_score(clf["logreg"], F[model["feature_names"]].to_numpy(float)))
    cap = model["capacity"]
    log_kwp = linear_score(cap["ridge"], capacity_matrix(F, model["capacity_feature_names"]))
    kwp = np.clip(cap["kappa"] * np.exp(log_kwp), 0.5, 500)
    kwp = np.where(np.isfinite(kwp), kwp, cap["fallback_kwp"])
    rows = []
    for i, cid in enumerate(ids):
        cd = table.customers.get(cid)
        on = bool(prob[i] >= clf["threshold"]) and cd is not None and cd.n_valid > 0
        cp = cps[cid]
        if cp is not None and cp.found:
            regime, date = "event", cp.date + dt.timedelta(days=model["date"]["offset_event"])
        else:
            regime, date = "insufficient", cd.first_date if cd is not None else None
        rows.append(
            {
                "customer_id": cid,
                "pv_installed": on,
                "pv_installation_date": pd.Timestamp(date) if on else pd.NaT,
                "pv_capacity_kwp": float(kwp[i]) if on else np.nan,
                "prob": float(prob[i]),
                "regime": regime,
                "cp_found": bool(cp is not None and cp.found),
                "first_date": pd.Timestamp(cd.first_date) if cd is not None else pd.NaT,
                "raw_date": pd.Timestamp(date) if date is not None else pd.NaT,
                "raw_kwp": float(kwp[i]),
            }
        )
    pred = pd.DataFrame(rows)
    pred = pd.concat([pred, F.reset_index(drop=True)], axis=1)
    ext = pd.DataFrame(columns=["customer_id", "extension_date", "pv_capacity_after_extension_kwp"])
    return pred, ext
