"""Challenge metrics and output validation."""
from __future__ import annotations

from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pv"}


def _valid_date(value: object) -> date | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError, OverflowError):
        return None


def _valid_positive(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) and x > 0 else None


def score_predictions(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float | int]:
    """Compute the exact conditional challenge score.

    Date and capacity rewards are averaged over true-positive customer detections.
    Invalid or absent conditional targets contribute zero, and no true-positive
    detection makes either conditional reward zero.
    """
    for column in ("customer_id", "pv_installed"):
        if column not in labels.columns or column not in predictions.columns:
            raise ValueError(f"labels and predictions require {column}")
    left = labels.copy(); right = predictions.copy()
    left["customer_id"] = left.customer_id.astype(str); right["customer_id"] = right.customer_id.astype(str)
    if right.customer_id.duplicated().any():
        raise ValueError("predictions must contain one row per customer")
    frame = left.merge(right, on="customer_id", how="left", suffixes=("_true", "_pred"), validate="one_to_one")
    true = frame.pv_installed_true.map(_as_bool).to_numpy()
    pred = frame.pv_installed_pred.fillna(False).map(_as_bool).to_numpy()
    tp = int(np.sum(true & pred)); fp = int(np.sum(~true & pred)); fn = int(np.sum(true & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    date_rewards: list[float] = []; capacity_rewards: list[float] = []
    for i in range(len(frame)):
        if not (true[i] and pred[i]):
            continue
        actual_date = _valid_date(frame.iloc[i].get("pv_installation_date"))
        predicted_date = _valid_date(frame.iloc[i].get("predicted_installation_date", frame.iloc[i].get("pv_installation_date_pred")))
        if actual_date is None or predicted_date is None:
            date_rewards.append(0.0)
        else:
            date_rewards.append(max(0.0, 1.0 - abs((predicted_date - actual_date).days) / 30.0))
        actual_capacity = _valid_positive(frame.iloc[i].get("pv_capacity_kwp"))
        predicted_capacity = _valid_positive(frame.iloc[i].get("predicted_capacity_kwp", frame.iloc[i].get("pv_capacity_kwp_pred")))
        if actual_capacity is None or predicted_capacity is None:
            capacity_rewards.append(0.0)
        else:
            capacity_rewards.append(max(0.0, 1.0 - abs(predicted_capacity - actual_capacity) / actual_capacity / 0.5))
    d = float(np.mean(date_rewards)) if date_rewards else 0.0
    p = float(np.mean(capacity_rewards)) if capacity_rewards else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "D": d, "P": p, "main_score": float(f1 * (70.0 + 15.0 * d + 15.0 * p)),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn}


def validate_predictions(predictions: pd.DataFrame, customer_ids: list[str]) -> pd.DataFrame:
    """Normalize output columns and require complete, unique customer coverage."""
    required = {"customer_id", "pv_installed"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")
    frame = predictions.copy(); frame.customer_id = frame.customer_id.astype(str)
    expected = [str(x) for x in customer_ids]
    if set(frame.customer_id) != set(expected) or frame.customer_id.duplicated().any():
        raise ValueError("predictions must contain exactly one row for every evaluation customer")
    frame["pv_installed"] = frame.pv_installed.map(_as_bool).astype(bool)
    if "predicted_installation_date" not in frame:
        frame["predicted_installation_date"] = pd.NaT
    if "predicted_capacity_kwp" not in frame:
        frame["predicted_capacity_kwp"] = np.nan
    frame.loc[~frame.pv_installed, ["predicted_installation_date", "predicted_capacity_kwp"]] = [pd.NaT, np.nan]
    return frame[["customer_id", "pv_installed", "predicted_installation_date", "predicted_capacity_kwp"]]
