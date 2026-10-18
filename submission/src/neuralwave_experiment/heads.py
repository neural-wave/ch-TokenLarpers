"""Small downstream heads with explicit train-only fitting semantics."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .features import DateCandidate, candidate_change_point, date_candidates


def _safe_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = np.nan_to_num(x.mean(axis=0), nan=0.0) if len(x) else np.zeros(x.shape[1])
    scale = np.nan_to_num(x.std(axis=0), nan=1.0) if len(x) else np.ones(x.shape[1])
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return np.nan_to_num((x - mean) / scale), mean, scale


@dataclass
class PresenceHead:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    constant: float | None = None

    @classmethod
    def fit(cls, x: np.ndarray, y: Sequence[bool], c: float = 1.0) -> "PresenceHead":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=bool)
        if x.ndim != 2 or len(x) != len(y) or not len(x):
            raise ValueError("presence training requires a non-empty feature matrix")
        z, mean, scale = _safe_scale(x)
        classes = np.unique(y)
        if len(classes) < 2:
            return cls(mean, scale, np.zeros(x.shape[1]), 0.0, float(classes[0]))
        model = LogisticRegression(C=float(c), max_iter=1000, random_state=0, solver="liblinear")
        model.fit(z, y.astype(int))
        return cls(mean, scale, model.coef_[0].astype(float), float(model.intercept_[0]))

    def probability(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        if self.constant is not None:
            return np.full(len(z), float(self.constant))
        logits = np.clip(z @ self.coef + self.intercept, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, x: np.ndarray, threshold: float = .5) -> np.ndarray:
        return self.probability(x) >= float(threshold)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


@dataclass
class DateHead:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    fallback: bool = False

    @classmethod
    def fit(cls, candidates: Sequence[DateCandidate], labels: Mapping[str, date | None], tolerance_days: int = 3) -> "DateHead":
        xs: list[np.ndarray] = []
        ys: list[int] = []
        by_customer: dict[str, list[DateCandidate]] = {}
        for candidate in candidates:
            by_customer.setdefault(str(candidate.customer_id), []).append(candidate)
        for customer, items in by_customer.items():
            target = labels.get(customer)
            if target is None:
                continue
            target = pd.Timestamp(target).date()
            distances = np.array([abs((c.local_date - target).days) for c in items])
            positives = distances <= int(tolerance_days)
            # Balance each customer's contribution and include distant negatives.
            for c, positive in zip(items, positives):
                xs.append(c.features); ys.append(int(positive))
        if not xs or len(set(ys)) < 2:
            return cls(np.zeros(candidates[0].features.shape[0] if candidates else 5),
                       np.ones(candidates[0].features.shape[0] if candidates else 5),
                       np.zeros(candidates[0].features.shape[0] if candidates else 5), 0.0, True)
        x = np.vstack(xs); y = np.asarray(ys, dtype=int)
        z, mean, scale = _safe_scale(x)
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=0, solver="liblinear")
        model.fit(z, y)
        return cls(mean, scale, model.coef_[0].astype(float), float(model.intercept_[0]), False)

    def score(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if self.fallback:
            return np.zeros(len(x))
        z = (x - self.mean) / self.scale
        return z @ self.coef + self.intercept

    def choose(self, candidates: Sequence[DateCandidate], measurements: pd.DataFrame | None = None,
               customer_id: str | None = None) -> date | None:
        items = [c for c in candidates if customer_id is None or str(c.customer_id) == str(customer_id)]
        if items and not self.fallback:
            scores = self.score(np.vstack([c.features for c in items]))
            # Stable earliest tie-break required by the protocol.
            best = max(range(len(items)), key=lambda i: (float(scores[i]), -items[i].local_date.toordinal()))
            return items[best].local_date
        if measurements is not None and customer_id is not None:
            return candidate_change_point(measurements, str(customer_id))
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


@dataclass
class CapacityHead:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def fit(cls, x: np.ndarray, capacities: Sequence[float], alpha: float = 1.0) -> "CapacityHead":
        x = np.asarray(x, dtype=np.float64); y = np.asarray(capacities, dtype=np.float64)
        valid = np.isfinite(y) & (y > 0)
        if x.ndim != 2 or len(x) != len(y) or valid.sum() < 2:
            raise ValueError("capacity head requires at least two real positive training capacities")
        z, mean, scale = _safe_scale(x[valid])
        model = Ridge(alpha=float(alpha))
        model.fit(z, np.log(y[valid]))
        return cls(mean, scale, model.coef_.astype(float), float(model.intercept_))

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        value = np.exp(np.clip(z @ self.coef + self.intercept, -30.0, 30.0))
        return np.where(np.isfinite(value) & (value > 0), value, np.nan)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def training_date_labels(labels: pd.DataFrame) -> dict[str, date | None]:
    result: dict[str, date | None] = {}
    for _, row in labels.iterrows():
        value = row.get("pv_installation_date")
        valid = row.get("date_target_valid", True)
        result[str(row.customer_id)] = None if (not bool(valid) or pd.isna(value)) else pd.Timestamp(value).date()
    return result


def fit_heads(customer_features: np.ndarray, customer_ids: Sequence[str], labels: pd.DataFrame,
              measurements: pd.DataFrame, *, capacity_features: np.ndarray | None = None,
              candidate_tolerance_days: int = 3, candidates: Sequence[DateCandidate] | None = None) -> tuple[PresenceHead, DateHead, CapacityHead | None, dict]:
    """Fit all heads from explicitly supplied training customers only."""
    label_map = labels.assign(customer_id=labels.customer_id.astype(str)).set_index("customer_id")
    ids = [str(x) for x in customer_ids]
    present = np.array([bool(label_map.loc[i, "pv_installed"]) for i in ids])
    presence = PresenceHead.fit(customer_features, present)
    dates = training_date_labels(labels)
    candidates = list(candidates) if candidates is not None else date_candidates(measurements, ids)
    date_head = DateHead.fit(candidates, dates, candidate_tolerance_days)
    capacities = np.array([label_map.loc[i, "pv_capacity_kwp"] for i in ids], dtype=float)
    valid_capacity = np.isfinite(capacities) & (capacities > 0)
    if int(valid_capacity.sum()) < 2:
        raise ValueError("capacity head requires at least two real positive training capacities")
    cap = CapacityHead.fit(capacity_features if capacity_features is not None else customer_features, capacities)
    return presence, date_head, cap, {"candidate_count": len(candidates), "date_fallback": date_head.fallback,
                                      "capacity_train_positive": int(valid_capacity.sum())}
