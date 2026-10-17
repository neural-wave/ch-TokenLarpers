"""Leakage-safe temporal persistence scoring and diagnostics.

A window score is evidence for at most one elapsed UTC day.  In particular,
several overlapping windows in the same UTC day cannot manufacture persistence.
The public scorer deliberately has no label argument; labels are accepted only
by :func:`temporal_diagnostics`, after the score has been produced.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .windows import WindowBatch


_SECONDS_PER_DAY = 86_400
_LOCAL_ZONE = ZoneInfo("Europe/Zurich")


def _number(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _settings(config: Mapping[str, Any]) -> tuple[float, float, float]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    horizon_days = _number(config, "persistence_days", 14.0)
    min_coverage = _number(config, "persistence_min_coverage", 0.9)
    max_gap_days = _number(config, "persistence_max_gap_days", 2.0)
    if horizon_days <= 0:
        raise ValueError("persistence_days must be positive")
    if not 0 <= min_coverage <= 1:
        raise ValueError("persistence_min_coverage must be in [0, 1]")
    if max_gap_days < 0:
        raise ValueError("persistence_max_gap_days must be non-negative")
    return horizon_days * _SECONDS_PER_DAY, min_coverage, max_gap_days * _SECONDS_PER_DAY


def _threshold(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("threshold must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be numeric") from exc
    if not isfinite(result) or not 0 < result <= 1:
        raise ValueError("threshold must be finite and in (0, 1]")
    return result


def _local_date(epoch: int) -> date:
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).astimezone(_LOCAL_ZONE).date()


def _trace_arrays(batch: WindowBatch, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate a one-customer batch and return starts, centres, fractions, probs."""
    if not isinstance(batch, WindowBatch):
        raise TypeError("batch must be a WindowBatch")
    values = np.asarray(batch.values)
    observed = np.asarray(batch.observed)
    if values.ndim != 3 or values.shape[2] != 1 or values.shape[1] <= 0:
        raise ValueError("batch.values must have shape [N, L, 1]")
    if observed.shape != values.shape:
        raise ValueError("batch.observed must have the same shape as values")
    n, length, _ = values.shape
    starts = np.asarray(batch.start_epochs)
    if starts.ndim != 1 or len(starts) != n or not np.issubdtype(starts.dtype, np.integer):
        raise ValueError("batch.start_epochs must be an integer vector matching values")
    if len(batch.customer_ids) != n or any(not isinstance(x, str) or not x for x in batch.customer_ids):
        raise ValueError("batch.customer_ids must contain non-empty strings")
    if len(batch.center_dates) != n:
        raise ValueError("batch.center_dates must match values")
    customers = set(batch.customer_ids)
    if n and len(customers) != 1:
        raise ValueError("summarize_trace requires windows for exactly one customer")
    if n and np.any(starts[1:] <= starts[:-1]):
        raise ValueError("batch windows must be in strictly increasing start order")
    if not np.issubdtype(observed.dtype, np.bool_):
        raise ValueError("batch.observed must be boolean")
    p = np.asarray(probabilities)
    if p.ndim != 1 or len(p) != n:
        raise ValueError("probabilities must be a vector matching batch windows")
    if not np.issubdtype(p.dtype, np.number) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    centres = starts.astype(np.int64, copy=False) + (length // 2) * 900
    fractions = observed.reshape(n, length).mean(axis=1, dtype=np.float64)
    return starts.astype(np.int64, copy=False), centres, fractions, p.astype(np.float64, copy=False), length


def _empty_result() -> dict[str, Any]:
    return {
        "score": 0.0,
        "predicted_installation_date": None,
        "coverage_eligible": False,
        "confirmation_date": None,
    }


def _support_seconds(included: list[tuple[int, float, float, int]],
                     anchor: int, end: float) -> float:
    """Union one-day center supports, retaining the largest observed weight."""
    intervals: list[tuple[float, float, float]] = []
    points: set[float] = {float(anchor), float(end)}
    for center, _, fraction, _ in included:
        lo = max(float(anchor), float(center))
        hi = min(float(end), float(center + _SECONDS_PER_DAY))
        if hi <= lo:
            continue
        weight = min(1.0, max(0.0, float(fraction)))
        intervals.append((lo, hi, weight))
        points.add(lo)
        points.add(hi)
    ordered = sorted(points)
    support = 0.0
    for lo, hi in zip(ordered, ordered[1:]):
        if hi <= lo:
            continue
        midpoint = (lo + hi) / 2.0
        weight = max((weight for start, stop, weight in intervals
                      if start <= midpoint < stop), default=0.0)
        support += (hi - lo) * weight
    return support


def _blocks(centres: np.ndarray, fractions: np.ndarray, probabilities: np.ndarray,
            threshold: float, horizon: float, min_coverage: float,
            max_gap: float) -> tuple[dict[str, Any], bool]:
    """Evaluate rolling elapsed-time blocks over unique UTC day buckets."""
    if len(centres) == 0:
        return _empty_result(), False

    # A center contributes one UTC day at most. Keep median score for duplicate
    # centers and measured support without allowing overlap to add days.
    buckets: dict[int, list[int]] = {}
    for i, center in enumerate(centres.tolist()):
        buckets.setdefault(int(center) // _SECONDS_PER_DAY, []).append(i)
    rows: list[tuple[int, float, float, int]] = []
    for bucket, indexes in sorted(buckets.items()):
        first = indexes[0]
        rows.append((int(centres[first]), float(np.median(probabilities[indexes])),
                     float(np.max(fractions[indexes])), bucket))

    row_centres = np.asarray([row[0] for row in rows], dtype=np.int64)
    qualifying: list[tuple[float, int, int]] = []
    coverage_seen = False
    for start_index, (anchor, _, _, _) in enumerate(rows):
        end = anchor + horizon
        end_index = int(np.searchsorted(row_centres, end, side="left"))
        included = rows[start_index:end_index]
        if not included:
            continue
        support = _support_seconds(included, anchor, end)
        coverage = support / horizon
        gaps_ok = all((b[0] - a[0]) <= max_gap for a, b in zip(included, included[1:]))
        eligible = support > 0 and coverage >= min_coverage and gaps_ok
        if eligible:
            coverage_seen = True
            median = float(np.median([row[1] for row in included]))
            qualifying.append((median, start_index, len(included)))
    if not qualifying:
        result = _empty_result()
        result["coverage_eligible"] = bool(coverage_seen)
        return result, coverage_seen

    best_score = max(item[0] for item in qualifying)
    result: dict[str, Any] = {
        "score": float(best_score),
        "predicted_installation_date": None,
        "coverage_eligible": True,
        "confirmation_date": None,
    }
    # Earliest qualifying block at the requested threshold, independent of the
    # block selected for the maximum score.
    detected = next((item for item in qualifying if item[0] >= threshold), None)
    if detected is not None:
        _, start_index, count = detected
        anchor = rows[start_index][0]
        last_center = rows[start_index + count - 1][0]
        confirmation_epoch = min(last_center + _SECONDS_PER_DAY, anchor + horizon)
        result["predicted_installation_date"] = _local_date(anchor).isoformat()
        result["confirmation_date"] = _local_date(int(confirmation_epoch)).isoformat()
    return result, True


def _evaluate(centres: np.ndarray, fractions: np.ndarray, probabilities: np.ndarray,
              threshold: float, config: Mapping[str, Any]) -> dict[str, Any]:
    horizon, min_coverage, max_gap = _settings(config)
    result, _ = _blocks(centres, fractions, probabilities, threshold, horizon,
                        min_coverage, max_gap)
    return result


def summarize_trace(batch: WindowBatch, probabilities: np.ndarray, threshold: float,
                    config: dict) -> dict[str, Any]:
    """Summarize one customer's temporally ordered probability trace.

    The result contains only JSON-compatible primitive values.  Labels are
    intentionally absent: this function is used unchanged for calibration and
    outer validation.
    """
    threshold_value = _threshold(threshold)
    _, centres, fractions, probs, _ = _trace_arrays(batch, probabilities)
    return _evaluate(centres, fractions, probs, threshold_value, config)


def _label(label: dict) -> tuple[bool, date | None]:
    if not isinstance(label, Mapping):
        raise TypeError("label must be a mapping")
    if "pv_installed" not in label:
        raise ValueError("label must contain pv_installed")
    installed = label["pv_installed"]
    if not isinstance(installed, (bool, np.bool_)):
        raise ValueError("pv_installed must be an explicit boolean")
    raw_date = label.get("pv_installation_date")
    if raw_date is None or (isinstance(raw_date, str) and not raw_date.strip()):
        installation = None
    elif isinstance(raw_date, str):
        try:
            installation = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise ValueError("pv_installation_date must be ISO YYYY-MM-DD") from exc
    else:
        missing = pd.isna(raw_date)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            installation = None
        elif isinstance(raw_date, datetime):
            installation = raw_date.date()
        elif isinstance(raw_date, date):
            installation = raw_date
        else:
            raise ValueError("pv_installation_date must be a date or ISO string")
    # A known non-PV customer cannot carry an installation date. A PV label
    # with a censored/missing date remains presence-evaluable.
    if not bool(installed) and installation is not None:
        raise ValueError("non-PV labels require no installation date")
    return bool(installed), installation


def _local_midnight_epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=_LOCAL_ZONE).timestamp())


def temporal_diagnostics(batch: WindowBatch, probabilities: np.ndarray, label: dict,
                         threshold: float, config: dict) -> dict[str, Any]:
    """Compute label-dependent diagnostics after obtaining an unlabeled score."""
    threshold_value = _threshold(threshold)
    starts, centres, fractions, probs, length = _trace_arrays(batch, probabilities)
    summary = _evaluate(centres, fractions, probs, threshold_value, config)
    installed, installation = _label(label)

    result: dict[str, Any] = {
        "pre_installation_false_alarm": None,
        "onset_error_days": None,
        "absolute_onset_error_days": None,
        "date_reward": None,
        "late_installation": None,
        "non_pv_false_alarm": None,
        "observable_onset": None,
    }
    if not installed:
        result["non_pv_false_alarm"] = summary["predicted_installation_date"] is not None
        return result

    if installation is None:
        # Presence is known, but date-dependent metrics are censored.
        return result
    install_epoch = _local_midnight_epoch(installation)
    end_epochs = starts + (length - 1) * 900
    result["observable_onset"] = bool(
        len(starts) and int(np.min(starts)) <= install_epoch <= int(np.max(end_epochs))
    )
    predicted = summary["predicted_installation_date"]
    if predicted is None:
        result["date_reward"] = 0.0
    else:
        predicted_day = date.fromisoformat(predicted)
        error = int((predicted_day - installation).days)
        result["onset_error_days"] = error
        result["absolute_onset_error_days"] = abs(error)
        result["date_reward"] = float(max(0.0, 1.0 - abs(error) / 30.0))

    # A pre-installation window must end more than seven elapsed days before
    # local installation midnight; this avoids calling a transition window a
    # pre-installation negative.
    cutoff = _local_midnight_epoch(installation) - 7 * _SECONDS_PER_DAY
    pre = end_epochs < cutoff
    if np.any(pre):
        pre_summary = _evaluate(centres[pre], fractions[pre], probs[pre], threshold_value, config)
        result["pre_installation_false_alarm"] = (
            bool(pre_summary["predicted_installation_date"] is not None)
            if pre_summary["coverage_eligible"]
            else None
        )

    support_end = int(np.max(end_epochs)) if len(end_epochs) else None
    if support_end is not None:
        result["late_installation"] = bool(0 <= support_end - install_epoch <= 60 * _SECONDS_PER_DAY)
    return result


__all__ = ["summarize_trace", "temporal_diagnostics"]
