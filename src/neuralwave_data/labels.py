from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
import math
import re

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold

from .io import _columns, _source_path
from .schema import EXTENSION_COLUMNS, LABEL_COLUMNS, SPLIT_COLUMNS, arrow_table, empty_extensions, empty_labels, empty_splits


def _read_source(source: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(source["_path"], sep=source.get("delimiter", ";"), encoding=source.get("encoding", "utf-8"),
                       dtype=str, keep_default_na=False, na_filter=False)


def _id_column(frame: pd.DataFrame) -> str:
    return _columns(frame, "customer_id")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        raise ValueError("pv_installed must be an explicit boolean")
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError("pv_installed must be true/false, 1/0, or yes/no")


def _date(value: Any, field: str, nullable: bool = True) -> date | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        if nullable:
            return None
        raise ValueError(f"{field} is required")
    text = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(f"{field} must be a plain ISO date")
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field} must be a plain ISO date")
    return parsed.date()


def _capacity(value: Any, field: str, decimal: str = ".", nullable: bool = True) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        if nullable:
            return None
        raise ValueError(f"{field} is required")
    text = str(value).strip().replace(decimal, ".") if decimal != "." else str(value).strip()
    try:
        number = float(text)
    except Exception as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _find(frame: pd.DataFrame, names: tuple[str, ...], field: str, required: bool = True) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"labels are missing {field} column")
    return None


def parse_labels(source: dict[str, Any]) -> pd.DataFrame:
    frame = _read_source(source)
    id_col = _id_column(frame)
    bool_col = _find(frame, ("pv_installed", "PV_installed", "is_pv", "pv"), "pv_installed")
    date_col = _find(frame, ("pv_installation_date", "installation_date", "PV_installation_date"), "pv_installation_date", False)
    cap_col = _find(frame, ("pv_capacity_kwp", "capacity_kwp", "PV_capacity_kwp"), "pv_capacity_kwp", False)
    rows = []
    seen: set[str] = set()
    for values in frame.itertuples(index=False, name=None):
        row = dict(zip(frame.columns, values))
        raw_id = row[id_col]
        if raw_id is None or pd.isna(raw_id) or str(raw_id) == "":
            raise ValueError("labels contain null or empty customer_id")
        customer = str(raw_id)
        if customer in seen:
            raise ValueError("labels contain duplicate customer_id")
        seen.add(customer)
        installed = _parse_bool(row[bool_col])
        installation = _date(row[date_col], "pv_installation_date") if date_col else None
        capacity = _capacity(row[cap_col], "pv_capacity_kwp", source.get("decimal", ".")) if cap_col else None
        if not installed and (installation is not None or capacity is not None):
            raise ValueError("non-PV labels must have null date and capacity")
        rows.append((customer, installed, installation, capacity))
    return pd.DataFrame(rows, columns=LABEL_COLUMNS) if rows else empty_labels()


def parse_extensions(source: dict[str, Any], labels: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = _read_source(source)
    id_col = _id_column(frame)
    date_col = _find(frame, ("extension_date", "date", "pv_extension_date"), "extension_date")
    cap_col = _find(frame, ("pv_capacity_after_extension_kwp", "capacity_after_extension_kwp", "extension_capacity_kwp", "capacity_kwp"), "pv_capacity_after_extension_kwp")
    rows = []
    seen: set[tuple[str, date]] = set()
    installed_dates = {} if labels is None else dict(zip(labels["customer_id"], labels["pv_installation_date"]))
    known_labels = {} if labels is None else dict(zip(labels["customer_id"], labels["pv_installed"]))
    for values in frame.itertuples(index=False, name=None):
        row = dict(zip(frame.columns, values))
        raw_id = row[id_col]
        if raw_id is None or pd.isna(raw_id) or str(raw_id) == "":
            raise ValueError("extensions contain null or empty customer_id")
        customer = str(raw_id)
        extension_date = _date(row[date_col], "extension_date", nullable=False)
        key = (customer, extension_date)
        if key in seen:
            raise ValueError("extensions contain duplicate events")
        seen.add(key)
        if labels is not None and (customer not in known_labels or not bool(known_labels[customer])):
            raise ValueError("extensions require a known PV-labelled customer")
        capacity = _capacity(row[cap_col], "pv_capacity_after_extension_kwp", source.get("decimal", "."), nullable=False)
        initial = installed_dates.get(customer)
        if initial is not None and extension_date <= initial:
            raise ValueError("extension dates must follow installation dates")
        rows.append((customer, extension_date, capacity))
    return pd.DataFrame(rows, columns=EXTENSION_COLUMNS) if rows else empty_extensions()


def write_frame(frame: pd.DataFrame, output: str | Path, kind: str) -> None:
    pq.write_table(arrow_table(frame, kind), output)


def make_splits(labels: pd.DataFrame, requested: int, seed: int) -> tuple[pd.DataFrame, int]:
    if requested < 2:
        raise ValueError("fold count must be at least 2")
    if labels.empty:
        raise ValueError("cannot create folds without labels")
    counts = labels["pv_installed"].value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        raise ValueError("customer folds need at least two customers in each class")
    actual = min(int(requested), int(counts.min()))
    ordered = labels.sort_values("customer_id", kind="mergesort").reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=actual, shuffle=True, random_state=int(seed))
    folds = pd.Series(index=ordered.index, dtype="int64")
    for fold, (_, test) in enumerate(splitter.split(ordered["customer_id"], ordered["pv_installed"])):
        folds.iloc[test] = fold
    result = pd.DataFrame({"customer_id": ordered["customer_id"].astype("string"), "fold": folds.astype("int64")})
    return result[SPLIT_COLUMNS], actual
