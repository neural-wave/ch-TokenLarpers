"""Submission writers (PDF §4.1): UTF-8, semicolons, TRUE/FALSE, YYYY-MM-DD, decimal point."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

PRED_HEADER = ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"]
EXT_HEADER = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]


def _date(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return ""
    return pd.Timestamp(v).strftime("%Y-%m-%d")


def _kwp(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return f"{float(v):.2f}"


def write_predictions(pred: pd.DataFrame, out_dir: Path) -> None:
    with open(Path(out_dir) / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(PRED_HEADER)
        for row in pred.itertuples(index=False):
            on = bool(row.pv_installed)
            w.writerow(
                [
                    str(row.customer_id),
                    "TRUE" if on else "FALSE",
                    _date(row.pv_installation_date) if on else "",
                    _kwp(row.pv_capacity_kwp) if on else "",
                ]
            )


def write_extensions(ext: pd.DataFrame, out_dir: Path) -> None:
    with open(Path(out_dir) / "pv_extensions_predictions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(EXT_HEADER)
        for row in ext.itertuples(index=False):
            w.writerow([str(row.customer_id), _date(row.extension_date), _kwp(row.pv_capacity_after_extension_kwp)])
