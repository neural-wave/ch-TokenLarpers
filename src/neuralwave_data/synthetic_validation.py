"""Independent hard-gate validation for synthetic augmentation artifacts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .synthesis import MEASUREMENT_COLUMNS, SCHEMA_VERSION


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def validate_equations(measurements: pd.DataFrame, provenance: pd.DataFrame,
                       latent: pd.DataFrame, meter_mode: str | None = None,
                       tolerance: float = 1e-9) -> dict[str, Any]:
    """Recompute every observed output reading from independent latent components."""
    _require(measurements, MEASUREMENT_COLUMNS, "measurements")
    _require(provenance, ["customer_id", "scenario", "meter_mode", "onset_date"], "provenance")
    _require(latent, ["customer_id", "timestamp_epoch", "baseline_kwh", "demand_kwh", "pv_kwh"], "latent")
    if meter_mode is not None and meter_mode not in {"import_only", "signed_net"}:
        raise ValueError("unknown meter mode")
    if measurements.customer_id.astype(str).duplicated().any() and measurements[["customer_id", "timestamp_epoch"]].duplicated().any():
        raise ValueError("duplicate generated measurement keys")
    if provenance.customer_id.astype(str).duplicated().any():
        raise ValueError("duplicate provenance customer IDs")
    if latent[["customer_id", "timestamp_epoch"]].duplicated().any():
        raise ValueError("duplicate latent keys")
    if set(measurements.customer_id.astype(str)) != set(provenance.customer_id.astype(str)):
        raise ValueError("missing provenance for generated customer")
    m = measurements.copy()
    m["customer_id"] = m.customer_id.astype(str)
    p = provenance.copy()
    p["customer_id"] = p.customer_id.astype(str)
    l = latent.copy()
    l["customer_id"] = l.customer_id.astype(str)
    joined = m.merge(l, on=["customer_id", "timestamp_epoch"], how="left", validate="one_to_one", indicator=True)
    if (joined["_merge"] != "both").any():
        raise ValueError("latent values missing for generated measurement")
    joined = joined.merge(p[["customer_id", "scenario", "meter_mode", "onset_date", "ramp_days",
                             "basis_width_hours", "basis_asymmetry"]], on="customer_id", how="left", validate="many_to_one")
    if joined[["baseline_kwh", "demand_kwh", "pv_kwh"]].apply(lambda col: (~np.isfinite(col)).any()).any():
        # Missing source intervals are represented by NaN baseline/demand and must remain missing;
        # pv is always defined, including for those intervals.
        for column in ["baseline_kwh", "demand_kwh"]:
            bad = joined[column].notna() & ~np.isfinite(joined[column])
            if bad.any():
                raise ValueError(f"nonfinite latent {column}")
        if (~np.isfinite(joined.pv_kwh)).any():
            raise ValueError("nonfinite latent pv_kwh")
    if (joined.pv_kwh < -tolerance).any():
        raise ValueError("negative PV latent values")
    expected = np.full(len(joined), np.nan, dtype=float)
    for scenario, group in joined.groupby("scenario", sort=False):
        values = group.demand_kwh.to_numpy(dtype=float) - group.pv_kwh.to_numpy(dtype=float)
        if scenario in {"original_negative", "demand_change_negative"}:
            values = group.demand_kwh.to_numpy(dtype=float)
        mode = group.meter_mode.iloc[0]
        if mode == "import_only":
            finite = np.isfinite(values)
            values[finite] = np.maximum(values[finite], 0.0)
        elif mode != "signed_net":
            raise ValueError("unknown meter mode")
        expected[group.index.to_numpy()] = values
    actual = pd.to_numeric(joined.energy_kwh, errors="coerce").to_numpy(dtype=float)
    missing_expected = ~np.isfinite(expected)
    if np.isfinite(actual[missing_expected]).any():
        raise ValueError("source missingness was not preserved")
    finite = np.isfinite(expected)
    if (~np.isfinite(actual[finite])).any() or not np.allclose(actual[finite], expected[finite], rtol=0, atol=tolerance):
        raise ValueError("synthetic meter equation mismatch")
    statuses = joined.value_status.astype(str)
    if ((statuses == "observed") & ~np.isfinite(actual)).any():
        raise ValueError("observed status has null generated energy")
    # PV must be zero before onset and outside the declared compact support.
    for customer, group in joined.groupby("customer_id", sort=False):
        onset = group.onset_date.iloc[0]
        pv = group.pv_kwh.to_numpy(dtype=float)
        if pd.isna(onset):
            if np.any(pv > tolerance):
                raise ValueError("negative scenario contains PV")
            continue
        onset_day = _as_date(onset)
        local_dates = group.local_date.map(_as_date).to_numpy()
        if np.any(pv[local_dates < onset_day] > tolerance):
            raise ValueError("PV generated before onset")
        hours = ((group.timestamp_epoch.to_numpy(dtype=np.int64) + 3600) % 86400) / 3600.0
        outside = (hours < 5.0) | (hours > 21.0)
        if np.any(pv[outside] > tolerance):
            raise ValueError("PV generated outside declared support")
    return {"rows": int(len(joined)), "customers": int(provenance.customer_id.nunique()), "equations": True}


def validate_labels(synthetic_labels: pd.DataFrame, provenance: pd.DataFrame) -> dict[str, Any]:
    _require(synthetic_labels, ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp",
                                "date_target_valid", "capacity_target_valid"], "synthetic_labels")
    _require(provenance, ["customer_id", "scenario", "onset_date"], "provenance")
    labels = synthetic_labels.copy()
    labels["customer_id"] = labels.customer_id.astype(str)
    prov = provenance.copy()
    prov["customer_id"] = prov.customer_id.astype(str)
    if labels.customer_id.duplicated().any() or set(labels.customer_id) != set(prov.customer_id):
        raise ValueError("synthetic labels do not match generated customers")
    if labels.pv_capacity_kwp.notna().any() or labels.capacity_target_valid.astype(bool).any():
        raise ValueError("synthetic capacity target must always be null/invalid")
    if not labels.pv_installed.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("synthetic pv_installed must be boolean")
    for _, row in labels.iterrows():
        p = prov.loc[prov.customer_id == row.customer_id].iloc[0]
        positive = p.scenario in {"pv_positive", "pv_demand_change_positive"}
        if bool(row.pv_installed) != positive or bool(row.date_target_valid) != positive:
            raise ValueError("inconsistent synthetic PV/date target")
        has_date = pd.notna(row.pv_installation_date)
        if has_date != positive or (positive and _as_date(row.pv_installation_date) != _as_date(p.onset_date)):
            raise ValueError("inconsistent synthetic installation date")
    return {"customers": int(len(labels)), "labels": True}


def validate_outputs(measurements: pd.DataFrame, synthetic_labels: pd.DataFrame,
                     provenance: pd.DataFrame, latent: pd.DataFrame,
                     held_out_fold: int | None = None, meter_mode: str | None = None) -> dict[str, Any]:
    result = validate_equations(measurements, provenance, latent, meter_mode=meter_mode)
    result.update(validate_labels(synthetic_labels, provenance))
    if held_out_fold is not None and (pd.to_numeric(provenance.source_fold, errors="coerce") == int(held_out_fold)).any():
        raise ValueError("held-out source ancestry present in synthetic output")
    if provenance.source_customer_id.astype(str).duplicated().any():
        # Four scenarios intentionally share a source; this is not an error.
        pass
    return result


def validate_synthetic_directory(directory: str | Path, held_out_fold: int | None = None,
                                  meter_mode: str | None = None) -> dict[str, Any]:
    directory = Path(directory)
    manifest_path = directory / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError("synthetic run manifest is missing")
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported synthetic schema version")
    measurements = pd.read_parquet(directory / "measurements.parquet")
    labels = pd.read_parquet(directory / "synthetic_labels.parquet")
    provenance = pd.read_parquet(directory / "provenance.parquet")
    latent = pd.read_parquet(directory / "latent.parquet")
    return validate_outputs(measurements, labels, provenance, latent, held_out_fold, meter_mode)

# Friendly aliases used by downstream smoke tests.
validate = validate_outputs
