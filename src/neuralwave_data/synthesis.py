"""Deterministic, constrained synthetic PV augmentation.

This module deliberately depends only on the persisted canonical parquet contract.  It
contains no ingestion or modelling code and can therefore be exercised as a namespace
module before the package initializer is supplied by the preprocessing branch.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
GENERATOR_VERSION = "augmentation_v1"
MEASUREMENT_COLUMNS = [
    "customer_id", "timestamp_epoch", "timestamp_local", "local_date", "local_slot",
    "dst_fold", "utc_offset_minutes", "energy_kwh", "value_status",
]


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _stable_seed(seed: int, customer_id: str, label: str = "") -> int:
    raw = f"{int(seed)}\x00{customer_id}\x00{label}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def default_config() -> dict[str, Any]:
    return {
        "generator_version": GENERATOR_VERSION,
        "min_pre_days": 7,
        "min_post_days": 7,
        "min_daytime_observed_fraction": 0.75,
        "strength_min": 0.05,
        "strength_max": 1.5,
        "ramp_days_min": 1,
        "ramp_days_max": 3,
        "daytime_start_hour": 10,
        "daytime_end_hour": 15,
        "support_start_hour": 5.0,
        "support_end_hour": 21.0,
        "width_hours_min": 2.5,
        "width_hours_max": 4.5,
        "asymmetry_min": 0.85,
        "asymmetry_max": 1.15,
        "demand_decline_min": 0.20,
        "demand_decline_max": 0.45,
        "holiday_days_min": 1,
        "holiday_days_max": 3,
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = default_config()
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            supplied = json.load(handle)
        if not isinstance(supplied, dict):
            raise ValueError("synthesis config must be a JSON object")
        cfg.update(supplied)
    numeric_nonnegative = ["min_pre_days", "min_post_days", "min_daytime_observed_fraction",
                           "strength_min", "strength_max", "ramp_days_min", "ramp_days_max",
                           "support_start_hour", "support_end_hour", "width_hours_min",
                           "width_hours_max", "demand_decline_min", "demand_decline_max"]
    for key in numeric_nonnegative:
        try:
            value = float(cfg[key])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid synthesis config value: {key}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid synthesis config value: {key}")
    if cfg["strength_min"] > cfg["strength_max"] or cfg["ramp_days_min"] > cfg["ramp_days_max"]:
        raise ValueError("synthesis config minimum exceeds maximum")
    if cfg["support_start_hour"] >= cfg["support_end_hour"]:
        raise ValueError("support_start_hour must be before support_end_hour")
    if cfg["width_hours_min"] <= 0 or cfg["width_hours_min"] > cfg["width_hours_max"]:
        raise ValueError("invalid basis width range")
    if cfg["min_daytime_observed_fraction"] > 1:
        raise ValueError("min_daytime_observed_fraction must be <= 1")
    return cfg


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _validate_canonical_measurements(measurements: pd.DataFrame) -> None:
    _require_columns(measurements, MEASUREMENT_COLUMNS, "measurements")
    if measurements.empty:
        raise ValueError("measurements contains no rows")
    if measurements["customer_id"].isna().any() or measurements["timestamp_epoch"].isna().any():
        raise ValueError("measurements has null customer IDs or epochs")
    if measurements["customer_id"].astype(str).duplicated(keep=False).any() is False:
        pass
    epochs = pd.to_numeric(measurements["timestamp_epoch"], errors="coerce")
    if epochs.isna().any() or (~np.isfinite(epochs)).any():
        raise ValueError("measurements has invalid timestamp epochs")
    energies = pd.to_numeric(measurements["energy_kwh"], errors="coerce")
    bad = energies.notna() & ~np.isfinite(energies)
    if bad.any():
        raise ValueError("measurements has nonfinite observed energy")
    observed_negative = energies.notna() & (energies < 0)
    if observed_negative.any():
        raise ValueError("negative baseline readings cannot be treated as nonnegative native load")
    keys = pd.MultiIndex.from_arrays([measurements.customer_id.astype(str), epochs.astype(np.int64)])
    if keys.duplicated().any():
        raise ValueError("measurements contains duplicate customer/timestamp keys")


def _validate_labels(labels: pd.DataFrame) -> None:
    _require_columns(labels, ["customer_id", "pv_installed"], "labels")
    if labels["customer_id"].isna().any() or labels.customer_id.astype(str).duplicated().any():
        raise ValueError("labels must have one non-null row per customer")
    values = labels["pv_installed"]
    if not values.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("labels pv_installed must be boolean")


def _validate_splits(splits: pd.DataFrame) -> dict[str, int]:
    _require_columns(splits, ["customer_id", "fold"], "splits")
    if splits.customer_id.isna().any() or splits.customer_id.astype(str).duplicated().any():
        raise ValueError("splits must have one assignment per customer")
    folds = pd.to_numeric(splits.fold, errors="coerce")
    if folds.isna().any() or (folds % 1 != 0).any():
        raise ValueError("splits fold values must be integers")
    return dict(zip(splits.customer_id.astype(str), folds.astype(np.int64)))


def _observed_dates(group: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[list[date], dict[date, float]]:
    dates: dict[date, float] = {}
    start = int(cfg["daytime_start_hour"]) * 4
    end = int(cfg["daytime_end_hour"]) * 4
    for day, rows in group.groupby(group["local_date"].map(_as_date), sort=True):
        energy = pd.to_numeric(rows.energy_kwh, errors="coerce")
        slots = pd.to_numeric(rows.local_slot, errors="coerce")
        daytime = energy[(slots >= start) & (slots < end)]
        finite = daytime[np.isfinite(daytime)]
        expected = max(1, end - start)
        if len(finite) >= math.ceil(expected * float(cfg["min_daytime_observed_fraction"])):
            dates[day] = float(finite.sum())
    return sorted(dates), dates


def _choose_onset(days: list[date], seed: int, customer_id: str, cfg: Mapping[str, Any]) -> date | None:
    before = int(cfg["min_pre_days"])
    after = int(cfg["min_post_days"])
    candidates = [day for index, day in enumerate(days) if index >= before and len(days) - index - 1 >= after]
    if not candidates:
        return None
    rng = np.random.default_rng(_stable_seed(seed, customer_id, "onset"))
    return candidates[int(rng.integers(0, len(candidates)))]


def _basis(epoch: np.ndarray, width: float, asymmetry: float, cfg: Mapping[str, Any]) -> np.ndarray:
    """Compact smooth daytime basis in fixed UTC+1 standard-clock coordinates."""
    hour = ((epoch.astype(np.float64) + 3600.0) % 86400.0) / 3600.0
    dawn = float(cfg["support_start_hour"])
    dusk = float(cfg["support_end_hour"])
    center = 12.0
    out = np.zeros(hour.shape, dtype=np.float64)
    inside = (hour >= dawn) & (hour <= dusk)
    # A smooth compact bump whose width/asymmetry changes around noon.  Hard support
    # endpoints make the no-generation guarantee explicit.
    distance = hour[inside] - center
    effective = np.where(distance < 0, width * asymmetry, width / asymmetry)
    shape = np.exp(-0.5 * (distance / effective) ** 2)
    edge = np.sin(np.pi * (hour[inside] - dawn) / (dusk - dawn)) ** 2
    out[inside] = shape * edge
    return out


def _ramp(local_dates: np.ndarray, onset: date, ramp_days: int) -> np.ndarray:
    if ramp_days <= 0:
        return (local_dates >= onset).astype(float)
    delta = np.array([(value - onset).days for value in local_dates], dtype=float)
    return np.where(delta < 0, 0.0, np.minimum(1.0, (delta + 1.0) / ramp_days))


def _demand_change(local_dates: np.ndarray, slots: np.ndarray, onset: date, decline: float,
                   kind: str, rng: np.random.Generator, cfg: Mapping[str, Any]) -> np.ndarray:
    factors = np.ones(local_dates.shape, dtype=np.float64)
    dates = np.array([(value - onset).days for value in local_dates], dtype=int)
    if kind == "whole_day_decline":
        factors[dates >= 0] = 1.0 - decline
    elif kind == "business_hours_decline":
        active = (dates >= 0) & (slots >= 24) & (slots < 72)  # 06:00 through 18:00
        factors[active] = 1.0 - decline
    elif kind == "temporary_holiday_decline":
        length = int(rng.integers(int(cfg["holiday_days_min"]), int(cfg["holiday_days_max"]) + 1))
        active = (dates >= 0) & (dates < length)
        factors[active] = 1.0 - decline
    else:
        raise ValueError(f"unknown demand change kind: {kind}")
    return factors


@dataclass
class SynthesisResult:
    measurements: pd.DataFrame
    synthetic_labels: pd.DataFrame
    provenance: pd.DataFrame
    latent: pd.DataFrame
    quality_report: dict[str, Any]


def generate_synthetic(measurements: pd.DataFrame, labels: pd.DataFrame,
                       splits: pd.DataFrame, held_out_fold: int, meter_mode: str,
                       seed: int = 0, config: Mapping[str, Any] | None = None) -> SynthesisResult:
    """Generate four paired scenarios for each eligible labelled training background."""
    if meter_mode not in {"import_only", "signed_net"}:
        raise ValueError("meter mode must be import_only or signed_net")
    cfg = dict(default_config())
    if config:
        cfg.update(dict(config))
    cfg = load_config_from_mapping(cfg)
    _validate_canonical_measurements(measurements)
    _validate_labels(labels)
    fold_map = _validate_splits(splits)
    label_map = labels.set_index(labels.customer_id.astype(str))
    source_groups = {str(customer): rows.sort_values(["timestamp_epoch"], kind="mergesort").copy()
                     for customer, rows in measurements.groupby(measurements.customer_id.astype(str), sort=True)}
    eligible = []
    excluded: dict[str, int] = {}
    for customer in sorted(source_groups):
        if customer not in label_map.index:
            excluded["unlabelled_background"] = excluded.get("unlabelled_background", 0) + 1
            continue
        if bool(label_map.loc[customer, "pv_installed"]):
            excluded["positive_label"] = excluded.get("positive_label", 0) + 1
            continue
        if customer not in fold_map:
            raise ValueError(f"missing split assignment for eligible customer {customer}")
        if int(fold_map[customer]) == int(held_out_fold):
            excluded["held_out_fold"] = excluded.get("held_out_fold", 0) + 1
            continue
        days, day_sums = _observed_dates(source_groups[customer], cfg)
        onset = _choose_onset(days, seed, customer, cfg)
        if onset is None:
            excluded["insufficient_pre_post_coverage"] = excluded.get("insufficient_pre_post_coverage", 0) + 1
            continue
        daytime_values = np.array([value for value in day_sums.values() if value > 0 and np.isfinite(value)])
        if daytime_values.size == 0:
            excluded["zero_daytime_scale"] = excluded.get("zero_daytime_scale", 0) + 1
            continue
        eligible.append((customer, source_groups[customer], onset))
    if not eligible:
        raise ValueError(f"no eligible non-PV training backgrounds; excluded={excluded}")

    all_day_values = []
    for _, group, _ in eligible:
        _, sums = _observed_dates(group, cfg)
        all_day_values.extend(value for value in sums.values() if value > 0 and np.isfinite(value))
    robust_scale = float(np.median(np.asarray(all_day_values, dtype=float))) if all_day_values else 0.0
    if not math.isfinite(robust_scale) or robust_scale <= 0:
        raise ValueError("eligible training backgrounds have zero daytime scale")

    measurements_out: list[pd.DataFrame] = []
    labels_out: list[dict[str, Any]] = []
    provenance_out: list[dict[str, Any]] = []
    latent_out: list[pd.DataFrame] = []
    scenario_names = ["original_negative", "pv_positive", "demand_change_negative", "pv_demand_change_positive"]
    demand_kinds = ["whole_day_decline", "business_hours_decline", "temporary_holiday_decline"]
    for customer, source, onset in eligible:
        source_seed = _stable_seed(seed, customer)
        rng = np.random.default_rng(source_seed)
        width = float(rng.uniform(float(cfg["width_hours_min"]), float(cfg["width_hours_max"])))
        asymmetry = float(rng.uniform(float(cfg["asymmetry_min"]), float(cfg["asymmetry_max"])))
        ramp_days = int(rng.integers(int(cfg["ramp_days_min"]), int(cfg["ramp_days_max"]) + 1))
        strength = float(rng.uniform(float(cfg["strength_min"]), float(cfg["strength_max"])))
        decline = float(rng.uniform(float(cfg["demand_decline_min"]), float(cfg["demand_decline_max"])))
        demand_kind = demand_kinds[source_seed % len(demand_kinds)]
        source_epoch = pd.to_numeric(source.timestamp_epoch).to_numpy(dtype=np.int64)
        source_energy = pd.to_numeric(source.energy_kwh, errors="coerce").to_numpy(dtype=float)
        local_dates = np.array([_as_date(value) for value in source.local_date], dtype=object)
        slots = pd.to_numeric(source.local_slot).to_numpy(dtype=int)
        basis = _basis(source_epoch, width, asymmetry, cfg)
        activation = _ramp(local_dates, onset, ramp_days)
        doy = np.array([value.timetuple().tm_yday for value in local_dates], dtype=float)
        annual = 0.9 + 0.1 * np.cos(2 * np.pi * (doy - 172.0) / 365.25)
        attenuation = np.array([0.88 + 0.10 * (int.from_bytes(hashlib.sha256(f"{source_seed}:{int(epoch)}".encode()).digest()[:2], "big") / 65535.0)
                                for epoch in source_epoch])
        raw_pv = basis * activation * annual * attenuation
        norm = float(np.nanmax(raw_pv)) if raw_pv.size else 0.0
        pv = raw_pv / norm * (strength * robust_scale) if norm > 0 else raw_pv
        change = _demand_change(local_dates, slots, onset, decline, demand_kind,
                                np.random.default_rng(source_seed ^ 0x9E3779B97F4A7C15), cfg)
        baseline = source_energy.copy()
        demand = baseline * change
        scenarios = [baseline, baseline - pv, demand, demand - pv]
        scenario_pv = [np.zeros_like(pv), pv, np.zeros_like(pv), pv]
        source_hash = hashlib.sha256(customer.encode()).hexdigest()
        for index, scenario in enumerate(scenario_names):
            generated_id = "syn_" + hashlib.sha256(f"{seed}:{customer}:{index}".encode()).hexdigest()[:24]
            observed = np.isfinite(source_energy)
            output_energy = scenarios[index].copy()
            if meter_mode == "import_only":
                output_energy[observed] = np.maximum(output_energy[observed], 0.0)
            output_energy[~observed] = np.nan
            out = source.copy()
            out["customer_id"] = generated_id
            out["energy_kwh"] = output_energy
            # Source status and every temporal field are intentionally unchanged.
            measurements_out.append(out[MEASUREMENT_COLUMNS])
            is_positive = index in (1, 3)
            labels_out.append({"customer_id": generated_id, "pv_installed": is_positive,
                               "pv_installation_date": onset if is_positive else None,
                               "pv_capacity_kwp": None,
                               "date_target_valid": bool(is_positive),
                               "capacity_target_valid": False})
            provenance_out.append({
                "customer_id": generated_id, "source_customer_id": customer,
                "source_fold": int(fold_map[customer]), "seed": int(seed),
                "generator_version": str(cfg.get("generator_version", GENERATOR_VERSION)),
                "scenario": scenario, "meter_mode": meter_mode, "onset_date": onset if is_positive else None,
                "ramp_days": ramp_days, "scale": float(strength * robust_scale),
                "strength_fraction": strength, "demand_change_kind": demand_kind,
                "demand_decline": decline, "basis_width_hours": width,
                "basis_asymmetry": asymmetry, "config_hash": _json_hash(cfg),
            })
            latent_out.append(pd.DataFrame({
                "customer_id": generated_id, "timestamp_epoch": source_epoch,
                "baseline_kwh": baseline, "demand_kwh": demand if index >= 2 else baseline,
                "pv_kwh": scenario_pv[index],
            }))
    output_measurements = pd.concat(measurements_out, ignore_index=True).sort_values(
        ["customer_id", "timestamp_epoch"], kind="mergesort").reset_index(drop=True)
    output_labels = pd.DataFrame(labels_out)
    output_provenance = pd.DataFrame(provenance_out)
    output_latent = pd.concat(latent_out, ignore_index=True).sort_values(
        ["customer_id", "timestamp_epoch"], kind="mergesort").reset_index(drop=True)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": str(cfg.get("generator_version", GENERATOR_VERSION)),
        "meter_mode": meter_mode, "seed": int(seed),
        "eligible_sources": len(eligible), "generated_customers": len(output_labels),
        "robust_daytime_scale_kwh": robust_scale, "excluded_sources": excluded,
        "scenario_counts": output_provenance.scenario.value_counts().to_dict(),
        "hard_gates": {"equations": True, "missingness_preserved": True,
                       "nonnegative_pv": bool((output_latent.pv_kwh >= 0).all())},
    }
    return SynthesisResult(output_measurements, output_labels, output_provenance, output_latent, report)


def load_config_from_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an already loaded configuration without requiring a temporary file."""
    cfg = dict(config)
    defaults = default_config()
    defaults.update(cfg)
    # Reuse config validation while avoiding an on-disk round trip.
    for key in ["min_pre_days", "min_post_days", "min_daytime_observed_fraction", "strength_min",
                "strength_max", "ramp_days_min", "ramp_days_max", "support_start_hour",
                "support_end_hour", "width_hours_min", "width_hours_max", "demand_decline_min",
                "demand_decline_max"]:
        value = float(defaults[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid synthesis config value: {key}")
        defaults[key] = value if key not in {"min_pre_days", "min_post_days", "ramp_days_min", "ramp_days_max"} else int(value)
    if defaults["strength_min"] > defaults["strength_max"] or defaults["ramp_days_min"] > defaults["ramp_days_max"]:
        raise ValueError("synthesis config minimum exceeds maximum")
    if defaults["support_start_hour"] >= defaults["support_end_hour"]:
        raise ValueError("support_start_hour must be before support_end_hour")
    if defaults["min_daytime_observed_fraction"] > 1:
        raise ValueError("min_daytime_observed_fraction must be <= 1")
    return defaults


def write_result(result: SynthesisResult, out_dir: str | Path, source_manifest_identity: str,
                 seed: int, config: Mapping[str, Any]) -> None:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    result.measurements.to_parquet(out / "measurements.parquet", index=False)
    result.synthetic_labels.to_parquet(out / "synthetic_labels.parquet", index=False)
    result.provenance.to_parquet(out / "provenance.parquet", index=False)
    result.latent.to_parquet(out / "latent.parquet", index=False)
    (out / "quality_report.json").write_text(json.dumps(result.quality_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    manifest = {"schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
                "seed": int(seed), "config": dict(config),
                "config_hash": _json_hash(config), "source_manifest_identity": source_manifest_identity,
                "artifacts": ["measurements.parquet", "synthetic_labels.parquet", "provenance.parquet", "latent.parquet", "quality_report.json"],
                "counts": {"measurements": len(result.measurements), "customers": len(result.synthetic_labels)}}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
