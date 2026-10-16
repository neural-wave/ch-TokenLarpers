from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pathlib
import sys
import types

_src = pathlib.Path(__file__).parents[1] / "src" / "neuralwave_data"
if "neuralwave_data" not in sys.modules:
    _namespace = types.ModuleType("neuralwave_data")
    _namespace.__path__ = [str(_src)]
    sys.modules["neuralwave_data"] = _namespace

from neuralwave_data.synthesis import generate_synthetic
from neuralwave_data.synthetic_validation import validate_outputs


def artificial_inputs(days: int = 29):
    rows = []
    start = pd.Timestamp("2024-01-01 00:00", tz="Europe/Zurich")
    for customer, multiplier in [("background-a", 1.0), ("background-b", 1.2), ("real-pv", 1.0)]:
        for day in range(days):
            for slot in range(96):
                timestamp = start + pd.Timedelta(days=day, minutes=15 * slot)
                energy = multiplier * (0.2 + (1.8 * np.sin(np.pi * (slot - 20) / 64) if 20 <= slot <= 84 else 0.0))
                rows.append({
                    "customer_id": customer, "timestamp_epoch": int(timestamp.timestamp()),
                    "timestamp_local": timestamp, "local_date": timestamp.date(),
                    "local_slot": slot, "dst_fold": 0, "utc_offset_minutes": 60,
                    "energy_kwh": energy, "value_status": "observed",
                })
    measurements = pd.DataFrame(rows)
    # One unavailable interval verifies that generation never fills source gaps.
    missing = (measurements.customer_id == "background-a") & (measurements.local_slot == 3) & (measurements.local_date == pd.Timestamp("2024-01-05").date())
    measurements.loc[missing, "energy_kwh"] = np.nan
    measurements.loc[missing, "value_status"] = "absent_interval"
    labels = pd.DataFrame({
        "customer_id": ["background-a", "background-b", "real-pv"],
        "pv_installed": [False, False, True],
        "pv_installation_date": [None, None, pd.Timestamp("2020-01-01").date()],
        "pv_capacity_kwp": [None, None, 4.0],
    })
    splits = pd.DataFrame({"customer_id": ["background-a", "background-b", "real-pv"], "fold": [1, 1, 0]})
    return measurements, labels, splits


def test_equations_missingness_and_four_scenarios():
    measurements, labels, splits = artificial_inputs()
    result = generate_synthetic(measurements, labels, splits, 0, "import_only", seed=17)
    assert set(result.provenance.scenario) == {
        "original_negative", "pv_positive", "demand_change_negative", "pv_demand_change_positive"
    }
    assert len(result.synthetic_labels) == 8
    assert result.synthetic_labels.pv_capacity_kwp.isna().all()
    assert (~result.synthetic_labels.capacity_target_valid).all()
    assert validate_outputs(result.measurements, result.synthetic_labels, result.provenance, result.latent, 0, "import_only")["equations"]
    missing = result.measurements.energy_kwh.isna()
    assert missing.sum() == 4


def test_signed_net_and_strong_clipping():
    measurements, labels, splits = artificial_inputs()
    cfg = {"strength_min": 1.5, "strength_max": 1.5}
    result = generate_synthetic(measurements, labels, splits, 0, "signed_net", seed=1, config=cfg)
    validate_outputs(result.measurements, result.synthetic_labels, result.provenance, result.latent, 0, "signed_net")
    positive = result.provenance.scenario == "pv_positive"
    ids = result.provenance.loc[positive, "customer_id"]
    vals = result.measurements[result.measurements.customer_id.isin(ids)].energy_kwh
    assert (vals < 0).any()


def test_held_out_and_real_pv_are_excluded_before_statistics():
    measurements, labels, splits = artificial_inputs()
    result = generate_synthetic(measurements, labels, splits, 0, "import_only", seed=2)
    assert set(result.provenance.source_fold) == {1}
    with pytest.raises(ValueError, match="no eligible"):
        generate_synthetic(measurements, labels, splits, 1, "import_only", seed=2)


def test_determinism_is_independent_of_input_order():
    measurements, labels, splits = artificial_inputs()
    first = generate_synthetic(measurements, labels, splits, 0, "import_only", seed=5)
    second = generate_synthetic(measurements.sample(frac=1, random_state=11), labels.sample(frac=1, random_state=12), splits.sample(frac=1, random_state=13), 0, "import_only", seed=5)
    pd.testing.assert_frame_equal(first.measurements, second.measurements)
    pd.testing.assert_frame_equal(first.latent, second.latent)


def test_validator_rejects_corrupt_equation_and_label():
    measurements, labels, splits = artificial_inputs()
    result = generate_synthetic(measurements, labels, splits, 0, "import_only", seed=5)
    corrupt = result.measurements.copy()
    index = corrupt.energy_kwh.first_valid_index()
    corrupt.loc[index, "energy_kwh"] += 0.1
    with pytest.raises(ValueError, match="equation"):
        validate_outputs(corrupt, result.synthetic_labels, result.provenance, result.latent)
    bad_labels = result.synthetic_labels.copy()
    bad_labels.loc[0, "pv_capacity_kwp"] = 2.0
    with pytest.raises(ValueError, match="capacity"):
        validate_outputs(result.measurements, bad_labels, result.provenance, result.latent)
