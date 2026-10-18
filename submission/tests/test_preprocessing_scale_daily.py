from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from neuralwave_data.features import daily_features
from neuralwave_data.labels import parse_extensions


def _rows(customer: str, day: date, slots: range, value: float = 1.0) -> list[dict]:
    return [
        {
            "customer_id": customer,
            "local_date": day,
            "local_slot": slot,
            "value_status": "observed",
            "energy_kwh": value,
        }
        for slot in slots
    ]


def test_daily_features_vectorized_preserves_dst_and_absent_days() -> None:
    rows = _rows("PODSYN001", date(2024, 3, 30), range(0, 1), 1.0)
    rows += _rows("PODSYN001", date(2024, 4, 1), range(0, 1), 2.0)
    features = daily_features(pd.DataFrame(rows))

    assert features.local_date.tolist() == [date(2024, 3, 30), date(2024, 3, 31), date(2024, 4, 1)]
    assert features.expected_intervals.tolist() == [96, 92, 96]
    assert features.observed_intervals.tolist() == [1, 0, 1]
    assert features.energy_observed_kwh.iloc[0] == 1.0 and pd.isna(features.energy_observed_kwh.iloc[1]) and features.energy_observed_kwh.iloc[2] == 2.0


def test_daily_features_exact_band_statistics_and_coverage() -> None:
    day = date(2024, 5, 6)
    rows = []
    for slot in range(96):
        if slot < 20:
            value = 2.0
        elif 24 <= slot < 40:
            value = 3.0
        elif 40 <= slot < 64:
            value = 4.0
        elif 72 <= slot < 92:
            value = 5.0
        else:
            value = 1.0
        rows += _rows("PODSYN002", day, range(slot, slot + 1), value)

    features = daily_features(pd.DataFrame(rows), coverage_threshold=0.9)
    row = features.iloc[0]
    assert row.expected_intervals == 96 and row.observed_intervals == 96
    assert row.night_mean_kwh == 2.0 and row.night_median_kwh == 2.0
    assert row.midday_mean_kwh == 4.0 and row.midday_median_kwh == 4.0
    # Both documented bands contain 20 slots: 80 kWh / 40 kWh.
    assert row.midday_night_ratio == pytest.approx(2.0)
    assert row.night_coverage_fraction == 1.0 and row.midday_coverage_fraction == 1.0


def test_parse_extensions_allows_multiple_events_and_rejects_duplicate_event(tmp_path) -> None:
    path = tmp_path / "extensions.csv"
    path.write_text(
        "customer_id;extension_date;pv_capacity_after_extension_kwp\n"
        "PODSYN003;2024-06-01;2.5\n"
        "PODSYN003;2025-06-01;3.5\n",
        encoding="utf-8",
    )
    labels = pd.DataFrame(
        {
            "customer_id": ["PODSYN003"],
            "pv_installed": [True],
            "pv_installation_date": [date(2020, 1, 1)],
        }
    )
    source = {"_path": str(path), "delimiter": ";", "decimal": ".", "encoding": "utf-8"}
    parsed = parse_extensions(source, labels)
    assert parsed.extension_date.tolist() == [date(2024, 6, 1), date(2025, 6, 1)]
    assert parsed.pv_capacity_after_extension_kwp.tolist() == [2.5, 3.5]

    path.write_text(
        "customer_id;extension_date;pv_capacity_after_extension_kwp\n"
        "PODSYN003;2024-06-01;2.5\n"
        "PODSYN003;2024-06-01;3.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate events"):
        parse_extensions(source, labels)
