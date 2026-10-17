from __future__ import annotations

import numpy as np
import pandas as pd

from pvdetect.features import feature_names, fitted_features, label_free_features
from pvdetect.pipeline import build_feature_table
from pvdetect.procedure import export_enabled_for
from pvdetect.season import zero_climatology
from pvdetect.synthetic import CustomerSpec, PVSpec, generate_customer
from tests.helpers import as_loaded


def _feats(kwp: float | None) -> dict:
    pv = PVSpec("2023-04-10", kwp) if kwp else None
    spec = CustomerSpec("000001", "2022-01-01", "2025-12-31", pv=pv)
    table = build_feature_table(as_loaded(generate_customer(spec, 5)))
    cd = table.customers["000001"]
    lf = label_free_features(cd)
    return {**lf, **fitted_features(cd, zero_climatology(), lf)[0]}


def test_capacity_signal_grows_with_capacity():
    """dip_kw saturates once PV exceeds midday load; the clipping-window features keep scaling."""
    small, large = _feats(4.0), _feats(10.0)
    assert large["zero_window_h"] > small["zero_window_h"]
    assert large["log_kwp_phys"] > small["log_kwp_phys"]


def test_pv_shifts_summer_signal_and_centres_dip_near_noon():
    pv, no = _feats(6.0), _feats(None)
    assert pv["summer_ds"] < no["summer_ds"]
    assert 10.5 <= pv["dip_center_hour"] <= 14.5


def test_export_features_only_with_signed_data():
    frames = [
        generate_customer(CustomerSpec(f"{i:06d}", "2023-01-01", "2023-12-31", pv=PVSpec("2023-03-01", 8.0, export=exp)), i)
        for i, exp in enumerate([False, False, False, True, True, True])
    ]
    table = build_feature_table(as_loaded(pd.concat(frames)))
    unsigned = ["000000", "000001", "000002"]
    assert not export_enabled_for(table, unsigned)
    assert "neg_frac" not in feature_names(False) and "export_kw" not in feature_names(False)
    assert export_enabled_for(table, table.ids)
    assert {"neg_frac", "export_kw"} <= set(feature_names(True))
    assert np.isfinite(label_free_features(table.customers["000004"])["export_kw"])
