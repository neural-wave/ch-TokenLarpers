from __future__ import annotations

from pvdetect.features import baseline_features
from pvdetect.synthetic import CustomerSpec, PVSpec
from tests.helpers import customer_daily


def _feats(kwp: float | None) -> dict:
    pv = PVSpec("2023-04-10", kwp) if kwp else None
    spec = CustomerSpec("000001", "2022-01-01", "2025-12-31", pv=pv)
    return baseline_features(customer_daily(spec, 5))[0]


def test_dip_grows_with_capacity():
    assert _feats(10.0)["dip_kw"] > _feats(4.0)["dip_kw"]


def test_pv_lowers_summer_dip():
    assert _feats(6.0)["summer_dip"] < _feats(None)["summer_dip"]
