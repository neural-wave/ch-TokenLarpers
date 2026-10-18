from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvdetect import extensions as X
from pvdetect.io import load_timeseries
from pvdetect.model import load_model, save_model
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.synthetic import CustomerSpec, Event, PVSpec, generate_customer, generate_extension_dataset
from tests.helpers import as_loaded


def _table(ts: pd.DataFrame):
    return build_feature_table(as_loaded(ts))


# ------------------------------------------------------------------------------------------- unit (exact)


def test_zero_run_counts_the_run_through_noon_only():
    M = np.full((3, 96), 0.3)
    M[0, 44:58] = 0.0  # 11:00–14:30 → 14 slots
    M[0, 30:32] = 0.0  # a separate morning dip is not part of the noon run
    M[1, 60:70] = 0.0  # zero, but not at 12:30
    M[2, X.NOON] = np.nan
    out = X.zero_run_hours(M)
    assert out[0] == 3.5 and out[1] == 0.0 and np.isnan(out[2])


def _noisy(levels: list[tuple[int, float]], n: int, sd: float = 0.04, seed: int = 0, swing: float = 0.0):
    """(k, doy): piecewise-constant log capacity over n weekly windows, plus a customer-specific seasonal
    swing of amplitude `swing` and ±sd noise."""
    doy = (np.arange(n) * 7 + 20) % 365 + 1.0
    k = np.zeros(n)
    for start, level in levels:
        k[start:] = level
    k = k + swing * np.cos(2 * np.pi * (doy - 40) / 365.25)
    return k + np.random.default_rng(seed).normal(0, sd, n), doy


def test_a_persistent_step_is_found_once_near_its_window_with_its_size():
    k, doy = _noisy([(0, np.log(6.0)), (90, np.log(9.0))], 170)
    steps = X.harmonic_steps(k, doy)
    assert len(steps) == 1
    tau, step, z = steps[0]
    assert abs(tau - 90) <= 3 and step == pytest.approx(np.log(1.5), abs=0.08) and z >= X.MIN_Z


def test_a_step_smaller_than_the_customers_seasonal_swing_is_still_found():
    """The failure the audit found: a ±0.18 seasonal nowcast swing hides a 0.25 step from a level test."""
    k, doy = _noisy([(0, np.log(6.0)), (100, np.log(6.0) + 0.25)], 190, sd=0.05, swing=0.18)
    (tau, step, _), = X.harmonic_steps(k, doy)
    assert abs(tau - 100) <= 3 and step == pytest.approx(0.25, abs=0.06)


def test_flat_noise_an_outage_and_a_seasonal_swing_are_not_extensions():
    flat, doy = _noisy([(0, 2.0)], 170)
    outage = flat.copy()
    outage[60:66] -= 1.5  # six windows near zero output, then back to the old level
    season, _ = _noisy([(0, 2.0)], 170, swing=0.25)  # nowcast bias repeating every year
    for k in (flat, outage, season):
        assert X.harmonic_steps(k, doy) == []


def test_two_changes_give_two_events_and_the_later_one_is_placed():
    """Greedy placement can put the first of two close steps between them (documented limitation); the second
    change is still located and both steps clear the ratio."""
    k, doy = _noisy([(0, np.log(5.0)), (70, np.log(8.0)), (130, np.log(12.0))], 190, swing=0.15)
    steps = X.harmonic_steps(k, doy)
    assert len(steps) == 2 and 70 <= steps[0][0] < 130 and abs(steps[1][0] - 130) <= 3
    assert all(s > np.log(X.MIN_RATIO) for _, s, _ in steps)


# --------------------------------------------------------------------- nowcaster on fabricated customers


@pytest.fixture(scope="module")
def nowcaster():
    ts, labels, ext, _ = generate_extension_dataset(45, seed=3)
    table = _table(ts)
    rows = []
    for r in labels[labels.pv_installed == "TRUE"].itertuples(index=False):
        regimes = [(pd.Timestamp(r.pv_installation_date), float(r.pv_capacity_kwp))]
        regimes += [(pd.Timestamp(e.extension_date), float(e.pv_capacity_after_extension_kwp))
                    for e in ext[ext.customer_id == r.customer_id].sort_values("extension_date").itertuples()]
        rows.append(X.regime_rows(r.customer_id, table.customers[r.customer_id], regimes))
    return X.fit_nowcaster(pd.concat(rows, ignore_index=True))


def _events(nowcaster, spec: CustomerSpec, seed: int):
    ts = generate_customer(spec, seed)
    cd = _table(ts).customers[spec.customer_id]
    return X.customer_events(nowcaster, spec.customer_id, cd, pd.Timestamp(spec.pv.date).date(), spec.pv.kwp)


def test_a_doubling_is_dated_within_15_days_and_sized_within_25_percent(nowcaster):
    pv = PVSpec("2022-05-10", 5.0, extensions=(("2023-08-20", 10.0),))
    ev = _events(nowcaster, CustomerSpec("000001", "2022-01-01", "2025-12-31", pv=pv, weather_seed=7), 41)
    assert len(ev) == 1
    assert abs((pd.Timestamp(ev[0].date) - pd.Timestamp("2023-08-20")).days) <= 15
    assert ev[0].kwp_after == pytest.approx(10.0, rel=0.25)


@pytest.mark.parametrize("seed", [21, 22, 23])
@pytest.mark.parametrize("event", [None, "vacation", "ac", "home_office"])
def test_no_extension_customers_give_no_event(nowcaster, seed, event):
    pv = PVSpec("2022-05-10", 6.0)
    events = () if event is None else (Event(event, "2023-06-01"),)
    spec = CustomerSpec("000002", "2022-01-01", "2025-12-31", pv=pv, events=events, weather_seed=7)
    assert _events(nowcaster, spec, seed) == []


# ----------------------------------------------------------------------- end to end through procedure.fit


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    from pvdetect.procedure import fit

    ts, labels, ext, _ = generate_extension_dataset(30, seed=5)
    table = _table(ts)
    lab = load_labels_frame(labels)
    on = fit(table, lab, lab["customer_id"].tolist(), inner=False, ext_labels=ext_frame(ext))
    off = fit(table, lab, lab["customer_id"].tolist(), inner=False, extensions=False)
    test_ts, *_ = generate_extension_dataset(15, seed=6)
    return on, off, _table(test_ts), tmp_path_factory.mktemp("ext_model")


def load_labels_frame(labels: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": labels.customer_id,
            "pv_installed": labels.pv_installed == "TRUE",
            "pv_installation_date": pd.to_datetime(labels.pv_installation_date, errors="coerce"),
            "pv_capacity_kwp": pd.to_numeric(labels.pv_capacity_kwp, errors="coerce"),
        }
    )


def ext_frame(ext: pd.DataFrame) -> pd.DataFrame:
    return ext.assign(
        extension_date=pd.to_datetime(ext.extension_date),
        pv_capacity_after_extension_kwp=ext.pv_capacity_after_extension_kwp.astype(float),
    )


def test_extensions_never_change_the_main_predictions_and_respect_the_output_contract(fitted):
    on, off, test, _ = fitted
    assert on["extensions"]["enabled"] and not off["extensions"]["enabled"]
    pred_on, ext_on = predict(on, test, test.ids)
    pred_off, ext_off = predict(off, test, test.ids)
    main = ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"]
    pd.testing.assert_frame_equal(pred_on[main], pred_off[main])
    assert ext_off.empty and list(ext_on.columns) == X.COLUMNS
    true = pred_on.set_index("customer_id")
    per_customer = ext_on.groupby("customer_id").size()
    assert len(ext_on) <= X.MAX_TOTAL and (per_customer.max() if len(per_customer) else 0) <= X.MAX_PER_CUSTOMER
    for r in ext_on.itertuples(index=False):
        assert true.at[r.customer_id, "pv_installed"]
        assert r.extension_date > true.at[r.customer_id, "pv_installation_date"]
        assert r.pv_capacity_after_extension_kwp > true.at[r.customer_id, "pv_capacity_kwp"]


def test_a_failing_detector_degrades_to_header_only_without_touching_predictions(fitted, capsys):
    on, off, test, _ = fitted

    class Broken:
        def predict(self, X):
            raise RuntimeError("boom")

    broken = {**on, "_objects": {**on["_objects"], "catboost_ext": Broken()}}
    pred_b, ext_b = predict(broken, test, test.ids)
    pred_off, _ = predict(off, test, test.ids)
    assert ext_b.empty and "header-only" in capsys.readouterr().err
    pd.testing.assert_frame_equal(pred_b, pred_off)


def test_saved_model_reproduces_the_extension_rows(fitted):
    on, _, test, out = fitted
    save_model(on, out / "model.json")
    assert (out / "catboost_ext.cbm").is_file()
    loaded = load_model(out / "model.json")
    pd.testing.assert_frame_equal(predict(loaded, test, test.ids)[1], predict(on, test, test.ids)[1])


def test_held_out_extension_labels_cannot_change_the_fit(fitted):
    """Leakage: extension labels of customers outside `train_ids` are ignored by the nowcaster."""
    from pvdetect.extensions import fit_extensions

    on, _, _, _ = fitted
    ts, labels, ext, _ = generate_extension_dataset(30, seed=5)
    table, lab = _table(ts), load_labels_frame(labels)
    train = lab.customer_id.tolist()[:20]
    held = ext_frame(ext)
    outsider = held[~held.customer_id.isin(train)]
    assert len(outsider)
    flipped = pd.concat([held[held.customer_id.isin(train)], outsider.assign(pv_capacity_after_extension_kwp=999.0)])
    rows = X.window_features(table.customers[train[0]]).dropna(subset=list(X.FEATURES))[list(X.FEATURES)].to_numpy(float)
    a = fit_extensions(table, lab, train, held, on)[1].predict(rows)
    b = fit_extensions(table, lab, train, flipped, on)[1].predict(rows)
    np.testing.assert_array_equal(a, b)


def test_an_early_predicted_install_date_does_not_turn_the_installation_into_an_extension(nowcaster):
    spec = CustomerSpec("000003", "2022-01-01", "2025-12-31", pv=PVSpec("2023-04-15", 6.0), weather_seed=7)
    cd = _table(generate_customer(spec, 31)).customers["000003"]
    early = pd.Timestamp("2022-12-01").date()  # main model dated the onset 4.5 months too early
    assert X.customer_events(nowcaster, "000003", cd, early, 6.0) == []


def test_unusable_capacity_labels_are_skipped_not_fatal():
    ts, labels, ext, _ = generate_extension_dataset(15, seed=8)
    table, lab = _table(ts), load_labels_frame(labels)
    pv = lab.index[lab.pv_installed]
    full, _ = X.fit_extensions(table, lab, lab.customer_id.tolist(), ext_frame(ext), {})
    lab.loc[pv[:2], "pv_capacity_kwp"] = [np.nan, 0.0]
    section, nowcaster = X.fit_extensions(table, lab, lab.customer_id.tolist(), ext_frame(ext), {})
    # the two initial regimes drop out (their extension regimes still count); no NaN target reaches CatBoost
    assert section["enabled"] and section["windows"] < full["windows"]
