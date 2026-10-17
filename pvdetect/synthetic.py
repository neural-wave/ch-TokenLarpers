"""Synthetic smart-meter generator for tests, benchmarks and smoke runs.

Never imported by `train.py`, `inference.py` or the pipeline. Variant A is the
development generator; variant B uses independent PV, cloud and load model
families plus confounder kinds absent from A (holdout benchmark only).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

TZ = "Europe/Zurich"

A_KINDS = ("ev", "heat_pump", "boiler_night", "vacation", "home_office", "vacancy", "ac")
B_KINDS = A_KINDS + ("daytime_absence", "heat_storage", "pool_pump", "ev_daytime")

SCENARIOS = (
    "base",
    "confounders_x2",
    "east_west_heavy",
    "battery_heavy",
    "short_coverage",
    "missing_5pct",
    "commercial_heavy",
    "pv_before_start",
    "label_lag_60",
    "meter_spikes",
    "shared_weather",
    "late_install",
    "small_pv",
)
# Development-only scenarios (variant A): `label_lag_60` labels each PV customer 30–90 days after its
# visible onset; `meter_spikes` adds a few isolated single-slot spikes of 20–50× the median slot value
# per customer-year. Both reuse the `base` customers, so the base series and labels stay as they are.
# `shared_weather` reuses the base customers too, but every customer sees one dataset-wide daily sunshine
# series (PV output and AC load follow it) and the non-PV hard negatives become sun-driven AC or a
# recurring summer weekday daytime absence, i.e. load patterns that mimic a PV midday dip or peak.
# `late_install` moves every PV installation into the last 8–16 weeks of its coverage (little post-install data);
# `small_pv` draws residential PV of 0.8–2.0 kWp (a shallow midday dip). Both use the `aux` stream, so the other
# draws of the base customers stay as they are.
A_ONLY_SCENARIOS = ("label_lag_60", "meter_spikes", "shared_weather", "late_install", "small_pv")
SHARED_WEATHER_KINDS = ("ac", "summer_absence")
SPIKES_PER_YEAR = 3
WEATHER_EPOCH = np.datetime64("2015-01-01", "D")
WEATHER_DAYS = 6000  # 2015-01-01 .. 2031-06


@dataclass(frozen=True)
class PVSpec:
    date: str
    kwp: float
    orientation: Literal["south", "east_west"] = "south"
    export: bool = False
    battery_kwh: float = 0.0
    extensions: tuple[tuple[str, float], ...] = ()  # (local date, total kWp after)
    label_lag_days: int = 0  # labelled date − visible onset (`label_lag_60`)


@dataclass(frozen=True)
class Event:
    kind: str
    start: str
    end: str | None = None
    size: float = 1.0


@dataclass(frozen=True)
class CustomerSpec:
    customer_id: str
    start: str
    end: str
    profile: Literal["residential", "commercial"] = "residential"
    pv: PVSpec | None = None
    events: tuple[Event, ...] = ()
    missing: bool = True
    missing_rate: float = 0.005
    spikes_per_year: int = 0  # isolated single-slot meter spikes (`meter_spikes`)
    weather_seed: int | None = None  # dataset-wide daily sunshine series (`shared_weather`)


@dataclass
class _Grid:
    """Per-row calendar arrays shared by the load, event and PV models."""

    epoch: np.ndarray  # int64 seconds, interval start
    local_str: np.ndarray  # object array of wall-clock strings
    local_day: np.ndarray  # datetime64[D] local date
    h: np.ndarray  # local fractional hour
    hs: np.ndarray  # UTC+1 fractional hour
    std_day: np.ndarray  # int day index in UTC+1 (0 = first day)
    doy: np.ndarray  # local day of year
    weekday: np.ndarray  # local weekday 0=Mon
    n_days: int
    extra: dict = field(default_factory=dict)


def _grid(start: str, end: str) -> _Grid:
    idx = pd.date_range(
        pd.Timestamp(start, tz=TZ),
        pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1),
        freq="15min",
        inclusive="left",
    )
    epoch = (idx.as_unit("ns").asi8 // 10**9).astype(np.int64)
    local = idx.tz_localize(None)
    local_day = local.values.astype("datetime64[D]")
    h = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    std = epoch + 3600
    std_day_abs = std // 86400
    std_day = (std_day_abs - std_day_abs[0]).astype(np.int64)
    hs = (std % 86400) / 3600.0
    return _Grid(
        epoch=epoch,
        local_str=np.asarray(local.strftime("%Y-%m-%d %H:%M:%S")),
        local_day=local_day,
        h=h,
        hs=hs,
        std_day=std_day,
        doy=local.dayofyear.to_numpy(),
        weekday=local.dayofweek.to_numpy(),
        n_days=int(std_day[-1]) + 1,
    )


def _group_cumsum(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cumulative sum restarting at each change of the (sorted) group key."""
    c = np.cumsum(x)
    starts = np.r_[0, np.flatnonzero(np.diff(groups)) + 1]
    base = np.r_[0.0, c[starts[1:] - 1]]
    lengths = np.diff(np.r_[starts, x.size])
    return c - np.repeat(base, lengths)


def _seasonal(doy: np.ndarray, peak: float) -> np.ndarray:
    return np.cos(2 * np.pi * (doy - peak) / 365.0)


# ---------------------------------------------------------------- load models


def _load_a(g: _Grid, profile: str, rng: np.random.Generator) -> np.ndarray:
    h, n = g.h, g.h.size
    weekend = g.weekday >= 5
    if profile == "commercial":
        k = rng.uniform(2.0, 8.0)
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
        load = np.where(
            weekend,
            k * 0.12,
            k * (0.1 + 0.5 * sig((h - 7) * 3) * sig((18 - h) * 3)),
        )
    else:
        k = rng.uniform(0.6, 1.6)
        b = rng.uniform(0.05, 0.15)
        load = k * (
            b
            + 0.25 * np.exp(-(((h - 7.5) / 1.0) ** 2))
            + 0.4 * np.exp(-(((h - 19) / 1.5) ** 2))
            + 0.10 * np.exp(-(((h - 12.5) / 1.5) ** 2))
        )
        load = load * np.where(weekend, 1.15, 1.0)
    load = load * (1 + 0.25 * _seasonal(g.doy, 15))
    return load * rng.lognormal(0.0, 0.25, n)


def _load_b(g: _Grid, profile: str, rng: np.random.Generator) -> np.ndarray:
    n = g.h.size
    weekend = g.weekday >= 5

    def archetype(lo: float, hi: float, day_hours: bool) -> np.ndarray:
        nb = int(rng.integers(4, 7))
        if day_hours:
            xs = np.sort(rng.uniform(5, 21, nb))
        else:
            xs = np.sort(rng.uniform(0, 24, nb))
        ys = rng.uniform(lo, hi, nb)
        # circular interpolation over 24 h
        xp = np.r_[xs[-1] - 24, xs, xs[0] + 24]
        fp = np.r_[ys[-1], ys, ys[0]]
        return np.interp(g.h, xp, fp)

    if profile == "commercial":
        k = rng.uniform(2.0, 8.0)
        base = k * archetype(0.05, 0.6, True)
        load = np.where(weekend, base * rng.uniform(0.1, 0.4), base)
    else:
        k = rng.uniform(0.6, 1.6)
        load = k * archetype(0.03, 0.5, False)
        load = load * np.where(weekend, rng.uniform(0.9, 1.3), 1.0)
    a = rng.uniform(0.05, 0.35)
    phi = rng.uniform(0, 60)
    load = load * (1 + a * _seasonal(g.doy, phi)) * rng.lognormal(0.0, 0.2, n)
    # Poisson appliance spikes: rate 3/day, 1-2 kW, 15-45 min.
    n_spikes = rng.poisson(3.0 * g.n_days)
    if n_spikes:
        starts = rng.integers(0, n, n_spikes)
        lens = rng.integers(1, 4, n_spikes)
        kw = rng.uniform(1.0, 2.0, n_spikes)
        add = np.zeros(n)
        for off in range(3):
            sel = lens > off
            np.add.at(add, np.minimum(starts[sel] + off, n - 1), kw[sel] * 0.25)
        load = load + add
    return load


# --------------------------------------------------------------------- events


def _active(g: _Grid, ev: Event) -> np.ndarray:
    start = np.datetime64(ev.start, "D")
    if ev.end is not None:
        end = np.datetime64(ev.end, "D")
    elif ev.kind == "vacation":
        end = start + np.timedelta64(14, "D")
    else:
        end = np.datetime64("9999-12-31", "D")
    return (g.local_day >= start) & (g.local_day < end)


def _apply_event(
    g: _Grid, load: np.ndarray, ev: Event, rng: np.random.Generator, sun: np.ndarray | None = None
) -> np.ndarray:
    act = _active(g, ev)
    h, doy, wd = g.h, g.doy, g.weekday
    out = load.copy()
    day_codes = (g.local_day - g.local_day[0]).astype(np.int64)
    n_local_days = int(day_codes[-1]) + 2
    if ev.kind == "ev":
        # night anchored on the evening date: 22:00 -> hh = 0
        anchor = np.where(h >= 12, day_codes, day_codes - 1)
        hh = np.where(h >= 12, h - 22, h + 2)
        night_on = rng.random(n_local_days + 1) < 4 / 7
        start_off = rng.uniform(0, 3, n_local_days + 1)
        a = np.clip(anchor, 0, None)
        on = night_on[a] & (hh >= start_off[a]) & (hh < start_off[a] + 3) & (anchor >= 0)
        out = out + np.where(act & on, ev.size * 0.25 * 3.7, 0.0)
    elif ev.kind == "heat_pump":
        peak = ((h >= 6) & (h < 9)) | ((h >= 17) & (h < 21))
        add = ev.size * 0.25 * np.maximum(0, _seasonal(doy, 15)) * (1 + 0.3 * peak)
        out = out + np.where(act, add, 0.0)
    elif ev.kind == "boiler_night":
        src = act & (h >= 11) & (h < 14)
        target = out - 2.5 * ev.size / 12
        new = np.where(src, np.maximum(target, 0.02), out)
        removed = np.where(src, out - new, 0.0)
        out = new
        per_day = np.bincount(day_codes, weights=removed, minlength=n_local_days)
        late = h >= 23
        early = h < 2
        add = np.zeros_like(out)
        add[late] = per_day[day_codes[late]] / 12
        prev = np.clip(day_codes[early] - 1, 0, None)
        add[early] = np.where(day_codes[early] >= 1, per_day[prev] / 12, 0.0)
        out = out + add
    elif ev.kind == "vacation":
        out = np.where(act, out * 0.3, out)
    elif ev.kind == "home_office":
        sel = act & (wd < 5) & (h >= 9) & (h < 17)
        out = out + np.where(sel, 0.05 * ev.size, 0.0)
    elif ev.kind == "vacancy":
        out = np.where(act, out * 0.4, out)
    elif ev.kind == "ac":
        sel = act & (doy >= 152) & (doy <= 243) & (h >= 12) & (h < 18)
        hot = 1.0 if sun is None else sun / 0.6  # shared weather: more cooling on sunny days
        out = out + np.where(sel, ev.size * 0.25 * np.maximum(0, _seasonal(doy, 200)) * hot, 0.0)
    elif ev.kind == "summer_absence":  # recurring every year: mid-June to August weekdays, 08–17 h
        sel = act & (doy >= 166) & (doy <= 243) & (wd < 5) & (h >= 8) & (h < 17)
        out = np.where(sel, out * 0.5, out)
    elif ev.kind == "daytime_absence":
        sel = act & (wd < 5) & (h >= 8) & (h < 17)
        out = np.where(sel, out * 0.5, out)
    elif ev.kind == "heat_storage":
        energy = rng.uniform(6.0, 8.0) * ev.size
        sel = act & ((h >= 22) | (h < 6))
        out = out + np.where(sel, energy / 32 * np.maximum(0, _seasonal(doy, 15)), 0.0)
    elif ev.kind == "pool_pump":
        sel = act & (doy >= 135) & (doy <= 260) & (h >= 10) & (h < 16)
        out = out + np.where(sel, 0.25 * ev.size, 0.0)
    elif ev.kind == "ev_daytime":
        start_off = rng.uniform(11, 13, n_local_days + 1)
        so = start_off[day_codes]
        sel = act & (wd >= 5) & (h >= so) & (h < so + 2)
        out = out + np.where(sel, 7.0 * 0.25 * ev.size, 0.0)
    else:
        raise ValueError(f"unknown event kind: {ev.kind}")
    return out


# ------------------------------------------------------------------- PV models


def _kwp_series(g: _Grid, pv: PVSpec) -> np.ndarray:
    kwp = np.where(g.local_day >= np.datetime64(pv.date, "D"), pv.kwp, 0.0)
    for date, total in sorted(pv.extensions):
        kwp = np.where(g.local_day >= np.datetime64(date, "D"), total, kwp)
    return kwp


def _shared_sun(weather_seed: int, g: _Grid) -> np.ndarray:
    """Per-row daily sunshine fraction in [0.2, 1], identical for every customer on the same absolute
    UTC+1 day: one AR(1) series drawn from `weather_seed` alone."""
    rng = np.random.default_rng([weather_seed, 7])
    eps = rng.normal(0.0, 0.25, WEATHER_DAYS)
    k = np.empty(WEATHER_DAYS)
    k[0] = 0.6
    for d in range(1, WEATHER_DAYS):
        k[d] = 0.6 + 0.5 * (k[d - 1] - 0.6) + eps[d]
    first = (WEATHER_EPOCH - np.datetime64("1970-01-01", "D")).astype(np.int64)
    return np.clip(k, 0.2, 1.0)[(g.epoch + 3600) // 86400 - first]


def _pv_a(g: _Grid, pv: PVSpec, rng: np.random.Generator, sun: np.ndarray | None = None) -> np.ndarray:
    c = _seasonal(g.doy, 172)
    hw = 4.0 + 2.5 * c
    amp = 0.55 + 0.25 * c
    if pv.orientation == "east_west":
        hw = hw * 1.25
        amp = amp * 0.85
    x = (g.hs - 12.5) / hw
    shape = np.where(np.abs(x) < 1, np.cos(np.pi / 2 * np.clip(x, -1, 1)) ** 1.5, 0.0)
    cloud = rng.uniform(0.2, 1.0, g.n_days)[g.std_day]  # drawn in every case: keeps the base stream
    if sun is not None:
        cloud = sun
    return _kwp_series(g, pv) * 0.25 * amp * shape * cloud


def _pv_b(g: _Grid, pv: PVSpec, rng: np.random.Generator) -> np.ndarray:
    c = _seasonal(g.doy, 172)
    day_len = 12 + 3.9 * c
    sunrise = 12.5 - day_len / 2
    phase = (g.hs - sunrise) / day_len
    inside = (phase > 0) & (phase < 1)
    s = np.sin(np.pi * np.clip(phase, 0, 1))
    amp = 0.5 + 0.3 * c
    if pv.orientation == "east_west":
        shape = np.where(inside, np.clip(1.6 * s, 0, 1), 0.0)
        amp = amp * 0.8
    else:
        shape = np.where(inside, s**1.2, 0.0)
    k = np.empty(g.n_days)
    k[0] = 0.6
    eps = rng.normal(0, 0.2, g.n_days)
    for d in range(1, g.n_days):
        k[d] = min(max(0.6 + 0.8 * (k[d - 1] - 0.6) + eps[d], 0.05), 1.0)
    hour_key = g.std_day * 24 + np.floor(g.hs).astype(np.int64)
    hour_noise = rng.lognormal(0, 0.15, g.n_days * 24 + 1)[hour_key]
    return _kwp_series(g, pv) * 0.25 * amp * shape * k[g.std_day] * hour_noise


def _battery(g: _Grid, net: np.ndarray, cap: float) -> np.ndarray:
    surplus = np.maximum(-net, 0.0)
    cum = _group_cumsum(surplus, g.std_day)
    absorbed_cum = np.minimum(cum, cap)
    absorbed = np.diff(np.r_[0.0, absorbed_cum])
    first = np.r_[True, np.diff(g.std_day) != 0]
    absorbed[first] = absorbed_cum[first]
    net = net + absorbed
    stored = np.bincount(g.std_day, weights=absorbed, minlength=g.n_days)
    window = (g.hs >= 17) & (g.hs < 23)
    demand = np.where(window, np.maximum(net, 0.0), 0.0)
    dcum = _group_cumsum(demand, g.std_day)
    delivered_cum = np.minimum(dcum, stored[g.std_day])
    delivered = np.diff(np.r_[0.0, delivered_cum])
    delivered[first] = delivered_cum[first]
    return net - delivered


# ------------------------------------------------------------------ customers


def generate_customer(spec: CustomerSpec, seed: int, variant: Literal["A", "B"] = "A") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = _grid(spec.start, spec.end)
    sun = None
    if spec.weather_seed is not None:  # shared sunshine with a little local (per-customer) variation
        local = np.random.default_rng([seed, 3]).lognormal(0.0, 0.1, g.n_days)[g.std_day]
        sun = np.clip(_shared_sun(spec.weather_seed, g) * local, 0.1, 1.0)
    load = (_load_a if variant == "A" else _load_b)(g, spec.profile, rng)
    for ev in spec.events:
        load = _apply_event(g, load, ev, rng, sun)
    net = load
    if spec.pv is not None:
        prod = _pv_a(g, spec.pv, rng, sun) if variant == "A" else _pv_b(g, spec.pv, rng)
        net = load - prod
        if spec.pv.battery_kwh > 0:
            net = _battery(g, net, spec.pv.battery_kwh)
        if not spec.pv.export:
            net = np.maximum(net, 0.0)
    if spec.spikes_per_year > 0:
        net = _spikes(g, net, spec.spikes_per_year, np.random.default_rng([seed, 2]))
    df = pd.DataFrame(
        {
            "customer_id": spec.customer_id,
            "timestamp_local": g.local_str,
            "timestamp_epoch": g.epoch,
            "energy_consumption_kwh": np.round(net, 4),
        }
    )
    if spec.missing:
        n = len(df)
        keep = rng.random(n) >= spec.missing_rate
        block_day = int(rng.integers(0, max(1, g.n_days - 5)))
        keep &= ~((g.std_day >= block_day) & (g.std_day < block_day + 5))
        kept = df[keep]
        dup_row = int(rng.integers(0, len(kept)))
        df = pd.concat([kept, kept.iloc[[dup_row]]], ignore_index=True)
    return df


def _spikes(g: _Grid, net: np.ndarray, per_year: int, rng: np.random.Generator) -> np.ndarray:
    """`per_year` single-slot spikes per customer-year at 20–50× the median positive slot value."""
    pos = net[net > 0]
    med = float(np.median(pos)) if pos.size else 0.1
    k = max(1, int(round(per_year * g.n_days / 365)))
    idx = rng.choice(net.size, size=k, replace=False)
    out = net.copy()
    out[idx] = np.round(rng.uniform(20, 50, k) * med, 4)
    return out


# -------------------------------------------------------------------- datasets


def _day(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def _dataset_specs(
    n: int,
    seed: int,
    start: str,
    end: str,
    hard_negatives: bool,
    with_extensions: bool,
    variant: str,
    scenario: str,
) -> list[CustomerSpec]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if scenario in A_ONLY_SCENARIOS and variant != "A":
        raise ValueError(f"scenario {scenario} is variant A only")
    rng = np.random.default_rng(seed)
    aux = np.random.default_rng([seed, 1])  # draws of the A-only scenarios; `rng` stays the base stream
    kinds = A_KINDS if variant == "A" else B_KINDS
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    specs = []
    for i in range(n):
        if scenario == "short_coverage":
            length = int(rng.integers(300, 541))
            cstart = t0 + pd.Timedelta(days=int(rng.integers(0, (t1 - t0).days - length)))
            cend = cstart + pd.Timedelta(days=length)
        else:
            cstart = t0 + pd.Timedelta(days=int(rng.integers(0, 541)))
            cend = t1 - pd.Timedelta(days=int(rng.integers(0, 541)))
            if (cend - cstart).days < 730:
                cend = min(t1, cstart + pd.Timedelta(days=730))
                cstart = max(t0, cend - pd.Timedelta(days=730))
        commercial = (i // 2) % 2 == 0 if scenario == "commercial_heavy" else i % 7 == 3
        profile = "commercial" if commercial else "residential"
        missing_rate = 0.05 if scenario == "missing_5pct" else 0.005
        lo = cstart + pd.Timedelta(days=120)
        hi = cend - pd.Timedelta(days=150)

        def draw_date() -> pd.Timestamp:
            return lo + pd.Timedelta(days=int(rng.integers(0, max(1, (hi - lo).days))))

        pv = None
        events: list[Event] = []
        if i % 2 == 0:
            before_p = 0.40 if scenario == "pv_before_start" else 0.15
            if rng.random() < before_p:
                pv_date = cstart - pd.Timedelta(days=int(rng.integers(30, 401)))
            else:
                pv_date = draw_date()
            kwp = round(float(rng.uniform(15, 60) if commercial else rng.uniform(3, 15)), 1)
            if scenario == "late_install":
                pv_date = cend - pd.Timedelta(days=int(aux.integers(56, 113)))
            if scenario == "small_pv" and not commercial:
                kwp = round(float(aux.uniform(0.8, 2.0)), 1)
            if scenario == "east_west_heavy":
                ew = (i // 2) % 2 == 0
            else:
                ew = i % 6 == 0
            battery = 8.0 if (((i // 2) % 2 == 1) if scenario == "battery_heavy" else i % 10 == 4) else 0.0
            exts: tuple[tuple[str, float], ...] = ()
            if with_extensions and i % 12 == 6:
                ext_date = pv_date + pd.Timedelta(days=int(rng.integers(200, 501)))
                if cstart <= ext_date < cend:
                    exts = ((_day(ext_date), round(kwp * 1.5, 1)),)
            pv = PVSpec(
                date=_day(pv_date),
                kwp=kwp,
                orientation="east_west" if ew else "south",
                export=i % 8 == 0,
                battery_kwh=battery,
                extensions=exts,
                label_lag_days=int(aux.integers(30, 91)) if scenario == "label_lag_60" else 0,
            )
            if hard_negatives and i % 6 == 2:
                events.append(Event(kinds[(i // 2) % len(kinds)], _day(draw_date())))
        elif hard_negatives:
            kind = kinds[(i // 2) % len(kinds)]
            if scenario == "shared_weather":
                kind = SHARED_WEATHER_KINDS[(i // 2) % 2]
            events.append(Event(kind, _day(draw_date())))
            if scenario == "confounders_x2":
                events.append(Event(kinds[(i // 2 + 3) % len(kinds)], _day(draw_date())))
        specs.append(
            CustomerSpec(
                customer_id=f"{i:06d}",
                start=_day(cstart),
                end=_day(cend),
                profile=profile,
                pv=pv,
                events=tuple(events),
                missing_rate=missing_rate,
                spikes_per_year=SPIKES_PER_YEAR if scenario == "meter_spikes" else 0,
                weather_seed=seed if scenario == "shared_weather" else None,
            )
        )
    return specs


def generate_dataset(
    n: int,
    seed: int,
    start: str = "2022-01-01",
    end: str = "2025-12-31",
    hard_negatives: bool = True,
    with_extensions: bool = False,
    variant: Literal["A", "B"] = "A",
    scenario: str = "base",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (timeseries, labels, extension labels) with string label columns."""
    specs = _dataset_specs(n, seed, start, end, hard_negatives, with_extensions, variant, scenario)
    frames, labels, exts = [], [], []
    for i, spec in enumerate(specs):
        frames.append(generate_customer(spec, seed * 1000 + i, variant))
        pv = spec.pv
        labels.append(
            {
                "customer_id": spec.customer_id,
                "pv_installed": "TRUE" if pv else "FALSE",
                "pv_installation_date": _day(pd.Timestamp(pv.date) + pd.Timedelta(days=pv.label_lag_days)) if pv else "",
                "pv_capacity_kwp": f"{pv.kwp}" if pv else "",
            }
        )
        if pv:
            for date, total in pv.extensions:
                exts.append(
                    {
                        "customer_id": spec.customer_id,
                        "extension_date": date,
                        "pv_capacity_after_extension_kwp": f"{total}",
                    }
                )
    ext_cols = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(labels, columns=["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"]),
        pd.DataFrame(exts, columns=ext_cols),
    )


def write_dataset(
    out_dir: Path,
    ts: pd.DataFrame,
    labels: pd.DataFrame,
    ext: pd.DataFrame,
    sep: str = ",",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts.to_csv(out_dir / "energy.csv", sep=sep, index=False, float_format="%.4f")
    labels.to_csv(out_dir / "labels.csv", sep=sep, index=False)
    ext.to_csv(out_dir / "extensions.csv", sep=sep, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write a synthetic (variant A) dataset.")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--no-hard-negatives", action="store_true")
    ap.add_argument("--with-extensions", action="store_true")
    args = ap.parse_args(argv)
    ts, labels, ext = generate_dataset(
        args.n,
        args.seed,
        hard_negatives=not args.no_hard_negatives,
        with_extensions=args.with_extensions,
    )
    write_dataset(args.out_dir, ts, labels, ext, sep=args.sep)
    print(f"wrote {len(ts)} rows, {len(labels)} customers to {args.out_dir}")
    return 0


__all__ = [
    "A_KINDS",
    "B_KINDS",
    "SCENARIOS",
    "A_ONLY_SCENARIOS",
    "PVSpec",
    "Event",
    "CustomerSpec",
    "generate_customer",
    "generate_dataset",
    "write_dataset",
]

if __name__ == "__main__":
    raise SystemExit(main())
