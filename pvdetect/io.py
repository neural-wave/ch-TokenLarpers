"""CSV loading, validation and the DST-safe UTC+1 day matrix."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Zurich"
REQUIRED = ("customer_id", "energy_consumption_kwh", "timestamp_local", "timestamp_epoch")
_USED = ("customer_id", "energy_consumption_kwh", "timestamp_epoch")
SLOT_S = 900
STD_OFFSET_S = 3600  # fixed UTC+1: PV follows solar time, DST days become 96 slots


@dataclass(frozen=True)
class LoadedTimeseries:
    """`frame` columns: customer_id (str), epoch_s (int64 UTC seconds floored to 15 min), energy_kwh."""

    frame: pd.DataFrame
    customer_ids: list[str]
    report: dict


def _sniff(path: str | Path) -> tuple[str, str, list[str]]:
    """Return (separator, decimal mark, stripped header names)."""
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline().rstrip("\r\n")
        first = f.readline().rstrip("\r\n")
    sep = ";" if ";" in header else ","
    cols = [c.strip().strip('"') for c in header.split(sep)]
    decimal = "."
    if sep == ";" and first and "energy_consumption_kwh" in cols:
        fields = first.split(sep)
        idx = cols.index("energy_consumption_kwh")
        if idx < len(fields) and "," in fields[idx]:
            decimal = ","
    return sep, decimal, cols


def _check_header(cols: list[str]) -> None:
    missing = set(REQUIRED) - set(cols)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")


def _read(path: str | Path, **kw) -> pd.DataFrame | pd.io.parsers.TextFileReader:
    sep, decimal, cols = _sniff(path)
    _check_header(cols)
    raw_names = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig").columns
    by_clean = {c.strip(): c for c in raw_names}
    usecols = [by_clean[c] for c in _USED]
    return pd.read_csv(
        path,
        sep=sep,
        decimal=decimal,
        usecols=usecols,
        dtype={by_clean["customer_id"]: str},
        encoding="utf-8-sig",
        **kw,
    )


def _parse_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw = raw.rename(columns=lambda c: c.strip())
    ids = raw["customer_id"].astype(str).str.strip()
    energy = raw["energy_consumption_kwh"]
    if not pd.api.types.is_numeric_dtype(energy):
        energy = energy.astype(str).str.strip().str.replace(",", ".", regex=False)
    energy = pd.to_numeric(energy, errors="coerce").astype("float64")
    epoch = pd.to_numeric(raw["timestamp_epoch"], errors="coerce")
    bad_epoch = epoch.isna().to_numpy()
    bad_energy = energy.isna().to_numpy() & ~bad_epoch
    keep = ~(bad_epoch | bad_energy)
    ep = epoch.to_numpy()[keep]
    if ep.size and np.median(ep) > 1e11:
        ep = ep / 1000.0
    ep = np.floor(ep).astype(np.int64)
    ep -= ep % SLOT_S
    frame = pd.DataFrame(
        {
            "customer_id": ids.to_numpy()[keep],
            "epoch_s": ep,
            "energy_kwh": energy.to_numpy()[keep],
        }
    )
    counts = {"rows_bad_epoch": int(bad_epoch.sum()), "rows_bad_energy": int(bad_energy.sum())}
    return frame, counts


def iter_chunks(path: str | Path, chunksize: int = 500_000) -> Iterator[tuple[pd.DataFrame, dict]]:
    """Typed `(frame, counts)` per chunk; same header check, sniffing and parsing as `load_timeseries`.

    Null customer ids become "" (not "nan") so streaming consumers can count and drop them."""
    with _read(path, chunksize=chunksize) as reader:
        for raw in reader:
            raw = raw.rename(columns=lambda c: c.strip())
            raw["customer_id"] = raw["customer_id"].fillna("")
            frame, counts = _parse_frame(raw)
            yield frame, {"rows_total": len(raw), **counts}



def local_dates(epoch_s: np.ndarray) -> np.ndarray:
    """Europe/Zurich calendar dates (datetime64[D]) for UTC epoch seconds."""
    ts = pd.to_datetime(epoch_s, unit="s", utc=True).tz_convert(TZ).tz_localize(None)
    return ts.to_numpy().astype("datetime64[D]")


def load_timeseries(path: str | Path) -> LoadedTimeseries:
    raw = _read(path).rename(columns=lambda c: c.strip())
    rows_total = len(raw)
    customer_ids = sorted(set(raw["customer_id"].astype(str).str.strip()))
    frame, counts = _parse_frame(raw)
    n_before = len(frame)
    frame = frame.drop_duplicates(["customer_id", "epoch_s"], keep="first")
    rows_duplicate = n_before - len(frame)
    frame = frame.sort_values(["customer_id", "epoch_s"], kind="stable").reset_index(drop=True)

    day_counts = {"92": 0, "96": 0, "100": 0, "other": 0}
    if len(frame):
        sizes = pd.DataFrame({"c": frame["customer_id"], "d": local_dates(frame["epoch_s"].to_numpy())}).groupby(
            ["c", "d"], sort=False
        ).size()
        vc = sizes.value_counts()
        for k, v in vc.items():
            key = str(k) if str(k) in day_counts else "other"
            day_counts[key] += int(v)
    neg_share = (frame["energy_kwh"] < 0).groupby(frame["customer_id"]).mean()
    report = {
        "rows_total": rows_total,
        **counts,
        "rows_duplicate": int(rows_duplicate),
        "customers": len(customer_ids),
        "day_length_counts": day_counts,
        "customers_signed": int((neg_share >= 0.01).sum()),
    }
    return LoadedTimeseries(frame=frame, customer_ids=customer_ids, report=report)


def _read_str_table(path: str | Path) -> tuple[pd.DataFrame, str]:
    sep, _, _ = _sniff(path)
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = df.rename(columns=lambda c: c.strip())
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df, sep


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce").astype("float64")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")


def parse_bool(s: pd.Series) -> pd.Series:
    up = s.astype(str).str.strip().str.upper()
    mapping = {"TRUE": True, "1": True, "FALSE": False, "0": False}
    bad = ~up.isin(list(mapping))
    if bad.any():
        raise ValueError(f"invalid pv_installed values: {sorted(set(up[bad]))[:5]}")
    return up.map(mapping).astype(bool)


def load_labels(path: str | Path) -> pd.DataFrame:
    df, _ = _read_str_table(path)
    out = pd.DataFrame({"customer_id": df["customer_id"]})
    out["pv_installed"] = parse_bool(df["pv_installed"])
    out["pv_installation_date"] = _to_date(df["pv_installation_date"])
    out["pv_capacity_kwp"] = _to_float(df["pv_capacity_kwp"])
    return out


def load_extension_labels(path: str | Path) -> pd.DataFrame:
    df, _ = _read_str_table(path)
    return pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "extension_date": _to_date(df["extension_date"]),
            "pv_capacity_after_extension_kwp": _to_float(df["pv_capacity_after_extension_kwp"]),
        }
    )


def day_matrix(frame_one: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(dates as datetime64[D], M of shape (n_days, 96)) in fixed UTC+1; NaN where missing."""
    std = frame_one["epoch_s"].to_numpy() + STD_OFFSET_S
    day = std // 86400
    slot = (std % 86400) // SLOT_S
    d0, d1 = int(day.min()), int(day.max())
    M = np.full((d1 - d0 + 1, 96), np.nan)
    M[day - d0, slot] = frame_one["energy_kwh"].to_numpy()
    dates = np.arange(d0, d1 + 1).astype("datetime64[D]")
    return dates, M
