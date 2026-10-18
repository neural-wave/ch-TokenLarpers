"""Memory-bounded streaming of the unlabelled series (Step 17), the label-free unlabelled prior (Step 18)
and the bounded pseudo-labelling pool.

Nothing here sees labels. The prior and the pool are computed once, outside `fit`, and attached to the
`FeatureTable`; `Procedure.fit` decides by inner CV whether any of its options is used.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pvdetect.daily import daily_from_sums
from pvdetect.features import _contrast, last365
from pvdetect.io import SLOT_S, STD_OFFSET_S, iter_chunks
from pvdetect.pipeline import CustomerData, customer_data
from pvdetect.season import N_WEEKS, SIGNALS, signal_curves, weekly_median

QUANTILES = (0.0, 0.25, 0.5, 0.75)
SENSITIVITY_MAX = 0.05  # log units, max over q, q', signal and week
MIN_PRIOR_CUSTOMERS = 30
MIN_SPREAD_DAYS = 56
_DAY_PAD = 64


class DailyAccumulator:
    """Streams typed chunks into dense per-(customer, day) block sums, then builds daily frames.

    Rows are dropped in this order (each counted once): null/empty id, labelled id (`exclude_ids`),
    id not starting with `id_prefix` (`""` disables the prefix filter). Duplicates of an already seen
    (customer, 15-min slot) are dropped keep-first across the whole file, like `load_timeseries`."""

    def __init__(self, id_prefix: str = "POD", exclude_ids: Iterable[str] = frozenset()):
        self.id_prefix = id_prefix
        self.exclude_ids = set(exclude_ids)
        self.codes: dict[str, int] = {}
        self.day_lo = 0
        self._alloc(0, 0)
        self.report = {
            "rows_total": 0,
            "rows_bad_epoch": 0,
            "rows_bad_energy": 0,
            "rows_excluded_null": 0,
            "rows_excluded_prefix": 0,
            "rows_excluded_labelled": 0,
            "rows_duplicate": 0,
            "rows_used": 0,
        }

    def _alloc(self, n_codes: int, n_days: int, old: dict | None = None, shift: int = 0) -> None:
        shapes = {
            "mid_sum": np.float64,
            "night_sum": np.float64,
            "eve_sum": np.float64,
            "tot_sum": np.float64,
            "mid_n": np.uint8,
            "night_n": np.uint8,
            "eve_n": np.uint8,
            "tot_n": np.uint8,
            "mid_zero": np.uint8,
        }
        new = {k: np.zeros((n_codes, n_days), dt) for k, dt in shapes.items()}
        new["seen"] = np.zeros((n_codes, n_days, 12), np.uint8)
        if old is not None:
            for k, a in old.items():
                new[k][: a.shape[0], shift : shift + a.shape[1]] = a
        self.state = new

    def _grow(self, n_codes: int, lo: int, hi: int) -> None:
        """Ensure capacity for `n_codes` customers and absolute days [lo, hi]."""
        cap_c, cap_d = self.state["tot_n"].shape
        if cap_d == 0:
            self.day_lo = lo - _DAY_PAD
            self._alloc(max(n_codes, 64), hi - lo + 1 + 2 * _DAY_PAD)
            return
        cur_hi = self.day_lo + cap_d - 1
        new_lo = lo - _DAY_PAD if lo < self.day_lo else self.day_lo
        new_hi = hi + _DAY_PAD if hi > cur_hi else cur_hi
        new_c = max(n_codes, 2 * cap_c) if n_codes > cap_c else cap_c
        if (new_lo, new_hi, new_c) == (self.day_lo, cur_hi, cap_c):
            return
        old, shift = self.state, self.day_lo - new_lo
        self.day_lo = new_lo
        self._alloc(new_c, new_hi - new_lo + 1, old, shift)

    def add(self, frame: pd.DataFrame) -> None:
        rep = self.report
        ids = frame["customer_id"].to_numpy(dtype=object)
        null = (ids == "") | pd.isna(ids)
        rep["rows_excluded_null"] += int(null.sum())
        keep = ~null
        if self.exclude_ids:
            lab = keep & pd.Series(ids).isin(self.exclude_ids).to_numpy()
            rep["rows_excluded_labelled"] += int(lab.sum())
            keep &= ~lab
        if self.id_prefix:
            pre = keep & ~pd.Series(ids).astype(str).str.startswith(self.id_prefix).to_numpy()
            rep["rows_excluded_prefix"] += int(pre.sum())
            keep &= ~pre
        if not keep.any():
            return
        ids = ids[keep]
        std = frame["epoch_s"].to_numpy(np.int64)[keep] + STD_OFFSET_S
        energy = frame["energy_kwh"].to_numpy(np.float64)[keep]
        local_codes, uniques = pd.factorize(ids)
        for u in uniques:
            if u not in self.codes:
                self.codes[u] = len(self.codes)
        lut = np.fromiter((self.codes[u] for u in uniques), np.int64, len(uniques))
        code = lut[local_codes]
        day = std // 86400
        slot = (std % 86400) // SLOT_S
        self._grow(len(self.codes), int(day.min()), int(day.max()))
        d = day - self.day_lo
        n_days = self.state["tot_n"].shape[1]

        # keep-first within the chunk, then against everything seen before
        key = (code * n_days + d) * 96 + slot
        _, first = np.unique(key, return_index=True)
        first.sort()
        seen = self.state["seen"]
        byte, bit = slot[first] >> 3, (1 << (slot[first] & 7)).astype(np.uint8)
        new = (seen[code[first], d[first], byte] & bit) == 0
        idx = first[new]
        rep["rows_duplicate"] += int(code.size - idx.size)
        rep["rows_used"] += int(idx.size)
        c, dd, sl, e = code[idx], d[idx], slot[idx], energy[idx]
        np.bitwise_or.at(seen, (c, dd, sl >> 3), (1 << (sl & 7)).astype(np.uint8))

        st = self.state
        cell = c * n_days + dd
        for name, mask in (
            ("mid", (sl >= 40) & (sl < 60)),
            ("night", sl < 20),
            ("eve", (sl >= 76) & (sl < 88)),
            ("tot", np.ones(sl.size, bool)),
        ):
            np.add.at(st[f"{name}_sum"].reshape(-1), cell[mask], e[mask])
            np.add.at(st[f"{name}_n"].reshape(-1), cell[mask], 1)
            if name == "mid":
                z = mask & (e <= 0.005)
                np.add.at(st["mid_zero"].reshape(-1), cell[z], 1)

    def add_counts(self, counts: dict) -> None:
        for k in ("rows_total", "rows_bad_epoch", "rows_bad_energy"):
            self.report[k] += int(counts.get(k, 0))

    def finalize(self) -> dict[str, pd.DataFrame]:
        """Daily frames over each customer's own contiguous day range (empty days as NaN rows)."""
        st = self.state
        out: dict[str, pd.DataFrame] = {}
        for cid, c in self.codes.items():
            has = np.flatnonzero(st["tot_n"][c] > 0)
            if has.size == 0:
                continue
            a, b = int(has[0]), int(has[-1]) + 1
            dates = np.arange(self.day_lo + a, self.day_lo + b).astype("datetime64[D]")
            row = {k: st[k][c, a:b] for k in st if k != "seen"}
            out[cid] = daily_from_sums(
                dates,
                row["mid_sum"],
                row["mid_n"],
                row["night_sum"],
                row["night_n"],
                row["eve_sum"],
                row["eve_n"],
                row["tot_sum"],
                row["tot_n"],
                row["mid_zero"],
            )
        self.report["customers"] = len(out)
        return out


def accumulate_file(
    path: str | Path,
    chunksize: int = 500_000,
    id_prefix: str = "POD",
    exclude_ids: Iterable[str] = frozenset(),
) -> tuple[dict[str, pd.DataFrame], dict]:
    acc = DailyAccumulator(id_prefix=id_prefix, exclude_ids=exclude_ids)
    for frame, counts in iter_chunks(path, chunksize=chunksize):
        acc.add_counts(counts)
        acc.add(frame)
    dailies = acc.finalize()
    return dailies, acc.report


# --------------------------------------------------------------------------- pseudo-labelling pool

POOL_SALT = b"pvdetect-pseudo-pool-1"
POOL_MIN_VALID_DAYS = 365


def _pool_rank(cid: str) -> str:
    return hashlib.sha256(POOL_SALT + cid.encode()).hexdigest()


def select_pool(n_valid: Mapping[str, int], size: int) -> list[str]:
    """At most `size` customers with ≥365 valid days, chosen by a salted hash of the id (independent of
    file order), returned sorted by id."""
    eligible = [c for c, n in n_valid.items() if n >= POOL_MIN_VALID_DAYS]
    return sorted(sorted(eligible, key=lambda c: (_pool_rank(c), c))[: max(size, 0)])


class PoolAccumulator:
    """Second streaming pass: the float64 UTC+1 day matrix of a fixed set of customers over the day
    range found by the first pass, keep-first per (customer, slot) across the file like `load_timeseries`."""

    def __init__(self, spans: Mapping[str, tuple[int, int]]):
        self.ids = sorted(spans)
        self.index = pd.Index(self.ids)
        self.first_day = np.array([spans[c][0] for c in self.ids], np.int64)
        self.M = [np.full((spans[c][1], 96), np.nan) for c in self.ids]

    def add(self, frame: pd.DataFrame) -> None:
        code = self.index.get_indexer(frame["customer_id"].to_numpy(dtype=object))
        keep = code >= 0
        if not keep.any():
            return
        code = code[keep]
        std = frame["epoch_s"].to_numpy(np.int64)[keep] + STD_OFFSET_S
        energy = frame["energy_kwh"].to_numpy(np.float64)[keep]
        cell = (std // 86400 - self.first_day[code]) * 96 + (std % 86400) // SLOT_S
        order = np.argsort(code, kind="stable")  # file order within each customer
        for grp in np.split(order, np.flatnonzero(np.diff(code[order])) + 1):
            flat = self.M[code[grp[0]]].reshape(-1)
            cells, first = np.unique(cell[grp], return_index=True)  # keep-first within the chunk
            values = energy[grp][first]
            new = np.isnan(flat[cells])  # keep-first against earlier chunks
            flat[cells[new]] = values[new]

    def finalize(self) -> dict[str, CustomerData]:
        """`CustomerData` exactly as `build_feature_table` builds it, then M stored as float32."""
        out: dict[str, CustomerData] = {}
        for i, cid in enumerate(self.ids):
            M, self.M[i] = self.M[i], None
            dates = np.arange(self.first_day[i], self.first_day[i] + len(M)).astype("datetime64[D]")
            cd = customer_data(dates, M, M[~np.isnan(M)])
            out[cid] = replace(cd, M=M.astype(np.float32))
        return out


def build_pool(
    path: str | Path, dailies: Mapping[str, pd.DataFrame], size: int, chunksize: int = 500_000
) -> dict[str, CustomerData]:
    """The pseudo-labelling pool: `select_pool` on the first pass's dailies, then one more streaming pass."""
    ids = select_pool({c: int(d["valid"].sum()) for c, d in dailies.items()}, size)
    if not ids:
        return {}
    day0 = {c: int(dailies[c].index[0].to_datetime64().astype("datetime64[D]").astype(np.int64)) for c in ids}
    acc = PoolAccumulator({c: (day0[c], len(dailies[c])) for c in ids})
    for frame, _ in iter_chunks(path, chunksize=chunksize):
        acc.add(frame)
    return acc.finalize()


# --------------------------------------------------------------------------- prior


def curve_sensitivity(curves_q: dict[float, dict[str, np.ndarray]]) -> float:
    """max over q, q', signal and week of |c_q - c_q'|."""
    worst = 0.0
    for name in SIGNALS:
        stack = np.vstack([np.asarray(c[name], float) for c in curves_q.values()])
        worst = max(worst, float(np.max(stack.max(axis=0) - stack.min(axis=0))))
    return worst


def check_sensitivity(curves_q: dict[float, dict[str, np.ndarray]], n_usable: int) -> tuple[float, str | None]:
    """(sensitivity, reason the unlabelled climatology option is disabled, or None when it passes)."""
    sensitivity = curve_sensitivity(curves_q)
    if n_usable < MIN_PRIOR_CUSTOMERS:
        return sensitivity, f"fewer than {MIN_PRIOR_CUSTOMERS} usable customers (n={n_usable})"
    if sensitivity > SENSITIVITY_MAX:
        return sensitivity, f"sensitivity {sensitivity:.3f} > {SENSITIVITY_MAX}"
    return sensitivity, None


@dataclass(frozen=True)
class UnlabelledPrior:
    """Label-free aggregates of the unlabelled series; fold-independent (cache rule compliant)."""

    curves_q: dict[float, dict[str, np.ndarray]]
    sensitivity: float
    passed: bool
    reason: str | None
    spread: np.ndarray
    n_customers: int
    n_contrast: int
    n_q: dict[float, int]

    @property
    def evidence_scale(self) -> np.ndarray | None:
        """spread / median(spread); None when the spread is unusable."""
        med = float(np.median(self.spread)) if self.spread.size else np.nan
        if not np.isfinite(med) or med <= 0:
            return None
        return self.spread / med

    def to_json(self) -> dict:
        return {
            "n_customers": self.n_customers,
            "n_contrast": self.n_contrast,
            "n_q": {str(q): n for q, n in self.n_q.items()},
            "sensitivity": self.sensitivity,
            "sensitivity_max": SENSITIVITY_MAX,
            "passed": self.passed,
            "reason": self.reason,
            "curves_q": {str(q): {k: v.tolist() for k, v in c.items()} for q, c in self.curves_q.items()},
            "spread": self.spread.tolist(),
        }


def _spread(dailies: dict[str, pd.DataFrame]) -> np.ndarray:
    weekly = []
    for d in dailies.values():
        v = d[d["valid"]]
        if len(v) < MIN_SPREAD_DAYS:
            continue
        s = v["s"].to_numpy(float)
        weekly.append(weekly_median(s - np.median(s), v["woy"].to_numpy()))
    if not weekly:
        return np.ones(N_WEEKS)
    W = np.vstack(weekly)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mad = np.nanmedian(np.abs(W - np.nanmedian(W, axis=0)), axis=0)
    ok = np.isfinite(mad) & (mad > 0)
    if not ok.any():
        return np.ones(N_WEEKS)
    return np.where(ok, mad, float(np.median(mad[ok])))


def build_unlabelled_prior(dailies: dict[str, pd.DataFrame]) -> UnlabelledPrior:
    contrast = {}
    for cid, d in dailies.items():
        v = d[d["valid"]]
        contrast[cid] = _contrast(last365(v)) if len(v) else np.nan
    finite = {c: x for c, x in contrast.items() if np.isfinite(x)}
    vals = np.array(list(finite.values()), float)
    curves_q, n_q = {}, {}
    for q in QUANTILES:
        ids = [c for c, x in finite.items() if x >= np.quantile(vals, q)] if vals.size else []
        n_q[q] = len(ids)
        curves_q[q] = signal_curves(dailies, ids)
    sensitivity, reason = check_sensitivity(curves_q, len(finite))
    return UnlabelledPrior(
        curves_q=curves_q,
        sensitivity=sensitivity,
        passed=reason is None,
        reason=reason,
        spread=_spread(dailies),
        n_customers=len(dailies),
        n_contrast=len(finite),
        n_q=n_q,
    )
