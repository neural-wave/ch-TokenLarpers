"""PV capacity-extension detection: a rolling capacity nowcast and a per-customer seasonal change-point test.

Steps, per customer the main model calls PV:
1. Every STEP days, a WIN-day window yields label-free physics features (`window_features`): the length of the
   zero-withdrawal run around solar noon on its typical and clearest days (clipping-robust: a larger array
   starts the run earlier and ends it later), load levels outside PV hours, midday/evening ratios, season.
2. A CatBoost regressor (`fit_nowcaster`, fitted inside `procedure.fit` on that fit's training customers
   only) maps each window to log kWp installed at that time. Targets: labelled PV customers' windows that
   lie inside one capacity regime (initial kWp, or the total after a labelled extension), plus optional
   pseudo-labelled windows of unlabelled customers the fit's own main model calls PV (`POOL_PROB`).
3. Per-customer seasonal change-point (`harmonic_steps`): the nowcast k(t) is modelled as a level plus the
   customer's own annual harmonics plus up to MAX_PER_CUSTOMER persistent steps, fitted jointly by least
   squares; a step needs ≥ MIN_RATIO, z ≥ MIN_Z, and the same margin between deseasonalized medians after
   and before it. (The nowcast's seasonal swing is customer-specific and larger than typical extension steps,
   so a plain level test against earlier plateaus cannot separate them; see docs/extension_evidence.md.)
4. Date: two-segment fit on the daily year-over-year difference of a 7-day zero-run envelope (±REFINE days).
   Capacity after: median nowcast on the new plateau; if that is not above the predicted initial kWp, the
   predicted initial kWp × the cumulative step ratio.
5. At most MAX_PER_CUSTOMER events per customer and MAX_TOTAL per prediction call, highest z first. The
   challenge text states 2 of the 20 hidden-test customers have extensions, one of them twice (3 events).

All thresholds are fixed constants from synthetic development seeds; nothing is tuned on real labels.
Extension detection never changes the main predictions (the main frame is an input, not an output).
"""

from __future__ import annotations

import datetime as dt
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pvdetect.pipeline import CustomerData, FeatureTable

WIN, STEP = 35, 7  # window length and stride, days
YEAR = 52  # windows per 364 days
NOON = 50  # 12:30 slot in the fixed UTC+1 day
ZERO_KWH = 0.005  # a slot at or below this withdrawal counts as zero (pvdetect.features uses the same)
MIN_DAYS = 18  # valid days a window needs
POST_INSTALL_DAYS = 14  # windows start at least this long after the (predicted) installation date
MIN_RATIO = 1.15
MIN_Z = 4.0
HARMONICS = 2  # annual harmonics fitted per customer (one when the usable span is under two years)
MIN_SPAN_H1 = 60  # windows of usable span below which no harmonic is fitted
MIN_SEG = 6  # windows on each side of a candidate step
MIN_GAP = 17  # windows between two steps of one customer
PLATEAU = 6
RAMP = WIN // STEP  # windows after a boundary still mixing both regimes
REFINE = 50  # days searched either side of a step for its date
MAX_PER_CUSTOMER = 2
MAX_TOTAL = 3
POOL_PROB = 0.95  # pseudo-labelled unlabelled customers: main-model P(PV) at least this
MIN_TRAIN_CUSTOMERS = 5
FEATURES = (
    "zw50",
    "zw80",
    "zw95",
    "zshare",
    "log_eve",
    "log_night",
    "log_morn",
    "log_mid10",
    "log_mid50",
    "mid10_eve",
    "mid50_eve",
    "eve_night",
    "woy_sin",
    "woy_cos",
)
COLUMNS = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]
BANDS = {"eve": (76, 88), "night": (0, 20), "morn": (20, 28), "mid": (40, 60)}  # UTC+1 slots: 19–22, 0–5, 5–7, 10–15 h


@dataclass(frozen=True)
class Event:
    customer_id: str
    date: dt.date
    kwp_after: float
    z: float


# ---------------------------------------------------------------------------------------------- features


def zero_run_hours(M: np.ndarray) -> np.ndarray:
    """Per day, hours of the zero-withdrawal run containing 12:30 (0 when 12:30 is not zero, NaN if missing)."""
    z = M <= ZERO_KWH
    out = np.where(np.isnan(M[:, NOON]), np.nan, 0.0)
    for d in np.flatnonzero(z[:, NOON]):
        row = z[d]
        lo = NOON
        while lo > 0 and row[lo - 1]:
            lo -= 1
        hi = NOON
        while hi < 95 and row[hi + 1]:
            hi += 1
        out[d] = (hi - lo + 1) / 4
    return out


def window_starts(n_days: int) -> np.ndarray:
    return np.arange(0, max(n_days - WIN + 1, 0), STEP)


def window_features(cd: CustomerData) -> pd.DataFrame:
    """One row per window: FEATURES plus `start` and `centre` (Timestamps); NaN rows for thin windows.
    Levels are kW (slot kWh × 4) over fixed UTC+1 bands; zero-run lengths are hours."""
    M = np.asarray(cd.M, float)
    zw = zero_run_hours(M)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        kw = {name: np.nanmean(M[:, a:b], 1) * 4 for name, (a, b) in BANDS.items()}
        zmid = np.nanmean(M[:, 40:60] <= ZERO_KWH, 1)
    dates = pd.DatetimeIndex(cd.dates.astype("datetime64[ns]"))
    starts = window_starts(M.shape[0])

    def lg(x: float) -> float:
        return float(np.log(max(x, 0.0) + 0.02))

    rows = []
    for a in starts:
        sl = slice(a, a + WIN)
        ok = np.isfinite(zw[sl]) & np.isfinite(kw["eve"][sl]) & np.isfinite(kw["mid"][sl])
        if ok.sum() < MIN_DAYS:
            rows.append(dict.fromkeys(FEATURES, np.nan))
            continue
        z, md = zw[sl][ok], kw["mid"][sl][ok]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            e, n, m = (float(np.nanmedian(kw[k][sl][ok])) for k in ("eve", "night", "morn"))
        m10, m50 = float(np.quantile(md, 0.1)), float(np.median(md))
        woy = 2 * np.pi * dates[a + WIN // 2].dayofyear / 365.25
        rows.append(
            {
                "zw50": float(np.median(z)),
                "zw80": float(np.quantile(z, 0.8)),
                "zw95": float(np.quantile(z, 0.95)),
                "zshare": float(np.nanmean(zmid[sl][ok])),
                "log_eve": lg(e),
                "log_night": lg(n),
                "log_morn": lg(m),
                "log_mid10": lg(m10),
                "log_mid50": lg(m50),
                "mid10_eve": lg(m10) - lg(e),
                "mid50_eve": lg(m50) - lg(e),
                "eve_night": lg(e) - lg(n),
                "woy_sin": float(np.sin(woy)),
                "woy_cos": float(np.cos(woy)),
            }
        )
    out = pd.DataFrame(rows, columns=list(FEATURES))
    out["start"] = dates[starts]
    out["centre"] = dates[starts + WIN // 2]
    return out


# ------------------------------------------------------------------------------------------- nowcaster


def regime_rows(cid: str, cd: CustomerData, regimes: list[tuple[pd.Timestamp, float]]) -> pd.DataFrame:
    """Windows of one customer lying wholly inside one capacity regime, ≥ POST_INSTALL_DAYS after it starts.
    `regimes`: (start, kWp) sorted by start; the last regime runs to the end of the series."""
    f = window_features(cd)
    start = f["start"]
    end = start + pd.Timedelta(days=WIN)
    keep = np.zeros(len(f), bool)
    target = np.full(len(f), np.nan)
    for j, (t0, kwp) in enumerate(regimes):
        t1 = regimes[j + 1][0] if j + 1 < len(regimes) else pd.Timestamp.max
        if pd.isna(t0) or not (np.isfinite(kwp) and kwp > 0):
            continue  # an unusable label contributes no windows (as in the main capacity fit)
        sel = (start >= t0 + pd.Timedelta(days=POST_INSTALL_DAYS)) & (end <= t1)
        keep |= sel.to_numpy()
        target[sel.to_numpy()] = np.log(kwp)
    f = f[keep].assign(customer_id=cid, log_kwp=target[keep])
    return f.dropna(subset=[*FEATURES, "log_kwp"])


def fit_nowcaster(rows: pd.DataFrame):
    """CatBoost regressor on log kWp; every customer carries the same total weight."""
    from catboost import CatBoostRegressor

    w = 1.0 / rows.groupby("customer_id")["log_kwp"].transform("size").to_numpy(float)
    reg = CatBoostRegressor(
        iterations=400, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, loss_function="RMSE",
        random_seed=0, verbose=False, allow_writing_files=False, thread_count=1,
    )
    reg.fit(rows[list(FEATURES)].to_numpy(float), rows["log_kwp"].to_numpy(float), sample_weight=w)
    return reg


def nowcast(nowcaster, f: pd.DataFrame, usable: np.ndarray) -> np.ndarray:
    k = np.full(len(f), np.nan)
    ok = usable & f[list(FEATURES)].notna().all(axis=1).to_numpy()
    if ok.any():
        k[ok] = nowcaster.predict(f.loc[ok, list(FEATURES)].to_numpy(float))
    return k


# --------------------------------------------------------------------------------------- change tests


def _harmonics(theta: np.ndarray, n_h: int) -> np.ndarray:
    cols = [np.ones_like(theta)]
    for h in range(1, n_h + 1):
        cols += [np.cos(h * theta), np.sin(h * theta)]
    return np.column_stack(cols)


def harmonic_steps(k: np.ndarray, doy: np.ndarray) -> list[tuple[int, float, float]]:
    """(window index, log step, z) of up to MAX_PER_CUSTOMER persistent capacity steps in the nowcast k.

    Model, per customer: k(t) = a + Σ_h [b_h cos(hθ_t) + c_h sin(hθ_t)] + Σ_j δ_j·1[t ≥ τ_j] + e_t, θ from the
    window-centre day of year `doy`. The customer's own seasonal swing of the nowcast is fitted jointly with
    the steps over the whole series, so it can neither mimic nor hide a step. Greedy: for every admissible τ
    add the step column and solve least squares; z = δ / se(δ), σ from the residual MAD, inflated for the
    overlap of consecutive windows. Accept the best positive step if δ ≥ log MIN_RATIO, z ≥ MIN_Z and the
    deseasonalized medians after vs before (ramp windows excluded) also differ by ≥ log MIN_RATIO, which a
    short dip (outage, vacation) cannot achieve."""
    ok = np.flatnonzero(np.isfinite(k))
    if ok.size < 2 * PLATEAU + 4:
        return []
    t, y = ok, k[ok]
    span = int(ok[-1] - ok[0] + 1)
    n_h = HARMONICS if span >= 2 * YEAR else (1 if span >= MIN_SPAN_H1 else 0)
    base = _harmonics(2 * np.pi * doy[ok] / 365.25, n_h)
    n_eff = WIN / STEP

    def best_step(fixed: list[int]):
        """Best positive step position given the other steps `fixed`: (τ, δ, z, coef) or None."""
        D0 = np.column_stack([base, *[(t >= s).astype(float) for s in fixed]])
        best = None
        for j in range(MIN_SEG, t.size - MIN_SEG):
            tau = int(t[j])
            if any(abs(tau - s) < MIN_GAP for s in fixed):
                continue
            D = np.column_stack([D0, (t >= tau).astype(float)])
            coef, *_ = np.linalg.lstsq(D, y, rcond=None)
            delta = float(coef[-1])
            if delta <= 0:
                continue
            res = y - D @ coef
            sig = max(1.4826 * float(np.median(np.abs(res - np.median(res)))), 0.03)
            try:
                var = float(np.linalg.inv(D.T @ D)[-1, -1])
            except np.linalg.LinAlgError:
                continue
            z = delta / (sig * np.sqrt(var * n_eff))
            if best is None or z > best[2]:
                best = (tau, delta, z, coef)
        return best

    def accepted(tau: int, delta: float, z: float, coef: np.ndarray, others: list[int]) -> bool:
        ds = y - base[:, 1:] @ coef[1 : base.shape[1]] if base.shape[1] > 1 else y
        earlier = [s for s in others if s < tau]
        later = [s for s in others if s > tau]
        pre = ds[(t < tau - RAMP) & (t >= (max(earlier) + RAMP if earlier else t[0]))]
        post = ds[(t >= tau + RAMP) & (t < (min(later) if later else t[-1] + 1))]
        if pre.size < 4 or post.size < 4:
            return False
        robust = float(np.median(post) - np.median(pre))
        return delta >= np.log(MIN_RATIO) and z >= MIN_Z and robust >= np.log(MIN_RATIO)

    # Greedy, as evaluated: with two close changes the first step can sit between them (re-placing each step
    # given the other changed no benchmark outcome on a fresh block, so it is not used; see the evidence doc).
    steps: list[tuple[int, float, float]] = []
    for _ in range(MAX_PER_CUSTOMER):
        found = best_step([s for s, _, _ in steps])
        if found is None or not accepted(*found, [s for s, _, _ in steps]):
            break
        steps.append(found[:3])
    return sorted(steps)


def refine_day(M: np.ndarray, d0: int) -> int:
    """Daily date near day index d0: two-segment SSE minimum of the year-over-year difference of the 7-day
    max zero-run envelope (the 7-day min midday load when the customer does not clip near d0)."""
    env = pd.Series(zero_run_hours(M)).rolling(7, center=True, min_periods=3).max().to_numpy()
    near = env[max(d0 - 60, 0) : d0 + 60]
    if not np.isfinite(near).any() or np.nanmax(near) < 1.0:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mid = np.nanmean(M[:, 40:60], 1)
        env = -pd.Series(mid).rolling(7, center=True, min_periods=3).min().to_numpy()
    diff = np.full(env.size, np.nan)
    diff[364:] = env[364:] - env[:-364]
    lo, hi = max(d0 - REFINE, 0), min(d0 + REFINE, env.size)
    seg = diff[lo:hi]
    if np.isfinite(seg).sum() < 30:
        seg = env[lo:hi]
    best, best_t = np.inf, d0
    for t in range(10, seg.size - 10):
        a, b = seg[:t], seg[t:]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 5 or b.size < 5:
            continue
        sse = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        if sse < best:
            best, best_t = sse, lo + t
    return int(best_t)


def customer_events(nowcaster, cid: str, cd: CustomerData, install: dt.date | None, initial_kwp: float) -> list[Event]:
    f = window_features(cd)
    if f.empty:
        return []
    first = pd.Timestamp(install) if install is not None else pd.Timestamp(cd.dates[0]) - pd.Timedelta(days=1)
    usable = (f["start"] >= first + pd.Timedelta(days=POST_INSTALL_DAYS)).to_numpy()
    k = nowcast(nowcaster, f, usable)
    steps = harmonic_steps(k, f["centre"].dt.dayofyear.to_numpy(float))
    events, cum = [], 1.0
    for j, (tau, step, z) in enumerate(steps):
        nxt = steps[j + 1][0] if j + 1 < len(steps) else len(k)
        plateau = k[tau + RAMP : nxt]
        new_log = float(np.nanmedian(plateau)) if np.isfinite(plateau).sum() >= 4 else np.nan
        # the installation itself (predicted install date too early): a step from below the predicted initial
        # capacity up to about it is not an extension
        if np.isfinite(new_log) and new_log < np.log(initial_kwp * MIN_RATIO) and new_log - step < np.log(initial_kwp / MIN_RATIO):
            continue
        d0 = int(np.searchsorted(cd.dates, f["centre"].iloc[tau].to_datetime64().astype("datetime64[D]")))
        date = pd.Timestamp(cd.dates[refine_day(np.asarray(cd.M, float), d0)]).date()
        if install is not None and date <= install:
            continue
        if events and (date - events[-1].date).days <= 30:
            continue  # one ±30-day match window holds at most one event
        cum *= float(np.exp(step))
        absolute = float(np.exp(new_log)) if np.isfinite(new_log) else np.nan
        kwp = absolute if np.isfinite(absolute) and absolute > initial_kwp else initial_kwp * cum
        events.append(Event(cid, date, float(kwp), float(z)))
    return events


# ------------------------------------------------------------------------------------------- fit / predict


def _pool_regimes(main_model: dict, table: FeatureTable) -> dict[str, tuple[CustomerData, list]]:
    """Unlabelled pool customers the main model calls PV with P ≥ POOL_PROB: one regime from the predicted
    date at the predicted kWp (pseudo-labels; extension customers in the pool are mislabelled before their
    extension, which the nowcaster must tolerate)."""
    from pvdetect.pipeline import predict

    pool = table.extension_pool or {}
    if not pool:
        return {}
    view = FeatureTable(ids=list(pool), customers=pool, cache=table.cache.setdefault("extension_pool", {}))
    main = {**main_model, "extensions": {"enabled": False}}
    try:
        pred, _ = predict(main, view, view.ids)
    except Exception as exc:  # e.g. an embedding block needs parsed rows the pool does not keep
        print(f"warning: extension pool not pseudo-labelled ({type(exc).__name__}: {exc})", file=sys.stderr)
        return {}
    out = {}
    for r in pred.itertuples(index=False):
        if r.pv_installed and r.prob >= POOL_PROB and pd.notna(r.pv_installation_date) and np.isfinite(r.pv_capacity_kwp):
            out[r.customer_id] = (pool[r.customer_id], [(pd.Timestamp(r.pv_installation_date), float(r.pv_capacity_kwp))])
    return out


def fit_extensions(
    table: FeatureTable,
    labels: pd.DataFrame,
    train_ids: list[str],
    ext_labels: pd.DataFrame | None,
    main_model: dict,
) -> tuple[dict, object | None]:
    """(model["extensions"] section, nowcaster object) from the training customers only (plus the optional
    pseudo-labelled pool). Disabled when fewer than MIN_TRAIN_CUSTOMERS labelled PV customers have windows."""
    lab = labels.set_index("customer_id")
    ext = ext_labels if ext_labels is not None else pd.DataFrame(columns=COLUMNS)
    ext = ext[ext["customer_id"].isin(set(train_ids))]
    parts = []
    for cid in train_ids:
        row, cd = lab.loc[cid], table.customers.get(cid)
        if cd is None or not bool(row["pv_installed"]):
            continue
        regimes = [(pd.Timestamp(row["pv_installation_date"]), float(row["pv_capacity_kwp"]))]
        for e in ext[ext["customer_id"] == cid].sort_values("extension_date").itertuples(index=False):
            regimes.append((pd.Timestamp(e.extension_date), float(e.pv_capacity_after_extension_kwp)))
        parts.append(regime_rows(cid, cd, regimes))
    labelled = sum(1 for p in parts if len(p))
    pool = _pool_regimes(main_model, table)
    parts += [regime_rows(f"pool:{cid}", cd, reg) for cid, (cd, reg) in pool.items()]
    info = {"labelled_customers": labelled, "pool_customers": sum(1 for _ in pool)}
    rows = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else None
    if labelled < MIN_TRAIN_CUSTOMERS or rows is None:
        return {"enabled": False, **info, "reason": "too few labelled PV customers"}, None
    return {"enabled": True, "catboost_file": "catboost_ext.cbm", "windows": int(len(rows)), **info}, fit_nowcaster(rows)


def predict_extensions(model: dict, table: FeatureTable, pred: pd.DataFrame) -> pd.DataFrame:
    """Extension rows for the TRUE customers of `pred`; ≤ MAX_PER_CUSTOMER each, ≤ MAX_TOTAL by z overall."""
    nowcaster = model.get("_objects", {}).get("catboost_ext")
    if not (model.get("extensions") or {}).get("enabled") or nowcaster is None:
        return pd.DataFrame(columns=COLUMNS)
    events: list[Event] = []
    for r in pred[pred["pv_installed"].astype(bool)].itertuples(index=False):
        cd = table.customers.get(r.customer_id)
        if cd is None or not np.isfinite(r.pv_capacity_kwp):
            continue
        install = pd.Timestamp(r.pv_installation_date).date() if pd.notna(r.pv_installation_date) else None
        events += customer_events(nowcaster, r.customer_id, cd, install, float(r.pv_capacity_kwp))
    events = sorted(events, key=lambda e: -e.z)[:MAX_TOTAL]
    events.sort(key=lambda e: (e.customer_id, e.date))
    return pd.DataFrame(
        [{"customer_id": e.customer_id, "extension_date": pd.Timestamp(e.date), "pv_capacity_after_extension_kwp": e.kwp_after} for e in events],
        columns=COLUMNS,
    )
