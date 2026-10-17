"""Bounded canonical-Parquet customer/window streaming for unlabelled pretraining.

The source keeps only a compact customer/window index in memory.  Customer arrays
are materialised as private per-customer NPZ shards and loaded one shard at a time.
IDs are runtime-only and are not included in index digests or checkpoint metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Collection

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .windows import WindowBatch, WindowConfig, build_windows


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class WindowStreamConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    max_windows_per_customer: int = 128
    windows_per_epoch: int = 4096
    batch_size: int = 32
    season_quota: str = "balanced"
    min_windows_per_available_season: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_windows_per_customer <= 0 or self.windows_per_epoch <= 0 or self.batch_size <= 0:
            raise ValueError("stream limits must be positive")
        if self.season_quota not in {"balanced", "proportional"}:
            raise ValueError("season_quota must be 'balanced' or 'proportional'")
        if self.min_windows_per_available_season < 0:
            raise ValueError("min_windows_per_available_season must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "window": {
                "context_length": self.window.context_length,
                "stride": self.window.stride,
                "min_observed_fraction": self.window.min_observed_fraction,
                "max_windows_per_customer": self.window.max_windows_per_customer,
            },
            "max_windows_per_customer": self.max_windows_per_customer,
            "windows_per_epoch": self.windows_per_epoch,
            "batch_size": self.batch_size,
            "season_quota": self.season_quota,
            "min_windows_per_available_season": self.min_windows_per_available_season,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class WindowIndex:
    n_customers: int
    n_windows: int
    season_counts: Mapping[str, int]
    source_order_sha256: str
    index_sha256: str
    # Runtime-only cache references.  They are deliberately excluded from repr/metadata.
    _customer_ids: tuple[str, ...] = field(default=(), repr=False, compare=False)
    _shards: tuple[str, ...] = field(default=(), repr=False, compare=False)
    _seasons: tuple[tuple[str, ...], ...] = field(default=(), repr=False, compare=False)


@dataclass
class WindowSamplerState:
    epoch: int = 0
    batch_in_epoch: int = 0
    windows_drawn: int = 0
    rng_state: dict[str, Any] | None = None
    source_order_sha256: str = ""
    index_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch), "batch_in_epoch": int(self.batch_in_epoch),
            "windows_drawn": int(self.windows_drawn), "rng_state": self.rng_state,
            "source_order_sha256": self.source_order_sha256, "index_sha256": self.index_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WindowSamplerState":
        return cls(epoch=int(payload.get("epoch", 0)), batch_in_epoch=int(payload.get("batch_in_epoch", 0)),
                   windows_drawn=int(payload.get("windows_drawn", 0)), rng_state=payload.get("rng_state"),
                   source_order_sha256=str(payload.get("source_order_sha256", "")),
                   index_sha256=str(payload.get("index_sha256", "")))


@dataclass(frozen=True)
class CustomerSeries:
    customer_id: str
    frame: pd.DataFrame


class EpochSource:
    """Explicit-role canonical epoch source.

    ``path`` can be a canonical Parquet file or a DataFrame for artificial tests.
    The latter is copied into a private temporary Parquet file, preserving the same
    bounded reader path as production.
    """
    def __init__(self, path: str | Path | Iterable[str | Path], *, role: str, manifest_sha256: str | None = None,
                 labelled_ids: Collection[str] = (), cache_dir: str | Path | None = None):
        if role not in {"unlabelled_measurements", "labelled_measurements"}:
            raise ValueError("source role must be explicit and measurement-only")
        self.role = role
        self._labelled_ids = frozenset(str(customer_id) for customer_id in labelled_ids)
        if isinstance(path, (str, Path)):
            paths = (Path(path).resolve(),)
        else:
            paths = tuple(Path(p).resolve() for p in path)
        if not paths or any(not p.is_file() for p in paths):
            raise FileNotFoundError("canonical source does not exist")
        self.paths = paths
        self.path = paths[0]
        self.source_bytes_sha256 = _json_hash([(str(p), _file_hash(p)) for p in paths])
        self.manifest_sha256 = manifest_sha256 or self.source_bytes_sha256
        self._cache_root = Path(cache_dir).resolve() if cache_dir else Path(tempfile.mkdtemp(prefix="neuralwave-stream-"))
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._index_cache: dict[str, WindowIndex] = {}
        self._raw_parts: dict[str, tuple[Path, ...]] | None = None
        self._check_columns()
        if self.role == "unlabelled_measurements" and self._labelled_ids:
            overlap = self._find_overlap(self._labelled_ids)
            if overlap:
                raise ValueError("unlabelled source overlaps labelled customer IDs")

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, role: str = "unlabelled_measurements",
                   labelled_ids: Collection[str] = ()) -> "EpochSource":
        required = {"customer_id", "timestamp_epoch", "energy_kwh", "value_status"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"measurements missing columns: {sorted(missing)}")
        tmp = Path(tempfile.mkdtemp(prefix="neuralwave-source-")) / "measurements.parquet"
        out = frame.loc[:, sorted(required)].copy()
        out["customer_id"] = out["customer_id"].astype(str)
        out["timestamp_epoch"] = pd.to_numeric(out["timestamp_epoch"], errors="raise").astype("int64")
        if (out["timestamp_epoch"] % 900 != 0).any():
            raise ValueError("timestamp_epoch is not on the 900-second grid")
        out.to_parquet(tmp, index=False)
        return cls(tmp, role=role, labelled_ids=labelled_ids, cache_dir=tmp.parent / "cache")

    def _check_columns(self) -> None:
        required = {"customer_id", "timestamp_epoch", "energy_kwh", "value_status"}
        for path in self.paths:
            missing = required - set(pq.ParquetFile(path).schema.names)
            if missing: raise ValueError(f"canonical source missing columns: {sorted(missing)}")

    def _find_overlap(self, ids: Collection[str]) -> set[str]:
        wanted = set(map(str, ids)); found: set[str] = set()
        for batch in self._iter_batches(["customer_id"]):
            found.update(set(map(str, batch.column(0).to_pylist())) & wanted)
            if found == wanted: break
        return found

    def _iter_batches(self, columns: list[str] | None = None) -> Iterator[Any]:
        selected = columns or ["customer_id", "timestamp_epoch", "energy_kwh", "value_status"]
        for path in self.paths:
            yield from pq.ParquetFile(path).iter_batches(columns=selected, batch_size=8192)

    def _ensure_raw_shards(self) -> dict[str, tuple[Path, ...]]:
        if self._raw_parts is not None: return self._raw_parts
        root = self._cache_root / ("raw-" + self.source_bytes_sha256[:24]); root.mkdir(parents=True, exist_ok=True)
        marker = root / "_complete.json"
        if marker.exists():
            try:
                payload = json.loads(marker.read_text())
                if payload.get("source_bytes") == self.source_bytes_sha256:
                    self._raw_parts = {str(k): tuple(Path(x) for x in v) for k, v in payload["parts"].items()}
                    return self._raw_parts
            except (OSError, ValueError, KeyError, TypeError): pass
        parts: dict[str, list[Path]] = {}; part_no = 0
        for batch in self._iter_batches():
            frame = pd.DataFrame({name: batch.column(i).to_pylist() for i, name in enumerate(["customer_id", "timestamp_epoch", "energy_kwh", "value_status"])})
            frame["customer_id"] = frame["customer_id"].astype(str)
            for customer, group in frame.groupby("customer_id", sort=False):
                bucket = root / hashlib.sha256(customer.encode()).hexdigest(); bucket.mkdir(exist_ok=True)
                part = bucket / f"part-{part_no:09d}.parquet"; group.to_parquet(part, index=False)
                parts.setdefault(customer, []).append(part)
            part_no += 1
        payload = {"source_bytes": self.source_bytes_sha256, "manifest": self.manifest_sha256, "parts": {k: [str(p) for p in v] for k, v in parts.items()}}
        marker.write_text(json.dumps(payload, sort_keys=True))
        self._raw_parts = {k: tuple(v) for k, v in parts.items()}
        return self._raw_parts

    def iter_customers(self, ids: Iterable[str] | None = None) -> Iterator[CustomerSeries]:
        wanted = None if ids is None else set(map(str, ids)); parts = self._ensure_raw_shards()
        for customer in sorted(parts):
            if wanted is not None and customer not in wanted: continue
            frames = [pd.read_parquet(part, columns=["customer_id", "timestamp_epoch", "energy_kwh", "value_status"]) for part in parts[customer]]
            frame = pd.concat(frames, ignore_index=True).sort_values("timestamp_epoch", kind="mergesort").reset_index(drop=True)
            yield CustomerSeries(customer, frame)

    @staticmethod
    def _frame(rows: list[tuple[str, int, Any, str]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=["customer_id", "timestamp_epoch", "energy_kwh", "value_status"])
        return frame.sort_values("timestamp_epoch", kind="mergesort").reset_index(drop=True)

    def _shard_path(self, customer: str, cfg: WindowStreamConfig) -> Path:
        return self._cache_root / (hashlib.sha256((self.source_bytes_sha256 + customer + _json_hash(cfg.as_dict())).encode()).hexdigest() + ".npz")

    def index_windows(self, cfg: WindowStreamConfig) -> WindowIndex:
        key = _json_hash({"source": self.source_bytes_sha256, "manifest": self.manifest_sha256, "config": cfg.as_dict()})
        if key in self._index_cache: return self._index_cache[key]
        customers: list[str] = []; shards: list[str] = []; seasons: list[tuple[str, ...]] = []
        season_counts = {"DJF": 0, "MAM": 0, "JJA": 0, "SON": 0}; n_windows = 0
        # iter_customers uses one frame at a time; no all-history tensor is retained.
        for series in self.iter_customers():
            if self.role == "unlabelled_measurements" and series.customer_id in self._labelled_ids:
                raise ValueError("unlabelled source overlaps labelled customer IDs")
            window_cfg = WindowConfig(context_length=cfg.window.context_length, stride=cfg.window.stride,
                                      min_observed_fraction=cfg.window.min_observed_fraction,
                                      max_windows_per_customer=min(cfg.max_windows_per_customer,
                                                                    cfg.window.max_windows_per_customer or cfg.max_windows_per_customer))
            windows = build_windows(series.frame, window_cfg)
            if len(windows.values) == 0: continue
            shard = self._shard_path(series.customer_id, cfg)
            if not shard.exists():
                np.savez_compressed(shard, values=windows.values, observed=windows.observed,
                                    calendar=windows.calendar, magnitude=windows.magnitude,
                                    start_epochs=windows.start_epochs,
                                    center_dates=np.asarray([d.isoformat() for d in windows.center_dates], dtype="U10"))
            cust_seasons = tuple(_season(d) for d in windows.center_dates)
            for s in cust_seasons: season_counts[s] += 1
            customers.append(series.customer_id); shards.append(str(shard)); seasons.append(cust_seasons); n_windows += len(cust_seasons)
        source_order = hashlib.sha256("\n".join(customers).encode()).hexdigest()
        # IDs are not exposed in public digests, but stable ordering is still auditable.
        digest_payload = {"source": self.manifest_sha256, "source_order": source_order,
                          "n_customers": len(customers), "n_windows": n_windows,
                          "seasons": season_counts, "config": cfg.as_dict()}
        index = WindowIndex(len(customers), n_windows, season_counts, source_order,
                            _json_hash(digest_payload), tuple(customers), tuple(shards), tuple(seasons))
        self._index_cache[key] = index
        return index

    def iter_window_batches(self, cfg: WindowStreamConfig, *, state: WindowSamplerState | None = None) -> Iterator[tuple[WindowBatch, WindowSamplerState]]:
        index = self.index_windows(cfg)
        return _iter_window_batches(self, index, cfg, state=state)


def _season(value: date) -> str:
    month = value.month
    return "DJF" if month in (12, 1, 2) else "MAM" if month in (3, 4, 5) else "JJA" if month in (6, 7, 8) else "SON"


def _epoch_sequence(index: WindowIndex, cfg: WindowStreamConfig, epoch: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), int(epoch)]))
    n_cust = len(index._customer_ids)
    if n_cust == 0 or index.n_windows == 0:
        return []
    budget = min(cfg.windows_per_epoch, index.n_windows)

    # Group window indices by customer and season
    cust_windows: list[dict[str, list[int]]] = []
    for ci, ss in enumerate(index._seasons):
        cw: dict[str, list[int]] = {}
        for wi, s in enumerate(ss):
            cw.setdefault(s, []).append(wi)
        for s_list in cw.values():
            rng.shuffle(s_list)
        cust_windows.append(cw)

    # Order customers randomly for this epoch
    cust_order = rng.permutation(n_cust).tolist()

    # Round-robin draw across customers, alternating seasons per customer
    picks: list[tuple[int, int]] = []
    active = list(cust_order)
    cust_season_idx = {ci: 0 for ci in range(n_cust)}

    while active and len(picks) < budget:
        next_active = []
        for ci in active:
            cw = cust_windows[ci]
            avail_seasons = [s for s, wlist in cw.items() if wlist]
            if not avail_seasons:
                continue
            s_idx = cust_season_idx[ci] % len(avail_seasons)
            s_target = avail_seasons[s_idx]
            cust_season_idx[ci] += 1
            wi = cw[s_target].pop()
            picks.append((ci, wi))
            if len(picks) >= budget:
                break
            if any(wlist for wlist in cw.values()):
                next_active.append(ci)
        active = next_active

    return picks[:budget]


def _iter_window_batches(source: EpochSource, index: WindowIndex, cfg: WindowStreamConfig,
                         *, state: WindowSamplerState | None) -> Iterator[tuple[WindowBatch, WindowSamplerState]]:
    st = state or WindowSamplerState(source_order_sha256=index.source_order_sha256, index_sha256=index.index_sha256)
    if st.source_order_sha256 and st.source_order_sha256 != index.source_order_sha256:
        raise ValueError("sampler source-order hash mismatch")
    if st.index_sha256 and st.index_sha256 != index.index_sha256:
        raise ValueError("sampler index hash mismatch")
    sequence = _epoch_sequence(index, cfg, st.epoch)
    drawn = st.windows_drawn
    offset = st.batch_in_epoch * cfg.batch_size
    if offset > len(sequence): raise ValueError("sampler batch cursor exceeds epoch")
    for begin in range(offset, len(sequence), cfg.batch_size):
        selections = sequence[begin:begin + cfg.batch_size]
        arrays: list[dict[str, Any]] = []
        for customer_i, window_i in selections:
            with np.load(index._shards[customer_i], allow_pickle=False) as z:
                arrays.append({k: z[k][window_i] for k in ("values", "observed", "calendar", "magnitude", "start_epochs", "center_dates")})
        batch = WindowBatch(np.stack([a["values"] for a in arrays]), np.stack([a["observed"] for a in arrays]),
                            np.stack([a["calendar"] for a in arrays]), np.stack([a["magnitude"] for a in arrays]),
                            [index._customer_ids[i] for i, _ in selections],
                            [date.fromisoformat(str(a["center_dates"])) for a in arrays],
                            np.asarray([a["start_epochs"] for a in arrays], dtype=np.int64))
        next_batch = begin // cfg.batch_size + 1
        drawn += len(selections)
        if next_batch * cfg.batch_size >= len(sequence):
            next_state = WindowSamplerState(st.epoch + 1, 0, drawn,
                                            rng_state={"seed": cfg.seed, "epoch": st.epoch + 1, "batch_in_epoch": 0},
                                            source_order_sha256=index.source_order_sha256,
                                            index_sha256=index.index_sha256)
        else:
            next_state = WindowSamplerState(st.epoch, next_batch, drawn,
                                            rng_state={"seed": cfg.seed, "epoch": st.epoch, "batch_in_epoch": next_batch},
                                            source_order_sha256=index.source_order_sha256,
                                            index_sha256=index.index_sha256)
        yield batch, next_state


def index_windows(source: EpochSource, cfg: WindowStreamConfig) -> WindowIndex:
    return source.index_windows(cfg)


def iter_window_batches(source: EpochSource, cfg: WindowStreamConfig, *, state: WindowSamplerState | None = None) -> Iterator[tuple[WindowBatch, WindowSamplerState]]:
    return source.iter_window_batches(cfg, state=state)


def assert_unlabelled_only(source: EpochSource, labelled_ids: Collection[str]) -> None:
    if source.role != "unlabelled_measurements":
        raise ValueError("pretraining source role must be unlabelled_measurements")
    overlap = source._find_overlap(set(map(str, labelled_ids)))
    if overlap:
        raise ValueError("unlabelled source overlaps labelled customer IDs")


def open_epoch_source(path: str | Path | Iterable[str | Path], *, role: str, labelled_ids: Collection[str] = (), manifest_sha256: str | None = None,
                      cache_dir: str | Path | None = None) -> EpochSource:
    return EpochSource(path, role=role, labelled_ids=labelled_ids, manifest_sha256=manifest_sha256, cache_dir=cache_dir)


__all__ = ["EpochSource", "CustomerSeries", "WindowStreamConfig", "WindowIndex", "WindowSamplerState",
           "open_epoch_source", "assert_unlabelled_only", "index_windows", "iter_window_batches"]
