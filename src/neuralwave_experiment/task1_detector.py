"""Shared window-classification utilities for the frozen evaluation protocol."""
from __future__ import annotations


from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from neuralwave_experiment.encoder import EncoderConfig
from neuralwave_experiment.windows import WindowConfig, WindowBatch, build_windows

# The configuration every Task 1 checkpoint is trained and loaded with.
TASK1_ENCODER_CONFIG = EncoderConfig(context_length=672, patch_length=16, d_model=64, n_heads=4, n_layers=2, dropout=0.2)


class WindowClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, values: torch.Tensor, observed: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        # values: [B, L, 1], observed: [B, L, 1], calendar: [B, L, 4]
        emb = self.encoder(values, observed, calendar)  # [B, d_model]
        emb = self.dropout(emb)
        logits = self.head(emb).squeeze(-1)  # [B]
        return logits


def assign_window_labels(batch: WindowBatch, labels_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Assign physical binary PV label to each window.
    
    y = 0: confirmed non-PV customer OR pre-installation window (< install_date - 7d)
    y = 1: PV customer post-installation window (>= install_date + 7d)
    y = -1: transition window within +- 7 days of install date (masked out)
    """
    label_map = labels_df.assign(customer_id=labels_df.customer_id.astype(str)).set_index("customer_id")
    y = np.full(len(batch.customer_ids), -1, dtype=np.int64)
    valid_mask = np.zeros(len(batch.customer_ids), dtype=bool)

    for i, (cid, center_d) in enumerate(zip(batch.customer_ids, batch.center_dates)):
        cid_str = str(cid)
        if cid_str not in label_map.index:
            continue
        row = label_map.loc[cid_str]
        installed = bool(row.pv_installed)

        if not installed:
            # Confirmed non-PV customer: every window is negative
            y[i] = 0
            valid_mask[i] = True
        else:
            # PV customer
            raw_date = row.pv_installation_date
            if pd.isna(raw_date):
                # Pre-existing PV: all windows are positive
                y[i] = 1
                valid_mask[i] = True
            else:
                inst_d = pd.Timestamp(raw_date).date()
                diff = (center_d - inst_d).days
                if diff < -7:
                    # Clearly before installation
                    y[i] = 0
                    valid_mask[i] = True
                elif diff > 7:
                    # Clearly after installation
                    y[i] = 1
                    valid_mask[i] = True
                else:
                    # Transition window: mask out
                    y[i] = -1
                    valid_mask[i] = False

    return y, valid_mask


def extract_customer_windows(canonical_dir: Path, customer_ids: list[str], win_cfg: WindowConfig) -> dict[str, WindowBatch]:
    """Read measurements per customer and build windows."""
    meas_path = canonical_dir / "measurements.parquet"
    import pyarrow.parquet as pq

    # Read all relevant rows
    cust_set = set(customer_ids)
    table = pq.read_table(meas_path, filters=[("customer_id", "in", list(cust_set))])
    df = table.to_pandas()
    df["customer_id"] = df["customer_id"].astype(str)

    batches = {}
    for cid in customer_ids:
        sub = df[df["customer_id"] == str(cid)]
        if not sub.empty:
            batch = build_windows(sub, win_cfg)
            if len(batch.customer_ids) > 0:
                batches[str(cid)] = batch
    return batches


def concat_batches(batch_list: list[WindowBatch]) -> WindowBatch:
    from dataclasses import replace
    return replace(
        batch_list[0],
        values=np.concatenate([b.values for b in batch_list]),
        observed=np.concatenate([b.observed for b in batch_list]),
        calendar=np.concatenate([b.calendar for b in batch_list]),
        magnitude=np.concatenate([b.magnitude for b in batch_list]),
        customer_ids=sum([list(b.customer_ids) for b in batch_list], []),
        center_dates=sum([list(b.center_dates) for b in batch_list], []),
        start_epochs=np.concatenate([b.start_epochs for b in batch_list]),
    )



