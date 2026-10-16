"""Task 1 Window-Level PV Presence Detector using PatchTST.

Trains PatchTST on localized 7-day windows with physical solar labels,
evaluates across 5 customer-disjoint folds, and aggregates window
probabilities into customer-level PV predictions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuralwave_experiment.encoder import EncoderConfig
from neuralwave_experiment.encoder_factory import build_encoder
from neuralwave_experiment.windows import WindowConfig, WindowBatch, build_windows


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


def aggregate_customer_score(probs: np.ndarray, dates: list[date]) -> float:
    """High-inertia customer PV score from time series of window probabilities.
    
    Uses 90th percentile of summer/spring probabilities to prevent 2-week vacations
    or random winter anomalies from triggering false positives.
    """
    if len(probs) == 0:
        return 0.0
    
    # Solar season weights: months 4 through 9 (April - Sept) are peak solar in Switzerland
    months = np.array([d.month for d in dates])
    summer_mask = (months >= 4) & (months <= 9)
    
    if np.sum(summer_mask) >= 5:
        # 90th percentile during solar season
        score = float(np.percentile(probs[summer_mask], 90))
    else:
        # Fallback to full-year 90th percentile
        score = float(np.percentile(probs, 90))
    
    return score


def train_fold(
    fold: int,
    train_ids: list[str],
    val_ids: list[str],
    windows_dict: dict[str, WindowBatch],
    labels_df: pd.DataFrame,
    enc_cfg: EncoderConfig,
    backend: str = "official_patchtst",
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    device: str = "cpu",
    seed: int = 17,
    out_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, Any]]:
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)

    # 1. Prepare training data
    train_batches = [windows_dict[c] for c in train_ids if c in windows_dict]
    train_batch = concat_batches(train_batches)
    y_train_raw, valid_train = assign_window_labels(train_batch, labels_df)

    X_val_t = torch.tensor(train_batch.values[valid_train], dtype=torch.float32)
    X_obs_t = torch.tensor(train_batch.observed[valid_train], dtype=torch.bool)
    X_cal_t = torch.tensor(train_batch.calendar[valid_train], dtype=torch.float32)
    y_t = torch.tensor(y_train_raw[valid_train], dtype=torch.float32)

    # Positive class weighting (handles mild imbalance)
    pos_count = (y_t == 1).sum().item()
    neg_count = (y_t == 0).sum().item()
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32).to(device)

    dataset = TensorDataset(X_val_t, X_obs_t, X_cal_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Build model
    backbone = build_encoder(enc_cfg, backend=backend)
    model = WindowClassifier(backbone, d_model=enc_cfg.d_model, dropout=enc_cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 3. Train
    model.train()
    history = []
    for ep in range(epochs):
        ep_losses = []
        for v, m, c, target in loader:
            v, m, c, target = v.to(device), m.to(device), c.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(v, m, c)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            ep_losses.append(loss.item())
        history.append(float(np.mean(ep_losses)))

    # 4. Evaluate on validation customers and get window probabilities
    model.eval()
    val_customer_scores = {}
    val_traces = {}
    with torch.no_grad():
        for cid in val_ids:
            if cid not in windows_dict:
                val_customer_scores[cid] = 0.0
                val_traces[cid] = []
                continue
            b = windows_dict[cid]
            v = torch.tensor(b.values, dtype=torch.float32, device=device)
            m = torch.tensor(b.observed, dtype=torch.bool, device=device)
            c = torch.tensor(b.calendar, dtype=torch.float32, device=device)
            logits = model(v, m, c)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_traces[cid] = [
                {"date": d.isoformat(), "prob": float(p)}
                for d, p in zip(b.center_dates, probs)
            ]
            val_customer_scores[cid] = aggregate_customer_score(probs, b.center_dates)

    # 5. Also evaluate on training customers to calibrate threshold and check overfitting
    train_customer_scores = {}
    with torch.no_grad():
        for cid in train_ids:
            if cid not in windows_dict:
                train_customer_scores[cid] = 0.0
                continue
            b = windows_dict[cid]
            v = torch.tensor(b.values, dtype=torch.float32, device=device)
            m = torch.tensor(b.observed, dtype=torch.bool, device=device)
            c = torch.tensor(b.calendar, dtype=torch.float32, device=device)
            logits = model(v, m, c)
            probs = torch.sigmoid(logits).cpu().numpy()
            train_customer_scores[cid] = aggregate_customer_score(probs, b.center_dates)

    # Calibrate threshold on training set
    label_map = labels_df.assign(customer_id=labels_df.customer_id.astype(str)).set_index("customer_id")
    train_y_true = np.array([bool(label_map.loc[c].pv_installed) for c in train_ids])
    train_scores_arr = np.array([train_customer_scores[c] for c in train_ids])

    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(0.2, 0.8, 61):
        preds = train_scores_arr >= thresh
        tp = np.sum((preds == 1) & (train_y_true == 1))
        fp = np.sum((preds == 1) & (train_y_true == 0))
        fn = np.sum((preds == 0) & (train_y_true == 1))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    val_y_true = np.array([bool(label_map.loc[c].pv_installed) for c in val_ids])
    val_scores_arr = np.array([val_customer_scores[c] for c in val_ids])
    val_preds = val_scores_arr >= best_thresh

    # Fold validation metrics
    tp = int(np.sum((val_preds == 1) & (val_y_true == 1)))
    fp = int(np.sum((val_preds == 1) & (val_y_true == 0)))
    fn = int(np.sum((val_preds == 0) & (val_y_true == 1)))
    tn = int(np.sum((val_preds == 0) & (val_y_true == 0)))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)

    fold_res = {
        "fold": fold,
        "calibrated_threshold": best_thresh,
        "train_f1": float(best_f1),
        "val_f1": float(f1),
        "val_precision": float(prec),
        "val_recall": float(rec),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "customer_scores": {c: float(val_customer_scores[c]) for c in val_ids},
        "customer_preds": {c: bool(val_customer_scores[c] >= best_thresh) for c in val_ids},
        "customer_true": {c: bool(label_map.loc[c].pv_installed) for c in val_ids},
    }
    if out_dir is not None:
        torch.save(model.state_dict(), out_dir / f"task1_model_fold_{fold}.pt")
    return fold_res, val_traces, {"train_loss": history}


def main():
    parser = argparse.ArgumentParser(description="Run Task 1 Window Classifier across 5 folds")
    parser.add_argument("--canonical-dir", default="data_private/patchtst_comparison_canonical")
    parser.add_argument("--backend", default="official_patchtst", choices=["official_patchtst", "custom"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--stride-days", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=250)
    parser.add_argument("--out-dir", default="artifacts_private/task1_results")
    args = parser.parse_args()

    canonical = Path(args.canonical_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(canonical / "labels.parquet")
    splits = pd.read_parquet(canonical / "splits.parquet")
    labels["customer_id"] = labels["customer_id"].astype(str)
    splits["customer_id"] = splits["customer_id"].astype(str)

    all_ids = labels["customer_id"].tolist()
    fold_map = dict(zip(splits["customer_id"], splits["fold"]))
    folds = sorted(set(fold_map.values()))

    print(f"=== Running Task 1 Window-Level Classifier ({args.backend}) ===")
    print(f"Total customers: {len(all_ids)} (PV: {labels.pv_installed.sum()}, Non-PV: {(~labels.pv_installed).sum()})")
    print(f"Folds: {len(folds)} (5-fold GroupKFold)")

    # 1. Build windows once
    stride_steps = args.stride_days * 96
    win_cfg = WindowConfig(context_length=672, stride=stride_steps, min_observed_fraction=0.75, max_windows_per_customer=args.max_windows)
    print(f"Extracting 7-day windows (stride={args.stride_days} days, max={args.max_windows}/cust)...")
    t0 = time.time()
    windows_dict = extract_customer_windows(canonical, all_ids, win_cfg)
    total_w = sum(len(b.customer_ids) for b in windows_dict.values())
    print(f"Extracted {total_w} total windows across {len(windows_dict)} customers in {time.time()-t0:.1f}s")

    # 2. Run cross validation
    enc_cfg = EncoderConfig(context_length=672, patch_length=16, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
    all_fold_res = []
    all_traces = {}
    pooled_preds = {}
    pooled_true = {}

    for fold in folds:
        t_fold = time.time()
        val_ids = [c for c in all_ids if fold_map[c] == fold]
        train_ids = [c for c in all_ids if fold_map[c] != fold]

        fold_res, val_traces, meta = train_fold(
            fold=fold,
            train_ids=train_ids,
            val_ids=val_ids,
            windows_dict=windows_dict,
            labels_df=labels,
            enc_cfg=enc_cfg,
            backend=args.backend,
            epochs=args.epochs,
            device="cpu",
            out_dir=out,
        )
        all_fold_res.append(fold_res)
        all_traces.update(val_traces)
        pooled_preds.update(fold_res["customer_preds"])
        pooled_true.update(fold_res["customer_true"])

        print(f"  Fold {fold}: Train F1={fold_res['train_f1']:.3f} | Val F1={fold_res['val_f1']:.3f} (P={fold_res['val_precision']:.2f}, R={fold_res['val_recall']:.2f}) | TP={fold_res['tp']} FP={fold_res['fp']} FN={fold_res['fn']} TN={fold_res['tn']} [{time.time()-t_fold:.1f}s]")

    # 3. Compute Pooled Out-of-Fold Metrics
    y_true_all = np.array([pooled_true[c] for c in all_ids])
    y_pred_all = np.array([pooled_preds[c] for c in all_ids])

    tp_all = int(np.sum((y_pred_all == 1) & (y_true_all == 1)))
    fp_all = int(np.sum((y_pred_all == 1) & (y_true_all == 0)))
    fn_all = int(np.sum((y_pred_all == 0) & (y_true_all == 1)))
    tn_all = int(np.sum((y_pred_all == 0) & (y_true_all == 0)))

    prec_all = tp_all / max(1, tp_all + fp_all)
    rec_all = tp_all / max(1, tp_all + fn_all)
    f1_all = 2 * prec_all * rec_all / max(1e-8, prec_all + rec_all)
    task1_points = f1_all * 70.0

    print("\n" + "="*50)
    print("=== POOLED OUT-OF-FOLD RESULTS (40 CUSTOMERS) ===")
    print("="*50)
    print(f"True Positives:  {tp_all} / 22")
    print(f"False Positives: {fp_all} / 18 (False alarms)")
    print(f"False Negatives: {fn_all} / 22 (Missed PV)")
    print(f"True Negatives:  {tn_all} / 18")
    print(f"Precision:       {prec_all:.4f}")
    print(f"Recall:          {rec_all:.4f}")
    print(f"Task 1 F1 Score: {f1_all:.4f}")
    print(f"Task 1 Points:   {task1_points:.2f} / 70.00")
    print("="*50)

    # Save artifacts
    results_summary = {
        "backend": args.backend,
        "epochs": args.epochs,
        "pooled": {
            "f1": float(f1_all),
            "precision": float(prec_all),
            "recall": float(rec_all),
            "tp": tp_all, "fp": fp_all, "fn": fn_all, "tn": tn_all,
            "task1_score_pts": float(task1_points)
        },
        "folds": all_fold_res
    }
    (out / "task1_summary.json").write_text(json.dumps(results_summary, indent=2))
    (out / "task1_traces.json").write_text(json.dumps(all_traces, indent=2))

    # Save prediction table
    pred_df = pd.DataFrame([
        {
            "customer_id": cid,
            "pv_installed_true": pooled_true[cid],
            "pv_installed_pred": pooled_preds[cid],
            "correct": pooled_true[cid] == pooled_preds[cid]
        }
        for cid in all_ids
    ])
    pred_df.to_csv(out / "task1_predictions.csv", index=False)
    print(f"Saved results and traces to {out}")


if __name__ == "__main__":
    main()
