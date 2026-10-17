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
from neuralwave_experiment.oof_contract import OOFRow, load_split_files, write_oof_predictions
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


def aggregate_customer_score(probs: np.ndarray, dates: list[date], k: int = 4) -> float:
    """High-inertia customer PV score from time series of window probabilities.
    
    Uses the maximum rolling median over k consecutive windows (approx 12-14 days).
    This detects sustained solar adoption while preventing single-week vacations
    from triggering false alarms, and eliminates the structural blindspot for
    late-calendar installations.
    """
    if len(probs) == 0:
        return 0.0
    if len(probs) < k:
        return float(np.max(probs))
    rolling = [np.median(probs[i : i + k]) for i in range(len(probs) - k + 1)]
    return float(np.max(rolling))


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


def _pooled_metrics(all_ids: list[str], pooled_true: dict[str, bool], pooled_preds: dict[str, bool]) -> dict[str, float | int]:
    y_true = np.array([pooled_true[c] for c in all_ids])
    y_pred = np.array([pooled_preds[c] for c in all_ids])
    tp = int(np.sum(y_pred & y_true)); fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true)); tn = int(np.sum(~y_pred & ~y_true))
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    return {"f1": float(f1), "precision": float(prec), "recall": float(rec),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "task1_score_pts": float(f1 * 70.0)}


def run_repeat(repeat: int, fold_map: dict[str, int], all_ids: list[str], windows_dict: dict[str, WindowBatch],
               labels: pd.DataFrame, enc_cfg: EncoderConfig, backend: str, epochs: int, device: str,
               seed: int, out: Path) -> tuple[dict[str, Any], dict[str, Any], list[OOFRow]]:
    """Train and score every fold of one split file (one outer repeat).

    Checkpoints are written only for repeat 0 (``task1_model_fold_<k>.pt``).
    """
    folds = sorted(set(fold_map.values()))
    fold_results, traces, oof_rows = [], {}, []
    pooled_preds: dict[str, bool] = {}
    pooled_true: dict[str, bool] = {}
    for fold in folds:
        t_fold = time.time()
        val_ids = [c for c in all_ids if fold_map[c] == fold]
        train_ids = [c for c in all_ids if fold_map[c] != fold]
        fold_res, val_traces, _ = train_fold(
            fold=fold, train_ids=train_ids, val_ids=val_ids, windows_dict=windows_dict, labels_df=labels,
            enc_cfg=enc_cfg, backend=backend, epochs=epochs, device=device, seed=seed + 1000 * repeat,
            out_dir=out if repeat == 0 else None,  # checkpoints of the first repeat serve wild inference
        )
        fold_results.append(fold_res)
        traces.update(val_traces)
        pooled_preds.update(fold_res["customer_preds"])
        pooled_true.update(fold_res["customer_true"])
        for cid in val_ids:
            oof_rows.append(OOFRow(customer_id=cid, repeat=repeat, fold=int(fold),
                                   prob=float(fold_res["customer_scores"][cid]),
                                   pv_installed=bool(fold_res["customer_preds"][cid])))
        print(f"  repeat {repeat} fold {fold}: Train F1={fold_res['train_f1']:.3f} | Val F1={fold_res['val_f1']:.3f} "
              f"(P={fold_res['val_precision']:.2f}, R={fold_res['val_recall']:.2f}) | TP={fold_res['tp']} FP={fold_res['fp']} "
              f"FN={fold_res['fn']} TN={fold_res['tn']} [{time.time() - t_fold:.1f}s]")
    pooled = _pooled_metrics(all_ids, pooled_true, pooled_preds)
    summary = {"repeat": repeat, "pooled": pooled, "folds": fold_results,
               "pooled_true": pooled_true, "pooled_preds": pooled_preds}
    return summary, traces, oof_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Task 1 window classifier over customer-disjoint folds")
    parser.add_argument("--canonical-dir", default="data_private/patchtst_comparison_canonical")
    parser.add_argument("--splits", default=None,
                        help="comma-separated split files (parquet or CSV with customer_id,fold); each is one outer repeat. "
                             "Default: <canonical-dir>/splits.parquet")
    parser.add_argument("--oof-out", default=None,
                        help="write the shared out-of-fold prediction file (customer_id;repeat;fold;prob;pv_installed;...)")
    parser.add_argument("--backend", default="official_patchtst", choices=["official_patchtst", "custom"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--stride-days", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="artifacts_private/task1_results")
    args = parser.parse_args(argv)

    canonical = Path(args.canonical_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(canonical / "labels.parquet")
    labels["customer_id"] = labels["customer_id"].astype(str)
    all_ids = labels["customer_id"].tolist()
    n_pos = int(labels.pv_installed.sum()); n_neg = int(len(labels) - n_pos)
    split_paths = [p.strip() for p in args.splits.split(",")] if args.splits else [str(canonical / "splits.parquet")]
    fold_maps = load_split_files(split_paths, all_ids)

    print(f"=== Running Task 1 Window-Level Classifier ({args.backend}) ===")
    print(f"Total customers: {len(all_ids)} (PV: {n_pos}, Non-PV: {n_neg})")
    print(f"Repeats: {len(fold_maps)} split file(s); folds per repeat: {[len(set(m.values())) for m in fold_maps]}")

    # 1. Build windows once (label-free, shared by every repeat)
    stride_steps = args.stride_days * 96
    win_cfg = WindowConfig(context_length=TASK1_ENCODER_CONFIG.context_length, stride=stride_steps,
                           min_observed_fraction=0.75, max_windows_per_customer=args.max_windows)
    print(f"Extracting 7-day windows (stride={args.stride_days} days, max={args.max_windows}/cust)...")
    t0 = time.time()
    windows_dict = extract_customer_windows(canonical, all_ids, win_cfg)
    total_w = sum(len(b.customer_ids) for b in windows_dict.values())
    print(f"Extracted {total_w} total windows across {len(windows_dict)} customers in {time.time() - t0:.1f}s")

    # 2. One nested run per split file
    repeat_summaries, all_traces, all_oof = [], {}, []
    for repeat, fold_map in enumerate(fold_maps):
        summary, traces, oof_rows = run_repeat(repeat, fold_map, all_ids, windows_dict, labels, TASK1_ENCODER_CONFIG,
                                               args.backend, args.epochs, args.device, args.seed, out)
        pooled = summary["pooled"]
        print(f"repeat {repeat}: pooled F1={pooled['f1']:.4f} precision={pooled['precision']:.4f} recall={pooled['recall']:.4f} "
              f"TP={pooled['tp']}/{n_pos} FP={pooled['fp']}/{n_neg} FN={pooled['fn']} TN={pooled['tn']}")
        repeat_summaries.append(summary)
        all_traces[f"repeat_{repeat}"] = traces
        all_oof.extend(oof_rows)

    f1s = np.array([s["pooled"]["f1"] for s in repeat_summaries], dtype=float)
    print("\n" + "=" * 50)
    print(f"=== POOLED OUT-OF-FOLD RESULTS ({len(all_ids)} CUSTOMERS, {len(f1s)} REPEAT(S)) ===")
    print("=" * 50)
    print(f"Task 1 F1 mean:  {f1s.mean():.4f} ± {(f1s.std(ddof=1) if len(f1s) > 1 else 0.0):.4f}")
    print(f"Task 1 Points:   {70.0 * f1s.mean():.2f} / 70.00")
    print("=" * 50)

    first = repeat_summaries[0]
    results_summary = {
        "backend": args.backend, "epochs": args.epochs, "seed": args.seed,
        "split_files": split_paths, "n_repeats": len(repeat_summaries),
        "pooled": first["pooled"], "folds": first["folds"],
        "repeats": [{"repeat": s["repeat"], "pooled": s["pooled"], "folds": s["folds"]} for s in repeat_summaries],
        "pooled_f1_mean": float(f1s.mean()),
        "pooled_f1_sd": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
    }
    (out / "task1_summary.json").write_text(json.dumps(results_summary, indent=2))
    (out / "task1_traces.json").write_text(json.dumps(all_traces, indent=2))

    pred_df = pd.DataFrame([
        {"customer_id": cid, "repeat": s["repeat"], "pv_installed_true": s["pooled_true"][cid],
         "pv_installed_pred": s["pooled_preds"][cid], "correct": s["pooled_true"][cid] == s["pooled_preds"][cid]}
        for s in repeat_summaries for cid in all_ids
    ])
    pred_df.to_csv(out / "task1_predictions.csv", index=False)

    if args.oof_out:
        path = write_oof_predictions(all_oof, args.oof_out)
        print(f"Wrote shared out-of-fold predictions to {path}")
    print(f"Saved results and traces to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
