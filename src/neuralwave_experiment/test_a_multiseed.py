"""Test A: Multi-Seed Stability Test (5 Independent Seeds).
Measures whether the F1 score is stable across 5 different random seeds,
saving progress incrementally after each seed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from neuralwave_experiment.encoder import EncoderConfig
from neuralwave_experiment.windows import WindowConfig
from neuralwave_experiment.task1_detector import (
    extract_customer_windows,
    train_fold,
)

canonical = Path("data_private/patchtst_comparison_canonical")
labels = pd.read_parquet(canonical / "labels.parquet")
splits = pd.read_parquet(canonical / "splits.parquet")
labels["customer_id"] = labels["customer_id"].astype(str)
splits["customer_id"] = splits["customer_id"].astype(str)

all_ids = labels["customer_id"].tolist()
fold_map = dict(zip(splits["customer_id"], splits["fold"]))
folds = sorted(set(fold_map.values()))

out_file = Path("artifacts_private/task1_results/test_a_multiseed_results.json")
out_file.parent.mkdir(parents=True, exist_ok=True)

# Pre-populate completed seeds 17 and 42 from previous verified run
existing_results = [
    {
        "seed": 17, "f1": 0.9361702127659575, "precision": 0.88, "recall": 1.0,
        "tp": 22, "fp": 3, "fn": 0, "tn": 15, "mean_fold_f1": 0.937, "time_s": 199.6
    },
    {
        "seed": 42, "f1": 0.9166666666666666, "precision": 0.8461538461538461, "recall": 1.0,
        "tp": 22, "fp": 4, "fn": 0, "tn": 14, "mean_fold_f1": 0.920, "time_s": 208.7
    }
]

completed_seeds = {r["seed"] for r in existing_results}
all_target_seeds = [17, 42, 100, 777, 999]
remaining_seeds = [s for s in all_target_seeds if s not in completed_seeds]

print(f"Total target seeds: {all_target_seeds}")
print(f"Already completed:  {list(completed_seeds)}")
print(f"Running remaining:  {remaining_seeds}")

# Extract windows once
stride_steps = 3 * 96
win_cfg = WindowConfig(context_length=672, stride=stride_steps, min_observed_fraction=0.75, max_windows_per_customer=120)
print("\nExtracting 7-day windows...")
windows_dict = extract_customer_windows(canonical, all_ids, win_cfg)

enc_cfg = EncoderConfig(context_length=672, patch_length=16, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
seed_results = list(existing_results)

print("\n" + "="*80)
print("=== TEST A: MULTI-SEED STABILITY BENCHMARK (5 SEEDS) ===")
print("="*80)
for r in existing_results:
    print(f"Seed {r['seed']:<5}: Pooled F1 = {r['f1']:.4f} (P = {r['precision']:.3f}, R = {r['recall']:.3f}) | TP={r['tp']:<2} FP={r['fp']:<2} FN={r['fn']:<2} TN={r['tn']:<2} | Mean Fold F1 = {r['mean_fold_f1']:.3f} (Cached)")

for s in remaining_seeds:
    t0 = time.time()
    pooled_preds = {}
    pooled_true = {}
    fold_f1s = []

    for fold in folds:
        val_ids = [c for c in all_ids if fold_map[c] == fold]
        train_ids = [c for c in all_ids if fold_map[c] != fold]

        fold_res, _, _ = train_fold(
            fold=fold,
            train_ids=train_ids,
            val_ids=val_ids,
            windows_dict=windows_dict,
            labels_df=labels,
            enc_cfg=enc_cfg,
            backend="official_patchtst",
            epochs=5,
            device="cpu",
            seed=s,
        )
        fold_f1s.append(fold_res["val_f1"])
        pooled_preds.update(fold_res["customer_preds"])
        pooled_true.update(fold_res["customer_true"])

    y_true_all = np.array([pooled_true[c] for c in all_ids])
    y_pred_all = np.array([pooled_preds[c] for c in all_ids])

    tp = int(np.sum((y_pred_all == 1) & (y_true_all == 1)))
    fp = int(np.sum((y_pred_all == 1) & (y_true_all == 0)))
    fn = int(np.sum((y_pred_all == 0) & (y_true_all == 1)))
    tn = int(np.sum((y_pred_all == 0) & (y_true_all == 0)))

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    
    elapsed = time.time() - t0
    res_entry = {
        "seed": s, "f1": f1, "precision": prec, "recall": rec,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mean_fold_f1": float(np.mean(fold_f1s)), "time_s": elapsed
    }
    seed_results.append(res_entry)
    out_file.write_text(json.dumps(seed_results, indent=2))

    print(f"Seed {s:<5}: Pooled F1 = {f1:.4f} (P = {prec:.3f}, R = {rec:.3f}) | TP={tp:<2} FP={fp:<2} FN={fn:<2} TN={tn:<2} | Mean Fold F1 = {np.mean(fold_f1s):.3f} [{elapsed:.1f}s]")

f1s = [r["f1"] for r in seed_results]
precs = [r["precision"] for r in seed_results]
recs = [r["recall"] for r in seed_results]

print("="*80)
print(f"MULTI-SEED SUMMARY (across all {len(seed_results)} random seeds):")
print(f"  F1 Score:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  (Min: {np.min(f1s):.4f}, Max: {np.max(f1s):.4f})")
print(f"  Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
print(f"  Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
print("="*80)
print(f"Saved complete 5-seed benchmark to {out_file}")
