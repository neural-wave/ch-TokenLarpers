"""Test B: Synthetic Injection Acid Test.
Injects synthetic PV and hard negatives (vacations, factory shutdowns) into real non-PV
baselines using Pipeline B, and tests whether PatchTST can distinguish real solar
physics from non-PV demand reductions.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from neuralwave_data.synthesis import generate_synthetic
from neuralwave_experiment.encoder import EncoderConfig
from neuralwave_experiment.encoder_factory import build_encoder
from neuralwave_experiment.windows import WindowConfig, build_windows
from neuralwave_experiment.task1_detector import (
    WindowClassifier,
    aggregate_customer_score,
)

# 1. Load canonical dataset
canonical = Path("data_private/patchtst_comparison_canonical")
measurements = pd.read_parquet(canonical / "measurements.parquet")
labels = pd.read_parquet(canonical / "labels.parquet")
splits = pd.read_parquet(canonical / "splits.parquet")

# 2. Generate synthetic cases using fold 0 as held-out
print("Generating controlled synthetic test cases with Pipeline B...")
syn_result = generate_synthetic(
    measurements=measurements,
    labels=labels,
    splits=splits,
    held_out_fold=1,  # use fold 1 as held-out so fold 0 training data generates variants
    meter_mode="import_only",
    seed=999,
)

syn_meas = syn_result.measurements
syn_labels = syn_result.synthetic_labels
syn_prov = syn_result.provenance

print(f"Generated {len(syn_labels)} synthetic customers across 4 physical scenarios:")
print(syn_prov["scenario"].value_counts().to_string())

# 3. Load trained PatchTST model (Fold 0)
enc_cfg = EncoderConfig(context_length=672, patch_length=16, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
backbone = build_encoder(enc_cfg, backend="official_patchtst")
model = WindowClassifier(backbone, d_model=enc_cfg.d_model, dropout=enc_cfg.dropout)
model.load_state_dict(torch.load("artifacts_private/task1_results/task1_model_fold_0.pt", map_location="cpu"))
model.eval()

# 4. Build windows for synthetic customers
win_cfg = WindowConfig(context_length=672, stride=96 * 3, min_observed_fraction=0.75, max_windows_per_customer=120)

syn_cids = syn_labels["customer_id"].tolist()
label_map = dict(zip(syn_labels["customer_id"], syn_labels["pv_installed"]))
scenario_map = dict(zip(syn_prov["customer_id"], syn_prov["scenario"]))

results = []
threshold = 0.50

print("\n" + "="*80)
print("=== TEST B: SYNTHETIC ACID TEST EVALUATION ===")
print("="*80)

with torch.no_grad():
    for cid in syn_cids:
        sub = syn_meas[syn_meas["customer_id"] == cid]
        if len(sub) < 672:
            continue
        b = build_windows(sub, win_cfg)
        if len(b.customer_ids) == 0:
            continue
        
        v = torch.tensor(b.values, dtype=torch.float32)
        m = torch.tensor(b.observed, dtype=torch.bool)
        c = torch.tensor(b.calendar, dtype=torch.float32)
        logits = model(v, m, c)
        probs = torch.sigmoid(logits).cpu().numpy()
        score = aggregate_customer_score(probs, b.center_dates)
        pred_pv = score >= threshold
        true_pv = bool(label_map[cid])
        scenario = scenario_map[cid]

        results.append({
            "customer_id": cid,
            "scenario": scenario,
            "true_pv": true_pv,
            "pred_pv": pred_pv,
            "score": float(score),
            "correct": pred_pv == true_pv,
        })

df_res = pd.DataFrame(results)

# Breakdown by scenario
print("\nScenario Performance Breakdown:")
for sc, group in df_res.groupby("scenario"):
    acc = group["correct"].mean() * 100
    mean_score = group["score"].mean()
    expected = "PV" if group["true_pv"].iloc[0] else "No PV"
    print(f"  {sc:<28} (Expected: {expected:<6}): Accuracy = {acc:5.1f}% | Avg Score = {mean_score:.4f} ({len(group)} samples)")

overall_acc = df_res["correct"].mean() * 100
tp = np.sum((df_res["pred_pv"] == 1) & (df_res["true_pv"] == 1))
fp = np.sum((df_res["pred_pv"] == 1) & (df_res["true_pv"] == 0))
fn = np.sum((df_res["pred_pv"] == 0) & (df_res["true_pv"] == 1))
tn = np.sum((df_res["pred_pv"] == 0) & (df_res["true_pv"] == 0))

prec = tp / max(1, tp + fp)
rec = tp / max(1, tp + fn)
f1 = 2 * prec * rec / max(1e-8, prec + rec)

print("\n" + "="*80)
print(f"OVERALL SYNTHETIC ACID TEST ACCURACY: {overall_acc:.1f}%")
print(f"Synthetic Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print("="*80)

Path("artifacts_private/task1_results/test_b_synthetic_results.json").write_text(json.dumps(results, indent=2))
