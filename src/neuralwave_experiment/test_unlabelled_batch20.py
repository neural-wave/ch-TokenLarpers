"""Evaluate trained PatchTST on 20 additional unseen unlabelled Swiss customers (POD79 to POD98)."""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from neuralwave_experiment.encoder import EncoderConfig
from neuralwave_experiment.encoder_factory import build_encoder
from neuralwave_experiment.windows import WindowConfig, build_windows
from neuralwave_experiment.task1_detector import WindowClassifier, aggregate_customer_score

# 1. Define 20 new target customers
target_cids = [f"POD{i}" for i in range(79, 99)]
print(f"Targeting 20 unseen unlabelled customers: {target_cids[0]} to {target_cids[-1]}")

csv_path = "AIL_data_for_neural_wave_v1.1/challenge_set/energy_consumption_unlabelled_dev.csv"
print("Scanning unlabelled dataset for target customers...")
t0 = time.time()
dfs = []

# Stream chunks of 250k rows
chunksize = 250_000
for chunk in pd.read_csv(csv_path, sep=";", chunksize=chunksize, dtype={"customer_id": str}):
    chunk = chunk[chunk["customer_id"].str.startswith("POD", na=False)]
    matched = chunk[chunk["customer_id"].isin(target_cids)]
    if not matched.empty:
        dfs.append(matched)
    # Stop once we have substantial readings for all 20
    total_rows = sum(len(d) for d in dfs)
    if total_rows > 1_500_000:
        break

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.rename(columns={"energy_consumption_kwh": "energy_kwh"})
df_all["value_status"] = "observed"
df_all["timestamp_epoch"] = pd.to_numeric(df_all["timestamp_epoch"], errors="coerce").astype("int64")

print(f"Collected {len(df_all)} readings across {df_all['customer_id'].nunique()} customers in {time.time()-t0:.1f}s")

# 2. Build 7-day windows for each customer
win_cfg = WindowConfig(context_length=672, stride=96 * 3, min_observed_fraction=0.75, max_windows_per_customer=150)
batches = {}
for cid in target_cids:
    sub = df_all[df_all["customer_id"] == cid]
    if len(sub) > 672:
        b = build_windows(sub, win_cfg)
        if len(b.customer_ids) > 0:
            batches[cid] = b
            print(f"  {cid}: {len(b.customer_ids)} windows ({b.center_dates[0]} to {b.center_dates[-1]})")

print(f"Built windows for {len(batches)} / 20 customers")

# 3. Load trained PatchTST model (Fold 0)
enc_cfg = EncoderConfig(context_length=672, patch_length=16, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
backbone = build_encoder(enc_cfg, backend="official_patchtst")
model = WindowClassifier(backbone, d_model=enc_cfg.d_model, dropout=enc_cfg.dropout)
weights_path = Path("artifacts_private/task1_results/task1_model_fold_0.pt")
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

# 4. Predict
threshold = 0.50
results = []

print("\n" + "="*80)
print(f"{'Customer':<10} | {'Decision':<14} | {'Score':<8} | {'Min Prob':<8} | {'Max Prob':<8} | {'Midday kWh':<10} | {'Night kWh':<10} | {'Mid/Night Ratio'}")
print("="*80)

with torch.no_grad():
    for cid in target_cids:
        if cid not in batches:
            print(f"{cid:<10} | Insufficient data")
            continue
        b = batches[cid]
        v = torch.tensor(b.values, dtype=torch.float32)
        m = torch.tensor(b.observed, dtype=torch.bool)
        c = torch.tensor(b.calendar, dtype=torch.float32)
        logits = model(v, m, c)
        probs = torch.sigmoid(logits).cpu().numpy()
        score = aggregate_customer_score(probs, b.center_dates)
        is_pv = score >= threshold

        sub = df_all[df_all["customer_id"] == cid]
        times = pd.to_datetime(sub["timestamp_local"])
        midday_mean = sub[times.dt.hour.between(11, 14)]["energy_kwh"].mean()
        night_mean = sub[times.dt.hour.between(1, 4)]["energy_kwh"].mean()
        ratio = midday_mean / max(1e-4, night_mean)

        decision_str = "PV INSTALLED" if is_pv else "NO PV"
        results.append({
            "customer_id": cid,
            "decision": decision_str,
            "score": float(score),
            "min_prob": float(np.min(probs)),
            "max_prob": float(np.max(probs)),
            "midday_mean": float(midday_mean),
            "night_mean": float(night_mean),
            "mid_night_ratio": float(ratio),
            "n_windows": len(probs),
        })

        print(f"{cid:<10} | {decision_str:<14} | {score:<8.4f} | {np.min(probs):<8.4f} | {np.max(probs):<8.4f} | {midday_mean:<10.2f} | {night_mean:<10.2f} | {ratio:.2f}")

print("="*80)
pv_detected = sum(1 for r in results if r["decision"] == "PV INSTALLED")
print(f"Summary: {pv_detected} / {len(results)} customers ({pv_detected/len(results)*100:.1f}%) classified as PV Installed.")

out_path = Path("artifacts_private/task1_results/unlabelled_batch20_results.json")
out_path.write_text(json.dumps(results, indent=2))
print(f"Saved results to {out_path}")
