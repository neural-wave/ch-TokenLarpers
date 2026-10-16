"""Run trained PatchTST model on unlabelled Swiss customers to test generalization."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from neuralwave_experiment.encoder import EncoderConfig
from neuralwave_experiment.encoder_factory import build_encoder
from neuralwave_experiment.windows import WindowConfig, build_windows
from neuralwave_experiment.task1_detector import WindowClassifier, aggregate_customer_score

# 1. Read a few unlabelled customers
csv_path = "AIL_data_for_neural_wave_v1.1/challenge_set/energy_consumption_unlabelled_dev.csv"
print("Reading sample of unlabelled Swiss utility customers...")
target_cids = ["POD74", "POD75", "POD76", "POD77", "POD78"]
dfs = []

# Stream chunks until we get enough data for these target customers
chunksize = 100_000
for chunk in pd.read_csv(csv_path, sep=";", chunksize=chunksize, dtype={"customer_id": str}):
    chunk = chunk[chunk["customer_id"].str.startswith("POD", na=False)]
    matched = chunk[chunk["customer_id"].isin(target_cids)]
    if not matched.empty:
        dfs.append(matched)
    # Stop once we have substantial history
    total_rows = sum(len(d) for d in dfs)
    if total_rows > 300_000:
        break

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.rename(columns={"energy_consumption_kwh": "energy_kwh"})
df_all["value_status"] = "observed"
df_all["timestamp_epoch"] = pd.to_numeric(df_all["timestamp_epoch"], errors="coerce").astype("int64")

print(f"Collected {len(df_all)} readings across: {df_all['customer_id'].unique().tolist()}")

# 2. Build 7-day windows
win_cfg = WindowConfig(context_length=672, stride=96 * 3, min_observed_fraction=0.75, max_windows_per_customer=150)
batches = {}
for cid in target_cids:
    sub = df_all[df_all["customer_id"] == cid]
    if len(sub) > 672:
        b = build_windows(sub, win_cfg)
        if len(b.customer_ids) > 0:
            batches[cid] = b
            print(f"  {cid}: {len(b.customer_ids)} windows ({b.center_dates[0]} to {b.center_dates[-1]})")

# 3. Load trained model from Fold 0
enc_cfg = EncoderConfig(context_length=672, patch_length=16, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
backbone = build_encoder(enc_cfg, backend="official_patchtst")
model = WindowClassifier(backbone, d_model=enc_cfg.d_model, dropout=enc_cfg.dropout)
weights_path = Path("artifacts_private/task1_results/task1_model_fold_0.pt")
if weights_path.exists():
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    print(f"Loaded trained model weights from {weights_path}")
else:
    raise FileNotFoundError(f"Model weights not found at {weights_path}")

# 4. Predict PV score on each unlabelled customer
print("\n" + "="*50)
print("=== PREDICTIONS ON UNSEEN UNLABELLED CUSTOMERS ===")
print("="*50)

model.eval()
threshold = 0.50  # Calibrated threshold

with torch.no_grad():
    for cid, b in batches.items():
        v = torch.tensor(b.values, dtype=torch.float32)
        m = torch.tensor(b.observed, dtype=torch.bool)
        c = torch.tensor(b.calendar, dtype=torch.float32)
        logits = model(v, m, c)
        probs = torch.sigmoid(logits).cpu().numpy()
        score = aggregate_customer_score(probs, b.center_dates)
        is_pv = score >= threshold
        
        # Check raw midday vs night statistics as physical sanity check
        sub = df_all[df_all["customer_id"] == cid]
        times = pd.to_datetime(sub["timestamp_local"])
        midday_mean = sub[times.dt.hour.between(11, 14)]["energy_kwh"].mean()
        night_mean = sub[times.dt.hour.between(1, 4)]["energy_kwh"].mean()
        
        status = "PV INSTALLED" if is_pv else "NO PV (Standard Demand)"
        print(f"Customer {cid}:")
        print(f"  -> Model Decision:     {status}")
        print(f"  -> PV Activity Score:  {score:.4f} (Threshold = {threshold})")
        print(f"  -> Max Window Prob:    {np.max(probs):.4f}")
        print(f"  -> Min Window Prob:    {np.min(probs):.4f}")
        print(f"  -> Midday avg kWh:     {midday_mean:.2f} | Night avg kWh: {night_mean:.2f}")
        print("-" * 50)
