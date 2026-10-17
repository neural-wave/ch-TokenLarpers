"""Test C: Physical inspection of the false positive customers.
Examines the raw 15-minute readings to see whether they show genuine physical solar depression.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

# Load predictions from Task 1
preds = pd.read_csv("artifacts_private/task1_results/task1_predictions.csv")
false_positives = preds[(preds["pv_installed_true"] == False) & (preds["pv_installed_pred"] == True)]["customer_id"].tolist()
true_negatives = preds[(preds["pv_installed_true"] == False) & (preds["pv_installed_pred"] == False)]["customer_id"].tolist()[:3]
true_positives = preds[(preds["pv_installed_true"] == True) & (preds["pv_installed_pred"] == True)]["customer_id"].tolist()[:3]

print(f"False Positives identified: {false_positives}")

meas_path = "data_private/patchtst_comparison_canonical/measurements.parquet"
df = pd.read_parquet(meas_path, filters=[("customer_id", "in", false_positives + true_negatives + true_positives)])
df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
df["hour"] = df["timestamp_local"].dt.hour
df["month"] = df["timestamp_local"].dt.month
df["year"] = df["timestamp_local"].dt.year
df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")

print("\n" + "="*80)
print("=== TEST C: PHYSICAL KILOWATT-HOUR PROFILE INSPECTION ===")
print("="*80)

def profile_customer(cid, group_name):
    sub = df[df["customer_id"] == cid]
    # Summer vs Winter
    summer = sub[sub["month"].isin([6, 7, 8])]
    winter = sub[sub["month"].isin([12, 1, 2])]
    
    s_night = summer[summer["hour"].between(1, 4)]["energy_kwh"].mean()
    s_midday = summer[summer["hour"].between(11, 14)]["energy_kwh"].mean()
    
    w_night = winter[winter["hour"].between(1, 4)]["energy_kwh"].mean()
    w_midday = winter[winter["hour"].between(11, 14)]["energy_kwh"].mean()
    
    # Check midday drop in summer: is summer midday LESS than summer night?
    summer_drop = s_midday - s_night
    winter_drop = w_midday - w_night
    
    # Check frequency of zero readings in summer daytime (export clipping)
    s_day_zeros = (summer[summer["hour"].between(10, 16)]["energy_kwh"] == 0).mean() * 100
    
    print(f"Customer {cid} [{group_name}]:")
    print(f"  Summer Night (01:00-04:00): {s_night:.2f} kWh  |  Summer Noon (11:00-14:00): {s_midday:.2f} kWh")
    print(f"  Summer Midday Drop:         {summer_drop:+.2f} kWh (Negative indicates solar duck curve!)")
    print(f"  Winter Midday Drop:         {winter_drop:+.2f} kWh")
    print(f"  Summer Zero-Import Rate:    {s_day_zeros:.1f}%")
    print("-" * 80)

for cid in false_positives:
    profile_customer(cid, "FALSE ALARM (Label = No PV, Model = PV)")

print("\nReference True Negatives:")
for cid in true_negatives:
    profile_customer(cid, "TRUE NEGATIVE (Label = No PV, Model = No PV)")

print("\nReference True Positives:")
for cid in true_positives:
    profile_customer(cid, "TRUE POSITIVE (Label = PV, Model = PV)")
