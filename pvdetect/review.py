"""Per-customer review plots from an OOF report (`train.py --report-dir`). Dev-only (matplotlib)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pvdetect.io import load_timeseries
from pvdetect.pipeline import build_feature_table

REGIME_COLORS = {"event": "tab:red", "present_at_start": "tab:purple", "insufficient": "tab:gray"}


def _vline(ax, date: str, style: str, color: str, label: str) -> None:
    if isinstance(date, str) and date:
        ax.axvline(pd.Timestamp(date), linestyle=style, color=color, label=label)


def plot_customer(row: pd.Series, daily: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = daily[daily["valid"]]
    s = v["s"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(s.index, s, lw=0.5, color="tab:blue", label="s")
    axes[0].plot(s.index, s.rolling(15, center=True, min_periods=5).median(), lw=1.2, color="k", label="s (15d median)")
    axes[0].set_ylabel("log(mid/night)")
    axes[1].plot(v.index, 4 * v["mid"], lw=0.5, color="tab:orange", label="mid kW")
    axes[1].plot(v.index, 4 * v["night"], lw=0.5, color="tab:green", label="night kW")
    axes[1].set_ylabel("kW")
    prev = s.reindex(s.index - pd.Timedelta(days=364)).to_numpy()
    axes[2].plot(s.index, s.to_numpy() - prev, lw=0.5, color="tab:brown", label="s - s(364d ago)")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("YoY")
    color = REGIME_COLORS.get(row.get("pred_regime", ""), "tab:gray")
    for ax in axes:
        _vline(ax, row.get("true_date", ""), "-", "k", "true date")
        _vline(ax, row.get("pred_date", ""), "--", color, f"pred ({row.get('pred_regime', '')})")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{row['customer_id']} true={row['true_pv']} pred={row['pred_pv']} p={float(row['prob']):.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("oof_csv")
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    oof = pd.read_csv(args.oof_csv, sep=";", dtype=str, keep_default_na=False)
    table = build_feature_table(load_timeseries(args.timeseries))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for _, row in oof.iterrows():
        cd = table.customers.get(row["customer_id"])
        if cd is None or cd.n_valid == 0:
            continue
        plot_customer(row, cd.daily, out / f"{row['customer_id']}.png")
        n += 1
    print(f"wrote {n} plots to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
