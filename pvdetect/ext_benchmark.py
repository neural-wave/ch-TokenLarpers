"""End-to-end synthetic benchmark of the shipped extension detector (Tier R evidence; dev seeds < 1000).

Per repeat r: a dev-sized labelled training set (`generate_extension_dataset(n_train, 500 + r)`), an
unlabelled-like pool (`600 + r`, ids prefixed "U", labels never read) and a test population (`700 + r`).
`procedure.fit` trains the main model and the nowcaster exactly as `train.py` does; the test population is
then predicted by `pipeline.predict`, so presence, dates and initial kWp come from the main model, not labels.

Arms (same fitted main model per repeat):
- `shipped`: per-customer harmonic change-point test, labelled customers only (no `--ext-unlabelled`), top-3;
- `with_pool`: plus pool pseudo-labels from the fit's own main model (`--ext-unlabelled`);
- `no_topk`: shipped without the global top-3 cap (population rows of `shipped` and `no_topk` are the same
  uncapped per-customer candidates; the cap only acts in the Monte Carlo).

Reported: pooled TP/FP/FN, F1_ext, D_ext, P_ext and bonus over the whole test population, a customer-level
bootstrap 90 % interval of the bonus, per-case counts, and a Monte Carlo of the hidden-test geometry
(1 one-event customer, 1 two-event customer, 8 non-extension PV customers per draw, top-3 applied per draw).

    python -m pvdetect.ext_benchmark --variant A --repeats 3 --out /tmp/ext_bench_A.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from pvdetect import extensions as X
from pvdetect.io import LoadedTimeseries
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.procedure import fit
from pvdetect.scoring import extension_bonus
from pvdetect.synthetic import EXT_CASES, EXT_NEGATIVE_CASES, EXT_POSITIVE_CASES, generate_extension_dataset

ARMS = ("shipped", "with_pool", "no_topk")
ONE_EVENT_CASES = tuple(c for c in EXT_POSITIVE_CASES if c != "double")


def _loaded(ts: pd.DataFrame, prefix: str = "") -> LoadedTimeseries:
    frame = (
        pd.DataFrame(
            {
                "customer_id": prefix + ts["customer_id"].astype(str),
                "epoch_s": ts["timestamp_epoch"].astype(np.int64),
                "energy_kwh": ts["energy_consumption_kwh"].astype(float),
            }
        )
        .drop_duplicates(["customer_id", "epoch_s"])
        .sort_values(["customer_id", "epoch_s"])
        .reset_index(drop=True)
    )
    return LoadedTimeseries(frame=frame, customer_ids=sorted(frame["customer_id"].unique()), report={})


def _labels(labels: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": labels["customer_id"],
            "pv_installed": labels["pv_installed"] == "TRUE",
            "pv_installation_date": pd.to_datetime(labels["pv_installation_date"], errors="coerce"),
            "pv_capacity_kwp": pd.to_numeric(labels["pv_capacity_kwp"], errors="coerce"),
        }
    )


def _ext(ext: pd.DataFrame) -> pd.DataFrame:
    return ext.assign(
        extension_date=pd.to_datetime(ext["extension_date"]),
        pv_capacity_after_extension_kwp=ext["pv_capacity_after_extension_kwp"].astype(float),
    )


def _candidates(nowcaster, table, pred: pd.DataFrame) -> list[X.Event]:
    """Every event of every predicted-PV customer before the global cap (the module's own per-customer path)."""
    out = []
    for r in pred[pred["pv_installed"].astype(bool)].itertuples(index=False):
        cd = table.customers.get(r.customer_id)
        if cd is None or not np.isfinite(r.pv_capacity_kwp):
            continue
        install = pd.Timestamp(r.pv_installation_date).date() if pd.notna(r.pv_installation_date) else None
        out += X.customer_events(nowcaster, r.customer_id, cd, install, float(r.pv_capacity_kwp))
    return out


def _frame(events: list[X.Event], cap: int | None) -> pd.DataFrame:
    ev = sorted(events, key=lambda e: -e.z)[:cap] if cap else list(events)
    return pd.DataFrame(
        [{"customer_id": e.customer_id, "extension_date": pd.Timestamp(e.date), "pv_capacity_after_extension_kwp": e.kwp_after, "z": e.z} for e in ev],
        columns=[*X.COLUMNS, "z"],
    )


def run_repeat(r: int, variant: str, n_train: int, n_pool: int, n_test: int, base: int = 500) -> dict:
    t0 = time.time()
    ts, lab, ext, _ = generate_extension_dataset(n_train, base + r, variant=variant)
    pts, *_ = generate_extension_dataset(n_pool, base + 100 + r, variant=variant)  # pool labels are never read
    tts, tlab, text, tcases = generate_extension_dataset(n_test, base + 200 + r, variant=variant)
    table = build_feature_table(_loaded(ts))
    pool = build_feature_table(_loaded(pts, "U")).customers
    labels, ext_labels = _labels(lab), _ext(ext)
    ids = labels["customer_id"].tolist()
    with_pool = fit(replace(table, extension_pool=pool), labels, ids, ext_labels=ext_labels)
    no_pool = X.fit_extensions(table, labels, ids, ext_labels, with_pool)[1]
    test = build_feature_table(_loaded(tts))
    pred, _ = predict(with_pool, test, test.ids)
    cands = {
        "shipped": _candidates(no_pool, test, pred),
        "with_pool": _candidates(with_pool["_objects"]["catboost_ext"], test, pred),
    }
    cands["no_topk"] = cands["shipped"]
    truth = text.assign(pv_capacity_after_extension_kwp=text["pv_capacity_after_extension_kwp"].astype(float))
    main_tp = int((pred.set_index("customer_id")["pv_installed"] & (tlab.set_index("customer_id")["pv_installed"] == "TRUE")).sum())
    return {
        "repeat": r,
        "secs": round(time.time() - t0, 1),
        "pool_customers": with_pool["extensions"]["pool_customers"],
        "labelled_customers": with_pool["extensions"]["labelled_customers"],
        "main_true_positives": main_tp,
        "cands": {a: _frame(c, None).to_dict("records") for a, c in cands.items()},
        "truth": truth.to_dict("records"),
        "cases": dict(zip(tcases["customer_id"], tcases["case"])),
        "pred_pv": pred.loc[pred["pv_installed"].astype(bool), "customer_id"].tolist(),
    }


def _score(cands: pd.DataFrame, truth: pd.DataFrame, cap: int | None) -> dict:
    sel = cands.sort_values("z", ascending=False).head(cap) if cap else cands
    return extension_bonus(sel.drop(columns="z"), truth)


def summarize(results: list[dict], draws: int = 3000, boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cand = {a: pd.concat([pd.DataFrame(x["cands"][a], columns=[*X.COLUMNS, "z"]).assign(customer_id=lambda d, r=x["repeat"]: f"{r}_" + d["customer_id"]) for x in results]) for a in ARMS}
    truth = pd.concat([pd.DataFrame(x["truth"]).assign(customer_id=lambda d, r=x["repeat"]: f"{r}_" + d["customer_id"]) for x in results])
    cases = {f"{x['repeat']}_{c}": k for x in results for c, k in x["cases"].items()}
    cust = np.array(sorted(cases))
    out = {"pooled": {}, "bootstrap90": {}, "per_case": {}, "hidden_test_mc": {}}
    for a in ARMS:
        c = cand[a]
        out["pooled"][a] = extension_bonus(c.drop(columns="z"), truth)  # population-wide: no global cap
        bs = []
        for _ in range(boot):
            pick = rng.choice(cust, cust.size)
            keys = pd.Series(range(pick.size), index=pick)
            cc = c[c.customer_id.isin(pick)].merge(keys.rename("k"), left_on="customer_id", right_index=True)
            tt = truth[truth.customer_id.isin(pick)].merge(keys.rename("k"), left_on="customer_id", right_index=True)
            cc = cc.assign(customer_id=cc.customer_id + "#" + cc.k.astype(str)).drop(columns="k")
            tt = tt.assign(customer_id=tt.customer_id + "#" + tt.k.astype(str)).drop(columns="k")
            bs.append(extension_bonus(cc.drop(columns="z"), tt)["bonus"])
        out["bootstrap90"][a] = [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))]
        per = {}
        for k in EXT_CASES:
            ids = [i for i, v in cases.items() if v == k]
            b = extension_bonus(c[c.customer_id.isin(ids)].drop(columns="z"), truth[truth.customer_id.isin(ids)])
            per[k] = (b["tp"], b["fp"], b["fn"])
        out["per_case"][a] = per
    one = [i for i, v in cases.items() if v in ONE_EVENT_CASES]
    two = [i for i, v in cases.items() if v == "double"]
    neg = [i for i, v in cases.items() if v in EXT_NEGATIVE_CASES and v != "no_pv"]
    for a in ARMS:
        cap = None if a == "no_topk" else X.MAX_TOTAL
        vals = []
        for _ in range(draws):
            ids = [rng.choice(one), rng.choice(two), *rng.choice(neg, 8, replace=False)]
            b = _score(cand[a][cand[a].customer_id.isin(ids)], truth[truth.customer_id.isin(ids)], cap)
            vals.append((b["bonus"], b["tp"], b["fp"]))
        v = np.array(vals, float)
        out["hidden_test_mc"][a] = {
            "mean_bonus": float(v[:, 0].mean()),
            "p_bonus_positive": float((v[:, 0] > 0).mean()),
            "mean_tp": float(v[:, 1].mean()),
            "mean_fp": float(v[:, 2].mean()),
        }
    return out


def render(variant: str, results: list[dict], s: dict, base: int = 500) -> str:
    lines = [f"### Variant {variant}: {len(results)} repeats (train {base}+r, pool {base + 100}+r, test {base + 200}+r)", ""]
    lines.append(
        f"Training: labelled PV customers per fit {[x['labelled_customers'] for x in results]}, pseudo-labelled "
        f"pool customers {[x['pool_customers'] for x in results]}; seconds per repeat {[x['secs'] for x in results]}."
    )
    lines += ["", "|arm|TP|FP|FN|F1_ext|D_ext|P_ext|bonus (whole population)|bonus 90 % CI|hidden-test MC mean bonus|P(bonus>0)|MC TP|MC FP|", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        p, ci, mc = s["pooled"][a], s["bootstrap90"][a], s["hidden_test_mc"][a]
        lines.append(
            f"|{a}|{p['tp']}|{p['fp']}|{p['fn']}|{p['f1_ext']:.3f}|{p['D_ext']:.3f}|{p['P_ext']:.3f}|{p['bonus']:.2f}|"
            f"[{ci[0]:.2f}, {ci[1]:.2f}]|{mc['mean_bonus']:.2f}|{mc['p_bonus_positive']:.2f}|{mc['mean_tp']:.2f}|{mc['mean_fp']:.2f}|"
        )
    lines += ["", "Per case, shipped arm (TP/FP/FN):", "", "|" + "|".join(EXT_CASES) + "|", "|" + "---|" * len(EXT_CASES)]
    lines.append("|" + "|".join("/".join(map(str, s["per_case"]["shipped"][k])) for k in EXT_CASES) + "|")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=("A", "B"), default="A")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=90)
    ap.add_argument("--seed-base", type=int, default=500, help="train seeds base+r, pool base+100+r, test base+200+r (< 1000: dev)")
    ap.add_argument("--out", help="write raw candidates and the summary as JSON here (outside the repository)")
    args = ap.parse_args(argv)
    results = [run_repeat(r, args.variant, args.n_train, args.n_pool, args.n_test, args.seed_base) for r in range(args.repeats)]
    s = summarize(results)
    print(render(args.variant, results, s, args.seed_base))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results, "summary": s}, f, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
