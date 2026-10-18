"""Train the PV model; `--cv` reports a nested customer-level CV estimate of the whole procedure."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pvdetect.evaluation import REGIMES, count_decisions, load_split_file, nested_cv, outer_splits, splits_from_fold_map
from pvdetect.io import load_extension_labels, load_labels, load_timeseries, split_by_customer
from pvdetect.model import save_model
from pvdetect.output import write_oof
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.procedure import C_OPTIONS, DEFAULT_OPTIONS, WEEKLY_KINDS, export_enabled_for, fit
from pvdetect.unlabelled import POOL_MIN_VALID_DAYS, accumulate_file, build_pool, build_unlabelled_prior

REPO = Path(__file__).resolve().parent


def _fmt(x: float, nd: int = 3) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _selection_line(infos: list[dict]) -> str:
    n = len(infos)
    clf = Counter(i.get("classifier") for i in infos)
    cap = Counter(i.get("capacity") for i in infos)
    th = np.array([i.get("threshold", np.nan) for i in infos], float)
    ka = np.array([i.get("kappa", np.nan) for i in infos], float)
    off = np.array([i.get("offset_event", np.nan) for i in infos], float)
    rule = Counter(i.get("insufficient_rule") for i in infos if i.get("insufficient_rule"))
    clf_part = f"classifier logreg={clf.get('logreg', 0)}/{n} catboost={clf.get('catboost', 0)}/{n} avg={clf.get('avg', 0)}/{n}"
    if any("inner_score_weekly" in i for i in infos) or any(clf.get(k) for k in WEEKLY_KINDS):
        clf_part += "".join(f" {k}={clf.get(k, 0)}/{n}" for k in WEEKLY_KINDS)
    parts = [clf_part]
    cs = Counter(i.get("logreg_C") for i in infos if i.get("logreg_C") is not None)
    if cs:
        parts.append("logreg_C " + " ".join(f"{c}={cs.get(c, 0)}/{n}" for c in C_OPTIONS))
    parts += [
        f"capacity ridge={cap.get('ridge', 0)}/{n} catboost={cap.get('catboost', 0)}/{n}",
        f"threshold median={_fmt(np.nanmedian(th), 2)} [{_fmt(np.nanmin(th), 2)},{_fmt(np.nanmax(th), 2)}]",
        f"kappa median={_fmt(np.nanmedian(ka), 2)}",
        f"offset median={_fmt(np.nanmedian(off), 1)}",
    ]
    if rule:
        parts.append("insufficient_rule " + " ".join(f"{k}={v}/{n}" for k, v in sorted(rule.items())))
    clim = Counter(i.get("climatology") for i in infos if i.get("climatology"))
    if clim:
        parts.append(
            f"climatology labelled={clim.get('labelled_nonpv', 0)}/{n} unlabelled={clim.get('unlabelled_0.5', 0)}/{n}"
        )
    scale = Counter(i.get("evidence_scale") for i in infos if i.get("evidence_scale"))
    if scale:
        parts.append(f"evidence_scale none={scale.get('none', 0)}/{n} spread={scale.get('unlabelled_spread', 0)}/{n}")
    block = Counter(i.get("feature_block") for i in infos if i.get("feature_block"))
    if block:
        parts.append(f"features base={block.get('base', 0)}/{n} persistence={block.get('persistence', 0)}/{n}")
    for key in ("schedule", "sun", "recent", "smallpv"):
        got = Counter(i.get(f"{key}_block") for i in infos if i.get(f"{key}_block"))
        if got:
            parts.append(f"{key} off={got.get('off', 0)}/{n} on={got.get('on', 0)}/{n}")
    pseudo = Counter(i.get("pseudo") for i in infos if i.get("pseudo"))
    if pseudo:
        parts.append(f"pseudo off={pseudo.get('off', 0)}/{n} on={pseudo.get('on', 0)}/{n}")
    for key in ("offset_tri", "recur_strict", "onset_first", "tail_guard", "date_debias", "cap_ceiling"):
        got = Counter(i.get(key) for i in infos if i.get(key))
        if got:
            parts.append(f"{key} off={got.get('off', 0)}/{n} on={got.get('on', 0)}/{n}")
    return "selection: " + " | ".join(parts)


def _attach_unlabelled(table, dev_ids: list[str], args):
    """Stream the unlabelled file (dev customers excluded), build the label-free prior and, unless
    `--pseudo-pool 0`, the pseudo-labelling pool (a second pass); attach both."""
    dailies, rep = accumulate_file(
        args.unlabelled,
        chunksize=args.unlabelled_chunksize,
        id_prefix=args.unlabelled_id_prefix,
        exclude_ids=set(dev_ids),
    )
    prior = build_unlabelled_prior(dailies)
    pool = {}
    if args.pseudo_pool > 0:
        pool = build_pool(args.unlabelled, dailies, args.pseudo_pool, chunksize=args.unlabelled_chunksize)
    del dailies
    out = Path(args.out).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "clim_unlabelled.json").write_text(json.dumps(prior.to_json(), indent=2))
    option = "enabled" if prior.passed else f"disabled reason={prior.reason}"
    print(
        f"unlabelled: customers={prior.n_customers} used_q50={prior.n_q[0.5]} sensitivity={prior.sensitivity:.3f} "
        f"option={option} rows_excluded_labelled={rep['rows_excluded_labelled']}"
    )
    if args.pseudo_pool > 0:
        print(f"pseudo_pool: customers={len(pool)} max={args.pseudo_pool} min_valid_days={POOL_MIN_VALID_DAYS}")
    return replace(table, unlabelled=prior, unlabelled_pool=pool or None)


def _attach_extension_pool(table, dev_ids: list[str], args):
    """Stream `--ext-unlabelled` (dev customers excluded) and attach up to `--ext-pool` customers with ≥365
    valid days as the extension nowcaster's label-free pool; the main model's options are unaffected."""
    dailies, _ = accumulate_file(
        args.ext_unlabelled,
        chunksize=args.unlabelled_chunksize,
        id_prefix=args.unlabelled_id_prefix,
        exclude_ids=set(dev_ids),
    )
    pool = build_pool(args.ext_unlabelled, dailies, args.ext_pool, chunksize=args.unlabelled_chunksize)
    del dailies
    print(f"ext_pool: customers={len(pool)} max={args.ext_pool} min_valid_days={POOL_MIN_VALID_DAYS}")
    return replace(table, extension_pool=pool or None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext-labels")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--outer-repeats", type=int, default=4)
    ap.add_argument("--report-dir")
    ap.add_argument("--folds", help="comma-separated split files (customer_id, fold); each file is one outer repeat")
    ap.add_argument("--oof-out", help="write the shared out-of-fold prediction file (requires --cv)")
    ap.add_argument("--embedding-checkpoint", help="frozen encoder artifact directory")
    ap.add_argument("--feature-block", choices=("auto", "baseline", "embedding"), default="auto")
    ap.add_argument("--embedding-device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--unlabelled", help="unlabelled timeseries CSV for label-free Tier A options (Step 18)")
    ap.add_argument("--unlabelled-chunksize", type=int, default=500_000)
    ap.add_argument("--unlabelled-id-prefix", default="POD", help='keep only ids with this prefix ("" keeps all)')
    ap.add_argument("--no-e1e2", action="store_true", help="drop the E1 persistence and E2 avg options (baseline)")
    ap.add_argument("--no-weekly", action="store_true", help="drop the weekly-evidence head options (weekly, avg_weekly)")
    ap.add_argument("--no-ctune", action="store_true", help="keep the logreg C at its default (no inner-CV C choice)")
    ap.add_argument("--no-offset-tri", action="store_true", help="keep the shrunk-median date offsets (no inner-CV choice)")
    ap.add_argument("--no-recur-strict", action="store_true", help="drop the strict short-series detection option")
    ap.add_argument("--no-recent", action="store_true", help="drop the recent-evidence feature-block option")
    ap.add_argument("--no-smallpv", action="store_true", help="drop the scale-free small-PV feature-block option")
    ap.add_argument("--no-date-debias", action="store_true", help="drop the per-regime residual date-bias option")
    ap.add_argument("--no-cap-ceiling", action="store_true", help="drop the learned capacity-band option")
    ap.add_argument("--no-onset-first", action="store_true", help="drop the onset-first date option (earlier-year recurrence + earliest-qualifying-candidate selection)")
    ap.add_argument("--no-tail-guard", action="store_true", help="drop the tail-guard detection option (≥45 valid days after τ, ≥2 post months)")
    ap.add_argument("--schedule", action="store_true", help="add the schedule-robust feature-block option")
    ap.add_argument("--sun", action="store_true", help="add the sunshine-coupling feature-block option")
    ap.add_argument("--fraunhofer", action="store_true", help="add the Fraunhofer CNN+MLP candidate to inner CV")
    ap.add_argument(
        "--pseudo-pool",
        type=int,
        default=300,
        help="unlabelled customers (≥365 valid days) for the pseudo-labelling option; 0 disables; needs --unlabelled",
    )
    ap.add_argument("--no-extensions", action="store_true", help="header-only extension file (no nowcaster)")
    ap.add_argument(
        "--ext-unlabelled",
        help="not recommended (lowered recall in the synthetic benchmark): unlabelled CSV whose PV customers "
        "(main-model P ≥ 0.95, per fit) pseudo-label the extension nowcaster; changes no main-model option",
    )
    ap.add_argument("--ext-pool", type=int, default=400, help="max unlabelled customers for --ext-unlabelled")
    ap.add_argument(
        "--split-dir",
        help="split --timeseries by customer into DIR/training.csv and DIR/test.csv, then train on training.csv only",
    )
    ap.add_argument("--split-frac", type=float, default=0.75, help="share of customers in training.csv")
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.folds and not args.cv:
        ap.error("--folds requires --cv")
    dropped = (
        ({"e1", "e2"} if args.no_e1e2 else set())
        | ({"weekly"} if args.no_weekly else set())
        | ({"ctune"} if args.no_ctune else set())
        | ({"offset_tri"} if args.no_offset_tri else set())
        | ({"recur_strict"} if args.no_recur_strict else set())
        | ({"recent"} if args.no_recent else set())
        | ({"smallpv"} if args.no_smallpv else set())
        | ({"date_debias"} if args.no_date_debias else set())
        | ({"cap_ceiling"} if args.no_cap_ceiling else set())
        | ({"onset_first"} if args.no_onset_first else set())
        | ({"tail_guard"} if args.no_tail_guard else set())
    )
    options = DEFAULT_OPTIONS - dropped | {o for o in ("schedule", "sun", "fraunhofer") if getattr(args, o)}
    if args.oof_out and not args.cv:
        ap.error("--oof-out requires --cv")

    timeseries, input_ids, held_out = args.timeseries, None, set()
    if args.split_dir:
        split_dir = Path(args.split_dir).expanduser().resolve()
        if split_dir == REPO or REPO in split_dir.parents:
            ap.error("--split-dir is inside the repository; choose a private folder outside it")
        train_ids, test_ids = split_by_customer(
            args.timeseries, split_dir, train_frac=args.split_frac, seed=args.split_seed
        )
        input_ids, held_out = train_ids + test_ids, set(test_ids)
        timeseries = split_dir / "training.csv"
        print(f"split: customers={len(input_ids)} training={len(train_ids)} test={len(test_ids)} seed={args.split_seed}")

    loaded = load_timeseries(timeseries)
    labels = load_labels(args.labels)
    labels = labels[~labels["customer_id"].isin(held_out)].reset_index(drop=True)
    missing = set(labels["customer_id"]) - set(loaded.customer_ids)
    if missing:
        raise ValueError(f"{len(missing)} labelled customers missing from timeseries")
    ext_labels = load_extension_labels(args.ext_labels) if args.ext_labels else None
    if ext_labels is not None:
        ext_labels = ext_labels[~ext_labels["customer_id"].isin(held_out)].reset_index(drop=True)
    ids = labels["customer_id"].tolist()
    fold_maps = [load_split_file(p, ids) for p in args.folds.split(",")] if args.folds else None
    table = build_feature_table(loaded)
    encoder = None
    if args.embedding_checkpoint:
        from pvdetect.embedding import FrozenEncoder

        encoder = FrozenEncoder.load(args.embedding_checkpoint, device=args.embedding_device)
    rep = loaded.report
    dl = rep["day_length_counts"]
    signed = sum(1 for c in ids if table.customers[c] is not None and table.customers[c].neg_frac >= 0.01)
    print(
        f"data: customers={rep['customers']} labelled={len(ids)} customers_signed={signed}/{len(ids)} "
        f"export_enabled={str(export_enabled_for(table, ids)).lower()} days92={dl['92']} days100={dl['100']} "
        f"rows_total={rep['rows_total']} rows_bad_epoch={rep['rows_bad_epoch']} "
        f"rows_bad_energy={rep['rows_bad_energy']} rows_duplicate={rep['rows_duplicate']}"
    )
    if args.unlabelled:
        table = _attach_unlabelled(table, input_ids or loaded.customer_ids, args)
    if args.ext_unlabelled and not args.no_extensions:
        table = _attach_extension_pool(table, input_ids or loaded.customer_ids, args)
    ext_kw = {"ext_labels": ext_labels, "extensions": not args.no_extensions}

    if args.cv:
        if fold_maps is not None:
            splits = [splits_from_fold_map(m, ids) for m in fold_maps]
        else:
            try:
                splits = outer_splits(labels, repeats=args.outer_repeats)
            except ValueError:
                print("cv skipped: fewer than 2 customers in a class")
                splits = None
        if splits is not None:
            fit_fn = lambda t, l, train_ids: fit(
                t,
                l,
                train_ids,
                options=options,
                encoder=encoder,
                feature_block=args.feature_block,
                **ext_kw,
            )
            res = nested_cv(fit_fn, predict, table, labels, splits, ext_labels=ext_labels)
            k = count_decisions(REPO / "DECISIONS.md")
            biased = " selection-biased" if k > 0 else ""
            inner_scores = [x["inner_score"] for x in res.fit_infos if np.isfinite(x.get("inner_score", np.nan))]
            inner = float(np.mean(inner_scores)) if inner_scores else float("nan")
            print(
                f"nested: score={res.mean('score'):.2f}±{res.sd('score'):.2f} "
                f"ci90=[{res.ci90[0]:.2f},{res.ci90[1]:.2f}] f1={res.mean('f1'):.3f} "
                f"D={res.mean('D'):.3f} P={res.mean('P'):.3f} inner={_fmt(inner, 2)} gap={_fmt(inner - res.mean('score'), 2)} "
                f"outer_informed_decisions={k}{biased}"
            )
            # Exact split of 100 − score = 100·(1−F1) + 15·F1·(1−D) + 15·F1·(1−P), averaged over repeats.
            lost = {
                "detection": np.mean([100 * (1 - s["f1"]) for s in res.repeat_scores]),
                "dates": np.mean([15 * s["f1"] * (1 - s["D"]) for s in res.repeat_scores]),
                "capacity": np.mean([15 * s["f1"] * (1 - s["P"]) for s in res.repeat_scores]),
            }
            print(
                "points_lost: "
                + " ".join(f"{k}={v:.2f}" for k, v in lost.items())
                + f" largest={max(lost, key=lost.get)}"
            )
            by = {r: [x["by_regime"][r] for x in res.regimes] for r in REGIMES}
            print(
                "dates: "
                + " | ".join(
                    f"{r} n={np.mean([b['n'] for b in by[r]]):.1f} D={_fmt(np.nanmean([b['D'] for b in by[r]]) if any(b['n'] for b in by[r]) else np.nan)}"
                    for r in REGIMES
                )
            )
            conf = res.regimes[0]["confusion"]
            short = {"event": "event", "present_at_start": "start", "insufficient": "insuff"}
            print(
                "regime_confusion: "
                + " ".join(f"{short[k.split('->')[0]]}->{short[k.split('->')[1]]}={v}" for k, v in conf.items())
            )
            if res.ext_scores:
                b = res.ext_scores
                print(
                    f"ext: bonus={np.mean([x['bonus'] for x in b]):.2f} f1_ext={np.mean([x['f1_ext'] for x in b]):.3f} "
                    f"D_ext={np.mean([x['D_ext'] for x in b]):.3f} P_ext={np.mean([x['P_ext'] for x in b]):.3f} "
                    f"tp={np.mean([x['tp'] for x in b]):.2f} fp={np.mean([x['fp'] for x in b]):.2f} "
                    f"fn={np.mean([x['fn'] for x in b]):.2f} (means over repeats; top-3 cap applies per outer fold)"
                )
            print(_selection_line(res.fit_infos))
            if args.report_dir:
                _write_report(Path(args.report_dir), res, labels)
            if args.oof_out:
                if fold_maps is None:
                    fold_maps = [{c: j for j, (_, test) in enumerate(r) for c in test} for r in splits]
                write_oof(
                    pd.concat(
                        [p.assign(repeat=r, fold=p["customer_id"].map(fold_maps[r])) for r, p in enumerate(res.oof)],
                        ignore_index=True,
                    ),
                    Path(args.oof_out),
                )

    model = fit(
        table,
        labels,
        ids,
        options=options,
        encoder=encoder,
        feature_block=args.feature_block,
        **ext_kw,
    )
    save_model(model, args.out)
    sel = model["selection"]
    print(
        f"final: classifier={sel['classifier']} threshold={sel['threshold']:.2f} capacity={sel['capacity']} "
        f"kappa={sel['kappa']:.2f} offset_days={sel['offset_event']} start_lag_days={sel.get('start_lag', 0)}"
    )
    ext = model["extensions"]
    print(
        f"extensions: enabled={str(ext.get('enabled')).lower()} labelled_customers={ext.get('labelled_customers', 0)} "
        f"pool_customers={ext.get('pool_customers', 0)} windows={ext.get('windows', 0)}"
        + (f" reason={ext['reason']}" if ext.get("reason") else "")
    )
    return 0


def _write_report(out: Path, res, labels: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)
    p = res.oof[0]
    t = labels.set_index("customer_id").loc[p["customer_id"]].reset_index()
    rep = pd.DataFrame(
        {
            "customer_id": p["customer_id"],
            "true_pv": t["pv_installed"].map({True: "TRUE", False: "FALSE"}),
            "prob": p["prob"].round(4),
            "pred_pv": p["pv_installed"].map({True: "TRUE", False: "FALSE"}),
            "true_date": t["pv_installation_date"].dt.strftime("%Y-%m-%d"),
            "pred_date": p["raw_date"].dt.strftime("%Y-%m-%d"),
            "date_err_days": (p["raw_date"] - t["pv_installation_date"]).dt.days,
            "pred_regime": p["regime"],
            "true_regime": np.where(
                ~t["pv_installed"],
                "",
                np.where(t["pv_installation_date"] < p["first_date"], "present_at_start", "event"),
            ),
            "true_kwp": t["pv_capacity_kwp"],
            "pred_kwp": p["raw_kwp"].round(2),
            "ape": ((p["raw_kwp"] - t["pv_capacity_kwp"]).abs() / t["pv_capacity_kwp"]).round(4),
            "cp_found": p["cp_found"],
        }
    )
    feats = p.drop(columns=[c for c in p.columns if c in set(rep.columns) | {
        "pv_installed", "pv_installation_date", "pv_capacity_kwp", "regime", "first_date", "raw_date", "raw_kwp"}])
    pd.concat([rep, feats.reset_index(drop=True)], axis=1).to_csv(out / "oof_repeat0.csv", sep=";", index=False)
    (out / "fit_infos.json").write_text(json.dumps(res.fit_infos, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
