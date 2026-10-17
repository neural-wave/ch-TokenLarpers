"""Frozen, repeated nested customer evaluation; no architecture search."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from .evaluation_models import ARMS, fit_predict
from .oof_contract import OOFRow, load_split_files, write_oof_predictions
from .task1_detector import extract_customer_windows
from .temporal import summarize_trace, temporal_diagnostics
from .windows import WindowConfig


def default_config() -> dict:
    return dict(context_length=672, stride=96, min_observed_fraction=.75,
                patch_length=16, d_model=64, n_heads=4, n_layers=2, dropout=.2,
                epochs=6, pretrain_epochs=5, batch_size=64, learning_rate=.001,
                weight_decay=.01, mask_ratio=.3, persistence_days=14,
                persistence_min_coverage=.9, persistence_max_gap_days=2,
                outer_folds=5, inner_folds=4, outer_seeds=[17, 42, 100],
                model_seed=17, thresholds=np.round(np.linspace(.2, .8, 61), 2).tolist())


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classification_metrics(truth: list, predictions: list) -> dict:
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[False, True]).ravel()
    return dict(f1=float(f1_score(truth, predictions, zero_division=0)),
                precision=float(precision_score(truth, predictions, zero_division=0)),
                recall=float(recall_score(truth, predictions, zero_division=0)),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def select_threshold(scores: dict[str, float], labels: pd.DataFrame, candidates: list[float]) -> tuple[float, list[dict]]:
    """Select solely from inner-OOF customer scores; earliest threshold breaks ties."""
    indexed = labels.set_index("customer_id")
    if set(scores) != set(indexed.index):
        raise ValueError("threshold calibration requires exactly one OOF score per training customer")
    ids = sorted(scores)
    truth = [bool(indexed.loc[c, "pv_installed"]) for c in ids]
    grid = [dict(threshold=float(t), **classification_metrics(truth, [scores[c] >= t for c in ids]))
            for t in sorted(set(candidates))]
    best = max(grid, key=lambda row: row["f1"])
    return best["threshold"], grid


def _split(labels: pd.DataFrame, folds: int, seed: int) -> list[tuple[list[str], list[str]]]:
    counts = labels.pv_installed.value_counts()
    if len(counts) != 2 or counts.min() < folds:
        raise ValueError(f"each class needs at least {folds} customers for requested customer CV")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(labels.iloc[a].customer_id.tolist(), labels.iloc[b].customer_id.tolist())
            for a, b in splitter.split(labels.customer_id, labels.pv_installed)]


def _trace_file(path: Path, windows: dict, probabilities: dict, metadata: dict) -> None:
    with path.open("w") as stream:
        columns = [*metadata, "customer_id", "window_start_epoch", "window_center_epoch", "center_date", "observed_fraction", "probability"]
        stream.write(";".join(columns) + "\n")
        for cid, batch in windows.items():
            for i, probability in enumerate(probabilities[cid]):
                values = [*metadata.values(), cid, int(batch.start_epochs[i]),
                          int(batch.start_epochs[i] + batch.values.shape[1] // 2 * 900),
                          batch.center_dates[i].isoformat(), float(batch.observed[i].mean()), float(probability)]
                stream.write(";".join(map(str, values)) + "\n")


def _temporal_summary(frame: pd.DataFrame) -> dict:
    positive = frame[frame.pv_installed_true]
    detected = positive[positive.pv_installed_pred]
    negative = frame[~frame.pv_installed_true]
    errors = positive.absolute_onset_error_days.dropna()
    signed_errors = positive.onset_error_days.dropna()
    pre = positive.pre_installation_false_alarm.dropna()
    late = positive[positive.late_installation.eq(True)]
    return dict(
        pv_customers=len(positive), non_pv_customers=len(negative),
        pre_installation_evaluable=len(pre), pre_installation_false_alarms=int(pre.sum()),
        non_pv_false_alarms=int(negative.pv_installed_pred.sum()),
        date_predictions_with_known_truth=len(errors),
        onset_outside_recorded_support=int(positive.observable_onset.eq(False).sum()),
        missing_installation_truth=int(positive.observable_onset.isna().sum()),
        missing_onset_predictions=int(positive.predicted_installation_date.isna().sum()),
        median_signed_onset_error_days=float(signed_errors.median()) if len(signed_errors) else None,
        median_absolute_onset_error_days=float(errors.median()) if len(errors) else None,
        fraction_onsets_within_7_days=float((errors <= 7).mean()) if len(errors) else None,
        fraction_onsets_within_15_days=float((errors <= 15).mean()) if len(errors) else None,
        fraction_onsets_within_30_days=float((errors <= 30).mean()) if len(errors) else None,
        date_reward_all_pv=float(positive.date_reward.mean()) if positive.date_reward.notna().any() else None,
        date_reward_detected_pv=float(detected.date_reward.mean()) if detected.date_reward.notna().any() else None,
        late_installations=len(late), late_installations_detected=int(late.pv_installed_pred.sum()),
        insufficient_persistence_coverage=int((~frame.coverage_eligible).sum()))


def run_evaluation(canonical_dir: str | Path, out_dir: str | Path, *, data_kind: str,
                   config: dict | None = None, split_files: list[str | Path] | None = None) -> dict:
    """Train real arms on nested folds and retain every inner/outer prediction."""
    if data_kind not in {"artificial", "development"}:
        raise ValueError("data_kind must explicitly be artificial or development; CV is not blind evaluation")
    cfg = default_config()
    supplied = dict(config or {})
    if set(supplied) - set(cfg):
        raise ValueError(f"unknown evaluation settings: {sorted(set(supplied) - set(cfg))}")
    cfg.update(supplied)
    # Daily centers and uncapped windows are part of the coverage contract.
    if cfg["stride"] != 96 or cfg["context_length"] != 672:
        raise ValueError("frozen evaluation requires daily stride and seven-day windows")
    for name in ("outer_folds", "inner_folds"):
        if not isinstance(cfg[name], int) or cfg[name] < 2:
            raise ValueError(f"{name} must be an integer >=2")
    seeds = cfg["outer_seeds"]
    if not split_files and (len(seeds) < 2 or len(set(seeds)) != len(seeds)):
        raise ValueError("use at least two distinct outer split seeds")
    if not cfg["thresholds"] or any(not np.isfinite(t) or not 0 < t < 1 for t in cfg["thresholds"]):
        raise ValueError("threshold candidates must be finite and strictly between zero and one")
    source, out = Path(canonical_dir), Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("evaluation output directory must be empty; refusing to overwrite evidence")
    labels = pd.read_parquet(source / "labels.parquet")
    if labels.customer_id.isna().any() or labels.customer_id.astype(str).duplicated().any():
        raise ValueError("labels require unique non-null customer IDs")
    if not labels.pv_installed.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("pv_installed labels must be booleans")
    labels["customer_id"] = labels.customer_id.astype(str)
    labels = labels.sort_values("customer_id").reset_index(drop=True)
    label_lookup = labels.set_index("customer_id")
    if split_files:
        maps = load_split_files(split_files, labels.customer_id)
        fold_labels = [sorted(set(mapping.values())) for mapping in maps]
        if any(len(values) < 2 for values in fold_labels):
            raise ValueError("each supplied split file requires at least two customer folds")
        ids = labels.customer_id.tolist()
        outer = [[([c for c in ids if mapping[c] != f],
                   [c for c in ids if mapping[c] == f]) for f in values]
                 for mapping, values in zip(maps, fold_labels)]
        seeds = [cfg["model_seed"] + 1000 * r for r in range(len(outer))]
    else:
        outer = [_split(labels, cfg["outer_folds"], seed) for seed in seeds]
        fold_labels = [list(range(len(assignments))) for assignments in outer]
    # Preflight every nested split before fitting anything.
    signatures = []
    nested = {}
    for repeat, assignments in enumerate(outer):
        signature = tuple(sorted(tuple(sorted(val)) for _, val in assignments))
        if signature in signatures:
            raise ValueError("outer seeds produced identical customer partitions; choose genuinely different seeds")
        signatures.append(signature)
        for index, (train, _) in enumerate(assignments):
            fold = fold_labels[repeat][index]
            nested[repeat, fold] = _split(labels[labels.customer_id.isin(train)], cfg["inner_folds"], seeds[repeat] + fold + 10000)
    windows = extract_customer_windows(source, labels.customer_id.tolist(), WindowConfig(
        context_length=cfg["context_length"], stride=cfg["stride"], min_observed_fraction=cfg["min_observed_fraction"]))
    if set(windows) != set(labels.customer_id):
        raise ValueError(f"customers without eligible windows: {sorted(set(labels.customer_id) - set(windows))}")
    out.mkdir(parents=True, exist_ok=True)
    import sklearn
    import torch
    _json(out / "config.json", cfg)
    _json(out / "manifest.json", dict(protocol="evaluation-v1", data_kind=data_kind,
          evidence_status="artificial_smoke_only" if data_kind == "artificial" else "development_only",
          blind_evaluation=dict(performed=False, reason="No fresh blinded labelled evaluation supplied; organizer test labels remain hidden."),
          inputs={name: _hash(source / name) for name in ("labels.parquet", "measurements.parquet")},
          supplied_splits=[dict(path=str(p), sha256=_hash(Path(p))) for p in split_files or []],
          config_sha256=_hash(out / "config.json"),
          versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, torch=torch.__version__),
          arms=list(ARMS), capacity_evaluated=False,
          budget_note="Identical supervised budgets; pretrained arm additionally receives the frozen masked-pretraining budget. No architecture or hyperparameter search."))
    all_rows = []
    split_rows = []
    per_repeat = []
    for repeat, assignments in enumerate(outer):
        split_seed = None if split_files else seeds[repeat]
        repeat_rows = []
        for index, (train_ids, val_ids) in enumerate(assignments):
            fold = fold_labels[repeat][index]
            train_labels = labels[labels.customer_id.isin(train_ids)].copy()
            train_windows = {c: windows[c] for c in train_ids}
            val_windows = {c: windows[c] for c in val_ids}
            split_rows.extend(dict(repeat=repeat, split_seed=split_seed, outer_fold=fold, inner_fold=None,
                                   customer_id=c, role="train" if c in train_ids else "validation") for c in labels.customer_id)
            inner = nested[repeat, fold]
            for inner_fold, (it, iv) in enumerate(inner):
                split_rows.extend(dict(repeat=repeat, split_seed=split_seed, outer_fold=fold, inner_fold=inner_fold,
                                       customer_id=c, role="train" if c in it else "validation") for c in train_ids)
            for arm in ARMS:
                directory = out / f"repeat_{repeat}" / f"fold_{fold}" / arm
                directory.mkdir(parents=True)
                inner_scores = {}
                calibration = []
                for inner_fold, (it, iv) in enumerate(inner):
                    inner_dir = directory / f"inner_{inner_fold}"
                    inner_dir.mkdir()
                    iw = {c: windows[c] for c in iv}
                    probability = fit_predict(arm, {c: windows[c] for c in it},
                        train_labels[train_labels.customer_id.isin(it)].copy(), iw, cfg, cfg["model_seed"], inner_dir)
                    _trace_file(inner_dir / "traces.csv", iw, probability, dict(repeat=repeat, outer_fold=fold, inner_fold=inner_fold, arm=arm))
                    for cid in iv:
                        result = summarize_trace(windows[cid], probability[cid], .5, cfg)
                        if cid in inner_scores:
                            raise ValueError("duplicate inner OOF customer")
                        inner_scores[cid] = result["score"]
                        calibration.append(dict(customer_id=cid, inner_fold=inner_fold, score=result["score"],
                                                pv_installed_true=bool(train_labels.set_index("customer_id").loc[cid, "pv_installed"])))
                threshold, grid = select_threshold(inner_scores, train_labels, cfg["thresholds"])
                pd.DataFrame(calibration).to_csv(directory / "inner_oof.csv", sep=";", index=False)
                _json(directory / "calibration.json", dict(threshold=threshold, source="inner_customer_oof", candidates=grid))
                outer_dir = directory / "outer"
                outer_dir.mkdir()
                probability = fit_predict(arm, train_windows, train_labels, val_windows, cfg, cfg["model_seed"], outer_dir)
                _trace_file(outer_dir / "traces.csv", val_windows, probability, dict(repeat=repeat, outer_fold=fold, arm=arm))
                # Outer labels are used only after all fitting, calibration and predictions.
                for cid in val_ids:
                    result = summarize_trace(windows[cid], probability[cid], threshold, cfg)
                    label = label_lookup.loc[cid].to_dict()
                    diagnostics = temporal_diagnostics(windows[cid], probability[cid], label, threshold, cfg)
                    row = dict(repeat=repeat, split_seed=split_seed, outer_fold=fold, arm=arm, customer_id=cid,
                               threshold=threshold, pv_installed_true=bool(label["pv_installed"]),
                               pv_installed_pred=bool(result["score"] >= threshold), **result, **diagnostics)
                    repeat_rows.append(row)
                print(f"repeat={repeat} fold={fold} arm={arm} threshold={threshold:.2f}", flush=True)
        frame = pd.DataFrame(repeat_rows)
        for arm in ARMS:
            part = frame[frame.arm == arm]
            if part.customer_id.duplicated().any() or set(part.customer_id) != set(labels.customer_id):
                raise ValueError("outer OOF predictions must cover each customer exactly once per arm/repeat")
            per_repeat.append(dict(repeat=repeat, split_seed=split_seed, arm=arm,
                **classification_metrics(part.pv_installed_true.tolist(), part.pv_installed_pred.tolist()),
                temporal=_temporal_summary(part)))
            all_for_arm = [r for r in all_rows + repeat_rows if r["arm"] == arm]
            write_oof_predictions([
                OOFRow(customer_id=r["customer_id"], repeat=r["repeat"],
                       fold=r["outer_fold"], prob=r["score"],
                       pv_installed=r["pv_installed_pred"])
                for r in all_for_arm
            ], out / f"oof_{arm}.csv")
        all_rows.extend(repeat_rows)
        pd.DataFrame(all_rows).to_csv(out / "predictions.csv", sep=";", index=False)
        pd.DataFrame(split_rows).to_csv(out / "splits.csv", sep=";", index=False)
    aggregate = {arm: dict(mean_f1=float(np.mean([r["f1"] for r in per_repeat if r["arm"] == arm])),
                           std_f1=float(np.std([r["f1"] for r in per_repeat if r["arm"] == arm]))) for arm in ARMS}
    result = dict(per_repeat=per_repeat, descriptive_across_repeats=aggregate,
                  interpretation="Repeated customer CV on the same customers; repeats are not independent new customers. No blind-test claim and no capacity/main-score claim.")
    _json(out / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--data-kind", required=True, choices=["artificial", "development"])
    parser.add_argument("--splits", help="comma-separated shared customer split files; one file per repeat")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else None
    splits = [p.strip() for p in args.splits.split(",")] if args.splits else None
    result = run_evaluation(args.canonical_dir, args.out, data_kind=args.data_kind,
                            config=config, split_files=splits)
    print(json.dumps(result["descriptive_across_repeats"], indent=2))


if __name__ == "__main__":
    main()
