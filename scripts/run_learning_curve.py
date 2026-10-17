"""Compare baseline and frozen-encoder hybrids at fixed labelled sample sizes."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from pvdetect.embedding import FrozenEncoder
from pvdetect.evaluation import splits_from_fold_map
from pvdetect.io import load_labels, load_timeseries
from pvdetect.pipeline import build_feature_table, predict
from pvdetect.procedure import fit
from pvdetect.scoring import main_score
from neuralwave_experiment.oof_contract import load_split_files


def _stratified_subset(ids: list[str], labels_df: pd.DataFrame, n_samples: int, seed: int) -> list[str]:
    sub = labels_df.set_index("customer_id").loc[ids]
    y = sub["pv_installed"].to_numpy(bool)
    if len(ids) < n_samples:
        raise ValueError("requested training size exceeds an outer training fold")
    if len(ids) == n_samples:
        return ids
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    train_idx, _ = next(sss.split(ids, y))
    return [ids[i] for i in sorted(train_idx)]


def _digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_learning_curve(
    timeseries_path: str | Path,
    labels_path: str | Path,
    split_paths: list[str | Path],
    small_encoder_path: str | Path | None = None,
    medium_encoder_path: str | Path | None = None,
    subset_sizes: tuple[int, ...] = (8, 16, 32),
    seed: int = 17,
    *,
    out_path: str | Path | None = None,
    resume: bool = False,
) -> dict:
    """Persist each finished fold privately; resume only identical inputs/code."""
    started = time.perf_counter()
    def progress(message: str) -> None:
        print(f"[{time.perf_counter() - started:.1f}s] {message}", flush=True)

    if resume and out_path is None:
        raise ValueError("resume requires an output path")
    if not subset_sizes or any(n < 4 for n in subset_sizes):
        raise ValueError("training subset sizes must be at least four")
    progress("Loading measurements and labels")
    loaded = load_timeseries(timeseries_path)
    labels = load_labels(labels_path)
    ids = labels["customer_id"].tolist()
    all_splits = [splits_from_fold_map(m, ids) for m in load_split_files(split_paths, ids)]
    if any(n > len(tr) for n in subset_sizes for rep in all_splits for tr, _ in rep):
        raise ValueError("requested training size exceeds an outer training fold")
    progress("Building physical features")
    table = build_feature_table(loaded)
    encoders = {}
    for name, path in (("hybrid_small", small_encoder_path), ("hybrid_medium", medium_encoder_path)):
        if path:
            encoders[name] = FrozenEncoder.load(path)
    configs = [("baseline", None, "baseline")]
    configs.extend((name, enc, "auto") for name, enc in encoders.items())

    checkpoint = Path(str(out_path) + ".progress.json") if out_path is not None else None
    state = {"folds": {}, "results": {}}
    if checkpoint is not None:
        repo = Path(__file__).resolve().parents[1]
        sources = sorted((repo / "pvdetect").rglob("*.py"))
        sources += sorted((repo / "src").rglob("*.py")) + [Path(__file__).resolve()]
        import importlib.metadata
        contract = {
            "schema": 1,
            "inputs": [_digest(p) for p in (timeseries_path, labels_path, *split_paths)],
            "encoders": {name: enc.hashes for name, enc in encoders.items()},
            "subsets": list(subset_sizes), "seed": seed,
            "code": {str(p.relative_to(repo)): _digest(p) for p in sources},
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "catboost")},
        }
        if encoders:
            import torch
            contract["torch"] = {"version": torch.__version__, "threads": torch.get_num_threads()}
        if resume:
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if state.get("contract") != contract:
                raise ValueError("resume refused: inputs, configuration, code or runtime changed")
            progress(f"Resuming {len(state['folds'])} completed folds")
        elif checkpoint.exists() or Path(out_path).exists():
            raise FileExistsError("output exists; use --resume or a new --out path")
        else:
            state["contract"] = contract
            _save(checkpoint, state)

    results = state["results"]
    for n in subset_sizes:
        group = results.setdefault(f"train_{n}", {})
        for cfg_name, enc, block in configs:
            if cfg_name in group:
                progress(f"N={n} {cfg_name}: already complete")
                continue
            if enc is not None:
                progress(f"N={n} {cfg_name}: extracting frozen features (window cap={enc.window_config.get('max_windows_per_customer')})")
                enc.raw_customer_features(table, ids)
                progress(f"N={n} {cfg_name}: frozen features ready")
            repeat_scores, selected_blocks = [], []
            for rep_idx, rep in enumerate(all_splits):
                preds = []
                for fold_idx, (tr_ids, te_ids) in enumerate(rep):
                    key = f"{n}/{cfg_name}/{rep_idx}/{fold_idx}"
                    saved = state["folds"].get(key)
                    if saved is None:
                        progress(f"N={n} {cfg_name} repeat={rep_idx + 1}/{len(all_splits)} fold={fold_idx + 1}/{len(rep)}: fitting")
                        sub_tr = _stratified_subset(tr_ids, labels, n, seed + rep_idx * 100 + fold_idx)
                        model = fit(table, labels, sub_tr, encoder=enc, feature_block=block)
                        p, _ = predict(model, table, te_ids)
                        saved = {"predictions": json.loads(p.to_json(orient="records", date_format="iso")),
                                 "selected": model.get("selection", {}).get("feature_block", "baseline")}
                        state["folds"][key] = saved
                        if checkpoint is not None:
                            _save(checkpoint, state)
                        progress(f"N={n} {cfg_name} repeat={rep_idx + 1} fold={fold_idx + 1}: saved ({saved['selected']})")
                    p = pd.DataFrame(saved["predictions"])
                    p["pv_installation_date"] = pd.to_datetime(p["pv_installation_date"])
                    preds.append(p)
                    selected_blocks.append(saved["selected"])
                pooled = pd.concat(preds, ignore_index=True)
                repeat_scores.append(main_score(pooled, labels[labels["customer_id"].isin(pooled["customer_id"])]))
            scores = [s["score"] for s in repeat_scores]
            counts = pd.Series(selected_blocks).value_counts().to_dict()
            group[cfg_name] = {
                "score_mean": float(np.mean(scores)), "score_std": float(np.std(scores)),
                "f1_mean": float(np.mean([s["f1"] for s in repeat_scores])),
                "d_mean": float(np.mean([s["D"] for s in repeat_scores])),
                "p_mean": float(np.mean([s["P"] for s in repeat_scores])),
                "repeat_scores": scores, "selected_feature_blocks": counts,
            }
            if checkpoint is not None:
                _save(checkpoint, state)
                _save(Path(out_path), results)
            progress(f"N={n} {cfg_name}: Score={np.mean(scores):.2f} ± {np.std(scores):.2f}; F1={group[cfg_name]['f1_mean']:.3f}; selected={counts}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", required=True, help="comma-separated list of split files")
    ap.add_argument("--small-encoder")
    ap.add_argument("--medium-encoder")
    ap.add_argument("--out", default="artifacts_private/learning_curve_results.json")
    ap.add_argument("--subsets", default="8,16,32")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()
    if args.torch_threads < 1:
        ap.error("--torch-threads must be positive")
    if args.small_encoder or args.medium_encoder:
        import torch
        torch.set_num_threads(args.torch_threads)
    run_learning_curve(
        args.timeseries, args.labels, [p.strip() for p in args.splits.split(",")],
        small_encoder_path=args.small_encoder, medium_encoder_path=args.medium_encoder,
        subset_sizes=tuple(int(x.strip()) for x in args.subsets.split(",")),
        out_path=args.out, resume=args.resume,
    )
    print(f"Results saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
