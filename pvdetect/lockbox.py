"""Evaluation protocol v2: nested CV on a development set, one sealed stratified lock-box, full refit.

    python -m pvdetect.lockbox seal    --dir PRIVATE --energy E --labels L [--config configs/lockbox.json]
    python -m pvdetect.lockbox round   --dir NEW --from PRIVATE [--config configs/lockbox_round2.json]
    python -m pvdetect.lockbox develop --dir PRIVATE NAME [--unlabelled U] [-- TRAIN_FLAGS ...]
    python -m pvdetect.lockbox compare --dir PRIVATE A B [--force]
    python -m pvdetect.lockbox freeze  --dir PRIVATE NAME
    python -m pvdetect.lockbox open    --dir PRIVATE [--force]
    python -m pvdetect.lockbox final   --dir PRIVATE --energy E --labels L [--unlabelled U] [--cv] [--force]
    python -m pvdetect.lockbox status  --dir PRIVATE

Plan: docs/evaluation_lockbox_plan.md. PRIVATE (outside the repository) holds everything: `state.json`
(stage, timestamps, git commit, config hash, runs, decisions used, forced overrides), `seal.json`, the
dev / lock-box files, the shared outer split files and one folder per run. Stages refuse to run out of
order: seal → develop / compare (≤ decision_budget comparisons) → freeze → open (once) → final.
`round` starts a new development round on the same seal (same dev / lock-box customers and split files,
hashes verified, hardlinked): only the decision budget may change, and earlier lock-box looks are carried
over (`prior_looks`) so every later look is reported as such. Every stage prints aggregate lines only.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pvdetect import compare as compare_mod
from pvdetect import diagnose
from pvdetect.evaluation import stratified_fold_map
from pvdetect.io import _read_str_table, load_labels, split_by_customer
from pvdetect.scoring import bootstrap_weights, score_components, weighted_scores

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "lockbox.json"
STATE_FILE = "state.json"
SEAL_FILE = "seal.json"
RESULT_FILE = "lockbox_result.json"
SCHEMA = 1
STAGES = ("sealed", "frozen", "opened", "final")
CONFIG_KEYS = ("seed", "lockbox_size", "strata", "outer_seeds", "outer_folds", "decision_budget")
SPLIT_KEYS = ("seed", "lockbox_size", "strata", "outer_seeds", "outer_folds")  # fixed across rounds
SEALED_FILES = ("dev.csv", "dev_labels.csv", "lockbox.csv", "lockbox_labels.csv")
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.90
# train.py flags the protocol sets itself; a run may not override them.
RESERVED_FLAGS = {
    "--timeseries", "--labels", "--out", "--cv", "--folds", "--oof-out", "--outer-repeats", "--report-dir",
    "--split-dir", "--split-frac", "--split-seed", "--unlabelled",
}
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
NESTED_RE = re.compile(
    r"nested: score=(?P<score>\S+?)±(?P<sd>\S+) ci90=\[(?P<lo>[^,\]]+),(?P<hi>[^\]]+)\] "
    r"f1=(?P<f1>\S+) D=(?P<D>\S+) P=(?P<P>\S+).*?outer_informed_decisions=(?P<k>\d+)"
)


# --------------------------------------------------------------------------- helpers


def _refuse(msg: str) -> SystemExit:
    return SystemExit(f"refusing: {msg}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_ids(ids) -> str:
    return hashlib.sha256("\n".join(sorted(str(c) for c in ids)).encode("utf-8")).hexdigest()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != set(CONFIG_KEYS):
        raise _refuse(f"config must have exactly the keys {list(CONFIG_KEYS)}")
    if cfg["strata"] != ["pv_installed"]:
        raise _refuse('only "strata": ["pv_installed"] is supported')
    seeds = cfg["outer_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) for s in seeds):
        raise _refuse("outer_seeds must be distinct integers")
    for key in ("seed", "lockbox_size", "outer_folds", "decision_budget"):
        if not isinstance(cfg[key], int) or cfg[key] < (2 if key == "outer_folds" else 0):
            raise _refuse(f"{key} must be a non-negative integer (outer_folds ≥ 2)")
    return cfg


def _git() -> tuple[str, bool | None]:
    """(HEAD commit, tracked files modified) of the repository; ("unknown", None) outside git."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return commit, bool(status.strip())


def _repo_module(name: str):
    """`train` / `inference` live at the repository root; attributes are looked up at call time."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return importlib.import_module(name)


class _Tee(io.TextIOBase):
    def __init__(self, out):
        self.out, self.buf = out, io.StringIO()

    def write(self, s: str) -> int:
        self.out.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self) -> None:
        self.out.flush()


def _run(main, argv: list[str]) -> str:
    """Call a CLI `main(argv)` in-process, printing as usual; returns what it printed."""
    tee = _Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        rc = main(argv)
    if rc not in (0, None):
        raise SystemExit(f"step failed with exit code {rc}")
    return tee.buf.getvalue()


def parse_nested(text: str) -> dict | None:
    """The `nested:` line of `train.py --cv` as numbers, or None when no nested CV ran."""
    m = NESTED_RE.search(text)
    if m is None:
        return None
    f = {k: float(m.group(k)) for k in ("score", "sd", "lo", "hi", "f1", "D", "P")}
    return {
        "score": f["score"], "sd": f["sd"], "ci90": [f["lo"], f["hi"]], "f1": f["f1"], "D": f["D"], "P": f["P"],
        "outer_informed_decisions": int(m.group("k")),
    }


def _fmt(x, nd: int = 2) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _ci(ci) -> str:
    return f"[{_fmt(ci[0])},{_fmt(ci[1])}]" if ci else "[nan,nan]"


def _write_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(diagnose._json_safe(doc), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _private_dir(raw: str, create: bool = False) -> Path:
    d = diagnose._outside_repo(Path(raw), "--dir")
    if create:
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o700)
    return d


def _load_state(d: Path) -> dict:
    path = d / STATE_FILE
    if not path.is_file():
        raise _refuse(f"no {STATE_FILE} in --dir; run `seal` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(d: Path, st: dict) -> None:
    _write_json(d / STATE_FILE, st)


def _stage(st: dict) -> int:
    return STAGES.index(st["stage"])


def _advance(st: dict, stage: str) -> None:
    if STAGES.index(stage) > _stage(st):
        st["stage"] = stage


def _force(st: dict, stage: str, detail: str) -> None:
    st["forced"].append({"stage": stage, "time": _now(), "detail": detail})


def _split_files(d: Path, cfg: dict) -> list[Path]:
    return [d / f"splits_s{s}.csv" for s in cfg["outer_seeds"]]


def _run_dir(d: Path, st: dict, name: str) -> Path:
    if name not in st["runs"]:
        raise _refuse(f"no run named {name!r} (see `status`)")
    return d / "runs" / name


def score_ci(pred: pd.DataFrame, truth: pd.DataFrame, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """90 % customer-bootstrap interval of the main score (customers resampled with replacement)."""
    comp = score_components(pred, truth)
    draws = weighted_scores(bootstrap_weights(len(comp), B, seed), comp)
    a = (1 - CI_LEVEL) / 2
    return float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


def vs_nested(score: float, nested: dict | None) -> str:
    """Where the lock-box score falls relative to the frozen run's nested 90 % interval."""
    if not nested or nested.get("ci90") is None or any(v is None or not np.isfinite(v) for v in nested["ci90"]):
        return "unavailable"
    lo, hi = nested["ci90"]
    return "below" if score < lo else "above" if score > hi else "inside"


# --------------------------------------------------------------------------- stages


def seal(args) -> int:
    d = _private_dir(args.dir, create=True)
    if (d / STATE_FILE).exists():
        raise _refuse(f"--dir is already sealed ({STATE_FILE} exists); the split is made once")
    cfg = load_config(args.config)
    labels = load_labels(args.labels)
    tmp = d / ".seal_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        dev_ids, box_ids = split_by_customer(
            args.energy, tmp, seed=cfg["seed"], labels=labels, n_test=cfg["lockbox_size"]
        )
    except ValueError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise _refuse(f"lock-box split: {e}") from None
    os.replace(tmp / "training.csv", d / "dev.csv")
    os.replace(tmp / "test.csv", d / "lockbox.csv")
    tmp.rmdir()

    raw, sep = _read_str_table(args.labels)
    dev_set, box_set = set(dev_ids), set(box_ids)
    raw[raw["customer_id"].isin(dev_set)].to_csv(d / "dev_labels.csv", sep=sep, index=False)
    raw[raw["customer_id"].isin(box_set)].to_csv(d / "lockbox_labels.csv", sep=sep, index=False)
    dev_lab = labels[labels["customer_id"].isin(dev_set)]
    box_lab = labels[labels["customer_id"].isin(box_set)]
    try:
        maps = [stratified_fold_map(dev_lab, cfg["outer_folds"], s) for s in cfg["outer_seeds"]]
    except ValueError as e:
        raise _refuse(f"dev set: {e}") from None
    for path, fold_map in zip(_split_files(d, cfg), maps):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";", lineterminator="\n")
            w.writerow(["customer_id", "fold"])
            w.writerows((c, fold_map[c]) for c in sorted(fold_map))

    counts = {
        "dev_customers": len(dev_ids),
        "dev_labelled": len(dev_lab),
        "dev_pv": int(dev_lab["pv_installed"].sum()),
        "lockbox_customers": len(box_ids),
        "lockbox_pv": int(box_lab["pv_installed"].sum()),
        "folds": len(set(maps[0].values())),
    }
    now, (commit, dirty), chash = _now(), _git(), config_hash(cfg)
    _write_json(
        d / SEAL_FILE,
        {
            "schema": SCHEMA,
            "sealed_at": now,
            "config": cfg,
            "config_hash": chash,
            "lockbox_ids_sha256": _sha256_ids(box_ids),
            "lockbox_csv_sha256": _sha256_file(d / "lockbox.csv"),
            "lockbox_labels_sha256": _sha256_file(d / "lockbox_labels.csv"),
            "counts": counts,
        },
    )
    _save_state(
        d,
        {
            "schema": SCHEMA,
            "stage": "sealed",
            "timestamps": {"sealed": now},
            "git": {"commit": commit, "dirty": dirty},
            "config_path": str(Path(args.config).resolve()),
            "config": cfg,
            "config_hash": chash,
            "runs": {},
            "decisions": [],
            "decisions_used": 0,
            "forced": [],
            "frozen": None,
            "opens": 0,
            "lockbox": None,
            "final": None,
        },
    )
    print(
        f"seal: dev customers={counts['dev_customers']} labelled={counts['dev_labelled']} pv={counts['dev_pv']} | "
        f"lockbox customers={counts['lockbox_customers']} pv={counts['lockbox_pv']}"
    )
    print(
        f"seal: split files={len(maps)} seeds={','.join(map(str, cfg['outer_seeds']))} folds={counts['folds']} "
        f"decision_budget={cfg['decision_budget']} config_hash={chash[:12]}"
    )
    print("seal: dev/lock-box files, split files, seal.json and state.json written to the private --dir")
    return 0


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def new_round(args) -> int:
    """New development round on an existing seal: the same customers, lock-box and split files."""
    d = _private_dir(args.dir, create=True)
    if (d / STATE_FILE).exists():
        raise _refuse(f"--dir already holds a protocol ({STATE_FILE} exists)")
    src = _private_dir(args.source)
    if src == d:
        raise _refuse("--from and --dir must differ")
    old = _load_state(src)
    seal_doc = json.loads((src / SEAL_FILE).read_text(encoding="utf-8"))
    for key, name in (("lockbox_csv_sha256", "lockbox.csv"), ("lockbox_labels_sha256", "lockbox_labels.csv")):
        if _sha256_file(src / name) != seal_doc[key]:
            raise _refuse(f"{name} in --from does not match its seal.json")
    cfg = load_config(args.config)
    changed = [k for k in SPLIT_KEYS if cfg[k] != old["config"][k]]
    if changed:
        raise _refuse(f"a new round keeps the sealed split; config differs in {changed}")
    for name in (*SEALED_FILES, *(p.name for p in _split_files(src, cfg))):
        if not (src / name).is_file():
            raise _refuse(f"{name} missing in --from")
        _link_or_copy(src / name, d / name)
    shutil.copy2(src / SEAL_FILE, d / SEAL_FILE)
    now, (commit, dirty), chash = _now(), _git(), config_hash(cfg)
    prior = int(old.get("prior_looks", 0)) + int(old["opens"])
    _save_state(
        d,
        {
            "schema": SCHEMA,
            "stage": "sealed",
            "timestamps": {"sealed": seal_doc["sealed_at"], "round_started": now},
            "git": {"commit": commit, "dirty": dirty},
            "config_path": str(Path(args.config).resolve()),
            "config": cfg,
            "config_hash": chash,
            "seal_config_hash": seal_doc["config_hash"],
            "round": int(old.get("round", 1)) + 1,
            "previous_round": str(src),
            "prior_looks": prior,
            "runs": {},
            "decisions": [],
            "decisions_used": 0,
            "forced": [],
            "frozen": None,
            "opens": 0,
            "lockbox": None,
            "final": None,
        },
    )
    print(
        f"round: {int(old.get('round', 1)) + 1} on the seal of {src.name} | prior lock-box looks={prior} | "
        f"decision_budget={cfg['decision_budget']} config_hash={chash[:12]}"
    )
    return 0


def develop(args, extra: list[str]) -> int:
    d = _private_dir(args.dir)
    st = _load_state(d)
    if st["stage"] != "sealed":
        raise _refuse(f"stage is {st['stage']}; develop runs only before freeze")
    if not NAME_RE.fullmatch(args.name):
        raise _refuse("run names use letters, digits, '_', '.', '-' (≤ 64 characters)")
    if args.name in st["runs"]:
        raise _refuse(f"run {args.name!r} exists; choose a new name")
    reserved = sorted({f.split("=", 1)[0] for f in extra} & RESERVED_FLAGS)
    if reserved:
        raise _refuse(f"the protocol sets {reserved} itself")
    run = d / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    unlabelled = str(Path(args.unlabelled).expanduser().resolve()) if args.unlabelled else None
    argv = [
        "--timeseries", str(d / "dev.csv"), "--labels", str(d / "dev_labels.csv"), "--cv",
        "--folds", ",".join(str(p) for p in _split_files(d, st["config"])),
        "--oof-out", str(run / "oof.csv"), "--out", str(run / "model.json"),
        *(["--unlabelled", unlabelled] if unlabelled else []), *extra,
    ]
    text = _run(_repo_module("train").main, argv)
    if not (run / "oof.csv").is_file() or not (run / "model.json").is_file():
        raise SystemExit("develop: train.py wrote no out-of-fold file or model; run not recorded")
    print()
    _run(diagnose.main, [str(run / "oof.csv"), str(d / "dev_labels.csv"), "--out-dir", str(run)])
    nested = parse_nested(text)
    commit, dirty = _git()
    st["runs"][args.name] = {
        "created": _now(), "flags": list(extra), "unlabelled": unlabelled,
        "git_commit": commit, "dirty": dirty, "nested": nested,
    }
    _save_state(d, st)
    n = nested or {}
    print(
        f"develop: run={args.name} recorded nested score={_fmt(n.get('score'))} ci90={_ci(n.get('ci90'))} "
        f"runs={len(st['runs'])}"
    )
    return 0


def compare(args) -> int:
    d = _private_dir(args.dir)
    st = _load_state(d)
    if st["stage"] != "sealed":
        raise _refuse(f"stage is {st['stage']}; comparisons happen only before freeze")
    if args.a == args.b:
        raise _refuse("compare two different runs")
    a_dir, b_dir = _run_dir(d, st, args.a), _run_dir(d, st, args.b)
    used, budget = st["decisions_used"], st["config"]["decision_budget"]
    if used >= budget and not args.force:
        raise _refuse(f"decision budget exhausted ({used}/{budget}); freeze, or --force (recorded)")
    try:
        lines, changes = compare_mod.compare(
            compare_mod.load_oof(a_dir / "oof.csv"), compare_mod.load_oof(b_dir / "oof.csv"),
            load_labels(d / "dev_labels.csv"), target=args.target,
        )
    except ValueError as e:
        raise _refuse(str(e)) from None
    out = b_dir / f"compare_vs_{args.a}"
    out.mkdir(parents=True, exist_ok=True)
    changes.to_csv(out / "changes.csv", sep=";", index_label="customer_id", float_format="%.4f")
    (out / "compare.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    boot = next(line for line in lines if line.startswith("bootstrap:"))
    m = re.search(r"score_diff=(\S+) ci90=\[([^,]+),([^\]]+)\]", boot)
    final = next((line for line in lines if line.startswith("targeted:")), boot)
    v = re.search(r"verdict=(\w+)\s*$", final)
    verdict = v.group(1) if v else "unknown"
    if used >= budget:
        _force(st, "compare", f"{args.a} vs {args.b} beyond the decision budget")
    st["decisions_used"] = used + 1
    st["decisions"].append(
        {
            "time": _now(), "base": args.a, "cand": args.b, "verdict": verdict, "forced": used >= budget,
            "target": args.target,
            "score_diff": float(m.group(1)) if m else None,
            "ci90": [float(m.group(2)), float(m.group(3))] if m else None,
        }
    )
    _save_state(d, st)
    print(f"verdict: {args.b} vs {args.a} = {verdict} (decisions used {used + 1}/{budget})")
    print(f"decision log (Tier B): append to DECISIONS.md: - {date.today().isoformat()} | lock-box dev compare "
          f"{args.a} vs {args.b}: <adopt or keep> | outer results consulted")
    return 0


def freeze(args) -> int:
    d = _private_dir(args.dir)
    st = _load_state(d)
    if st["stage"] != "sealed":
        raise _refuse(f"stage is {st['stage']}; a protocol run is frozen once")
    run = _run_dir(d, st, args.name)
    info = st["runs"][args.name]
    commit, dirty = _git()
    now = _now()
    st["frozen"] = {
        "run": args.name, "time": now, "git_commit": commit, "dirty": dirty, "flags": info["flags"],
        "unlabelled": info["unlabelled"], "model_sha256": _sha256_file(run / "model.json"), "nested": info["nested"],
    }
    st["timestamps"]["frozen"] = now
    _advance(st, "frozen")
    _save_state(d, st)
    print(
        f"freeze: run={args.name} commit={commit[:12]} dirty={'unknown' if dirty is None else 'yes' if dirty else 'no'} "
        f"flags={' '.join(info['flags']) or '(defaults)'}"
    )
    print(f"freeze: tag the code with: git tag freeze-vN {commit[:12]}")
    if dirty:
        print("warning: tracked files have uncommitted changes; the frozen code is not exactly this commit")
    return 0


def open_lockbox(args) -> int:
    d = _private_dir(args.dir)
    st = _load_state(d)
    if _stage(st) < STAGES.index("frozen"):
        raise _refuse("freeze a run before opening the lock-box")
    if st["opens"] and not args.force:
        raise _refuse("the lock-box was already opened (single look); --force reopens and is recorded")
    seal_doc = json.loads((d / SEAL_FILE).read_text(encoding="utf-8"))
    for key, name in (("lockbox_csv_sha256", "lockbox.csv"), ("lockbox_labels_sha256", "lockbox_labels.csv")):
        if _sha256_file(d / name) != seal_doc[key]:
            raise _refuse(f"{name} does not match seal.json; the lock-box was modified after sealing")
    expected = st.get("seal_config_hash", st["config_hash"])
    if seal_doc["config_hash"] != expected:
        raise _refuse("state.json and seal.json disagree on the config hash")
    truth = load_labels(d / "lockbox_labels.csv")
    if _sha256_ids(truth["customer_id"]) != seal_doc["lockbox_ids_sha256"]:
        raise _refuse("lock-box customer ids do not match seal.json")
    fr = st["frozen"]
    model = d / "runs" / fr["run"] / "model.json"
    if _sha256_file(model) != fr["model_sha256"]:
        raise _refuse("the frozen run's model.json changed after freeze")
    commit, dirty = _git()
    code_changed = commit != fr["git_commit"] or bool(dirty)
    if code_changed:
        print("warning: the repository is not at the frozen commit (or has uncommitted changes); recorded")
    pred_dir = d / "lockbox_predictions"
    shutil.rmtree(pred_dir, ignore_errors=True)
    pred_dir.mkdir()
    run = _repo_module("inference").run_inference(d / "lockbox.csv", model, pred_dir)
    if _sha256_ids(run.customer_ids) != seal_doc["lockbox_ids_sha256"]:
        raise _refuse("predicted customers do not match the sealed lock-box ids")
    pred, _ = diagnose.load_predictions(pred_dir / "predictions.csv")
    summary = diagnose.summarize(diagnose.error_table(pred, truth))
    lo, hi = score_ci(pred, truth)
    nested = fr["nested"]
    position = vs_nested(summary["score"], nested)
    forced = bool(st["opens"])
    now = _now()
    if forced:
        _force(st, "open", "lock-box opened again")
    st["opens"] += 1
    st["timestamps"].setdefault("opened", []).append(now)
    prior = int(st.get("prior_looks", 0))
    result = {
        "schema": SCHEMA,
        "single_look": prior + st["opens"] == 1,
        "run": fr["run"],
        "opened_at": now,
        "opens": st["opens"],
        "round": int(st.get("round", 1)),
        "prior_looks": prior,
        "look": prior + st["opens"],
        "forced": forced,
        "forced_overrides": len(st["forced"]),
        "code_changed_since_freeze": code_changed,
        "git_commit": fr["git_commit"],
        "customers": len(truth),
        "pv": int(truth["pv_installed"].sum()),
        "score": summary["score"],
        "ci90": [lo, hi],
        "bootstrap": {"draws": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "unit": "customer"},
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1": summary["f1"],
        "D": summary["D"],
        "P": summary["P"],
        "counts": summary["counts"],
        "lost": summary["lost"],
        "largest": summary["largest"],
        "failure_modes": diagnose.failure_modes(summary),
        "nested": {"score": (nested or {}).get("score"), "ci90": (nested or {}).get("ci90")},
        "vs_nested": position,
        "decisions_used": st["decisions_used"],
        "decision_budget": st["config"]["decision_budget"],
    }
    _write_json(d / RESULT_FILE, result)
    _write_json(model.parent / RESULT_FILE, result)
    st["lockbox"] = {"score": summary["score"], "ci90": [lo, hi], "vs_nested": position, "opened_at": now}
    _advance(st, "opened")
    _save_state(d, st)
    c = summary["counts"]
    look = f"look {prior + st['opens']} (round {result['round']}; earlier rounds' looks changed no decision of this round)" if prior else "single look"
    print(f"lockbox: run={fr['run']} customers={len(truth)} pv={result['pv']} opens={st['opens']} ({look})")
    print(
        f"lockbox: score={summary['score']:.2f} ci90=[{lo:.2f},{hi:.2f}] f1={_fmt(summary['f1'], 3)} "
        f"D={_fmt(summary['D'], 3)} P={_fmt(summary['P'], 3)} precision={_fmt(summary['precision'], 3)} "
        f"recall={_fmt(summary['recall'], 3)} TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']}"
    )
    n = nested or {}
    print(f"lockbox: nested score on dev={_fmt(n.get('score'))} ci90={_ci(n.get('ci90'))} -> lock-box {position}")
    for mode in result["failure_modes"]:
        print(f"failure_mode: {mode}")
    if position == "below":
        print("note: possible overfitting; the frozen model is still the submission (no re-tuning on the lock-box)")
    print(f"lockbox: {RESULT_FILE} written to the private --dir and next to the frozen model; predictions stay private")
    return 0


def final(args) -> int:
    d = _private_dir(args.dir)
    st = _load_state(d)
    if _stage(st) < STAGES.index("frozen"):
        raise _refuse("freeze a run (and open the lock-box) before the final refit")
    if not st["opens"]:
        if not args.force:
            raise _refuse("open the lock-box before the final refit, or --force (recorded)")
        _force(st, "final", "final refit before the lock-box was opened")
    fr = st["frozen"]
    unlabelled = str(Path(args.unlabelled).expanduser().resolve()) if args.unlabelled else fr["unlabelled"]
    labels = load_labels(args.labels)
    sealed = set(load_labels(d / "dev_labels.csv")["customer_id"]) | set(load_labels(d / "lockbox_labels.csv")["customer_id"])
    missing = len(sealed - set(labels["customer_id"]))
    if missing:
        print(f"warning: {missing} sealed customers have no label in --labels")
    out = d / "final"
    out.mkdir(exist_ok=True)
    for name in ("model.json", "oof.csv", "diagnosis.json"):
        (out / name).unlink(missing_ok=True)
    ext = [*(["--ext-labels", str(Path(args.ext_labels).expanduser().resolve())] if args.ext_labels else []),
           *(["--no-extensions"] if args.no_extensions else [])]
    argv = [
        "--timeseries", str(Path(args.energy).expanduser().resolve()), "--labels", str(Path(args.labels).expanduser().resolve()),
        "--out", str(out / "model.json"), *(["--unlabelled", unlabelled] if unlabelled else []), *fr["flags"], *ext,
        *(["--cv", "--oof-out", str(out / "oof.csv")] if args.cv else []),
    ]
    text = _run(_repo_module("train").main, argv)
    if not (out / "model.json").is_file():
        raise SystemExit("final: train.py wrote no model")
    if args.cv and (out / "oof.csv").is_file():
        print()
        _run(diagnose.main, [str(out / "oof.csv"), str(Path(args.labels).expanduser().resolve()), "--out-dir", str(out)])
    now = _now()
    st["final"] = {
        "time": now, "labelled": len(labels), "flags": fr["flags"], "unlabelled": unlabelled, "cv": bool(args.cv),
        "extensions": "off" if args.no_extensions else ("labelled" if args.ext_labels else "no extension labels"),
        "nested": parse_nested(text) if args.cv else None, "model_sha256": _sha256_file(out / "model.json"),
    }
    st["timestamps"]["final"] = now
    _advance(st, "final")
    _save_state(d, st)
    print(f"final: frozen run={fr['run']} refit on labelled={len(labels)} flags={' '.join(fr['flags']) or '(defaults)'}")
    print("final: model written to the private --dir (final/model.json); export it with:")
    print(f"  sh scripts/export_submission.sh {fr['git_commit'][:12]} NW26_DIR {out}")
    if not args.no_extensions:
        print("note: the model has an extension section; export a commit that contains pvdetect/extensions.py "
              "(the frozen commit may predate it), e.g. the current HEAD if it descends from the frozen commit")
    return 0


def status(args) -> int:
    d = _private_dir(args.dir)
    st = _load_state(d)
    cfg = st["config"]
    rnd = f" round={st.get('round', 1)} prior_looks={st.get('prior_looks', 0)}" if st.get("round", 1) > 1 else ""
    print(f"stage: {st['stage']} (sealed {st['timestamps']['sealed']}) config_hash={st['config_hash'][:12]}{rnd}")
    for name, info in st["runs"].items():
        n = info["nested"] or {}
        print(
            f"run: {name} nested score={_fmt(n.get('score'))} ci90={_ci(n.get('ci90'))} "
            f"flags={' '.join(info['flags']) or '(defaults)'}{' +unlabelled' if info['unlabelled'] else ''}"
        )
    print(f"decisions: used={st['decisions_used']}/{cfg['decision_budget']} forced_overrides={len(st['forced'])}")
    if st["frozen"]:
        print(f"frozen: run={st['frozen']['run']} commit={st['frozen']['git_commit'][:12]}")
    lb = st["lockbox"]
    if st["opens"]:
        print(f"lockbox opened: yes opens={st['opens']} score={_fmt(lb['score'])} ci90={_ci(lb['ci90'])} vs_nested={lb['vs_nested']}")
    else:
        print("lockbox opened: no")
    print(f"final: {'yes' if st['final'] else 'no'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", required=True, help="private protocol folder (outside the repository)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("seal", parents=[common], help="stratified dev / lock-box split, split files, seal.json")
    p.add_argument("--energy", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p = sub.add_parser("round", parents=[common], help="new development round on an existing seal")
    p.add_argument("--from", dest="source", required=True, help="previous protocol folder (its seal is reused)")
    p.add_argument("--config", default=str(REPO / "configs" / "lockbox_round2.json"))
    p = sub.add_parser("develop", parents=[common], help="nested CV of one candidate on the dev set")
    p.add_argument("name")
    p.add_argument("--unlabelled")
    p = sub.add_parser("compare", parents=[common], help="paired comparison of two runs (one decision)")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--force", action="store_true")
    p.add_argument("--target", choices=compare_mod.COMPONENTS, help="component the candidate was built for")
    p = sub.add_parser("freeze", parents=[common], help="freeze one run and the code commit")
    p.add_argument("name")
    p = sub.add_parser("open", parents=[common], help="score the frozen run on the lock-box (once)")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("final", parents=[common], help="refit the frozen run on all customers")
    p.add_argument("--energy", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--unlabelled")
    p.add_argument("--cv", action="store_true")
    p.add_argument("--ext-labels", help="pv_extensions_labels_dev.csv, passed to train.py for the extension nowcaster")
    p.add_argument("--no-extensions", action="store_true", help="refit without the extension detector (header-only)")
    p.add_argument("--force", action="store_true")
    sub.add_parser("status", parents=[common], help="stage, runs, decisions, lock-box")
    args = ap.parse_args(argv)
    if extra and args.stage != "develop":
        ap.error("train.py flags after -- are accepted by develop only")
    if args.stage == "develop":
        return develop(args, extra)
    return {"seal": seal, "round": new_round, "compare": compare, "freeze": freeze, "open": open_lockbox, "final": final,
            "status": status}[
        args.stage
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
