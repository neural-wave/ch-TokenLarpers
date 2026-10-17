from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import inference
import train
from pvdetect import lockbox
from pvdetect.evaluation import load_split_file, splits_from_fold_map
from pvdetect.gui import saved_lockbox
from pvdetect.io import load_labels
from pvdetect.output import write_extensions, write_oof, write_predictions
from pvdetect.synthetic import CustomerSpec, generate_customer, generate_dataset, write_dataset

NESTED = "nested: score=80.00±1.00 ci90=[70.00,90.00] f1=0.900 D=0.500 P=0.500 inner=80.00 gap=0.00 outer_informed_decisions=0"


def _config(tmp_path: Path, **kw) -> Path:
    cfg = {"seed": 2026, "lockbox_size": 4, "strata": ["pv_installed"], "outer_seeds": [17, 42], "outer_folds": 3,
           "decision_budget": 1} | kw
    path = tmp_path / "lockbox.json"
    path.write_text(json.dumps(cfg))
    return path


@pytest.fixture
def tiny(tmp_path):
    """12 three-day synthetic customers (5 labelled PV): enough to seal, not to train."""
    ids = [f"{i:06d}" for i in range(12)]
    pv = [i < 5 for i in range(12)]
    frames = [generate_customer(CustomerSpec(c, "2024-03-01", "2024-03-03", missing=False), i) for i, c in enumerate(ids)]
    energy = tmp_path / "energy.csv"
    pd.concat(frames, ignore_index=True).to_csv(energy, index=False, sep=";")
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "customer_id": ids,
            "pv_installed": ["TRUE" if p else "FALSE" for p in pv],
            "pv_installation_date": ["2023-05-01" if p else "" for p in pv],
            "pv_capacity_kwp": ["8.0" if p else "" for p in pv],
        }
    ).to_csv(labels, index=False, sep=";")
    return SimpleNamespace(energy=energy, labels=labels, ids=ids, dir=tmp_path / "private", config=_config(tmp_path))


def _lb(*argv) -> int:
    return lockbox.main([str(a) for a in argv])


def _seal(t) -> None:
    assert _lb("seal", "--dir", t.dir, "--energy", t.energy, "--labels", t.labels, "--config", t.config) == 0


def _state(t) -> dict:
    return json.loads((t.dir / "state.json").read_text())


def _arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


@pytest.fixture
def fake_train(monkeypatch):
    """Stands in for the nested run: perfect OOF predictions on the given split files (every third customer
    flipped with --no-e1e2), a placeholder model and a fixed `nested:` line."""
    calls = []

    def main(argv):
        calls.append(list(argv))
        labels = load_labels(_arg(argv, "--labels"))
        if _arg(argv, "--oof-out"):
            rows = []
            folds = _arg(argv, "--folds").split(",") if _arg(argv, "--folds") else []
            for r, path in enumerate(folds):
                fm = load_split_file(path, labels["customer_id"].tolist())
                flip = labels.index % 3 == 0 if "--no-e1e2" in argv else labels.index < 0
                pv = labels["pv_installed"] ^ flip
                rows.append(labels.assign(repeat=r, fold=labels["customer_id"].map(fm), prob=pv.astype(float), pv_installed=pv))
            write_oof(pd.concat(rows, ignore_index=True), Path(_arg(argv, "--oof-out")))
            print(NESTED)
        Path(_arg(argv, "--out")).write_text("{}")
        return 0

    monkeypatch.setattr(train, "main", main)
    return calls


@pytest.fixture
def fake_inference(monkeypatch):
    """Perfect lock-box predictions straight from the sealed labels next to the input file."""

    def run_inference(test_csv, model_path, out_dir, **kw):
        truth = load_labels(Path(test_csv).parent / "lockbox_labels.csv")
        write_predictions(truth, Path(out_dir))
        write_extensions(pd.DataFrame(columns=["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]), Path(out_dir))
        ids = pd.read_csv(test_csv, sep=";", dtype=str, usecols=["customer_id"])["customer_id"]
        return SimpleNamespace(customer_ids=sorted(set(ids)))

    monkeypatch.setattr(inference, "run_inference", run_inference)


def test_seal_stratifies_writes_valid_split_files_and_refuses_twice(tiny, capsys):
    _seal(tiny)
    d = tiny.dir
    box, dev = load_labels(d / "lockbox_labels.csv"), load_labels(d / "dev_labels.csv")
    assert (len(box), int(box["pv_installed"].sum())) == (4, 2)
    assert (len(dev), int(dev["pv_installed"].sum())) == (8, 3)
    dev_rows = set(pd.read_csv(d / "dev.csv", sep=";", dtype=str)["customer_id"])
    box_rows = set(pd.read_csv(d / "lockbox.csv", sep=";", dtype=str)["customer_id"])
    assert not dev_rows & box_rows and dev_rows | box_rows == set(tiny.ids)
    assert dev_rows == set(dev["customer_id"]) and box_rows == set(box["customer_id"])

    ids = dev["customer_id"].tolist()
    pv = set(dev.loc[dev["pv_installed"], "customer_id"])
    maps = []
    for seed in (17, 42):
        fm = load_split_file(d / f"splits_s{seed}.csv", ids)
        assert sorted(set(fm.values())) == [0, 1, 2]
        for _, test in splits_from_fold_map(fm, ids):
            assert set(test) & pv and set(test) - pv
        maps.append(fm)
    assert maps[0] != maps[1]

    seal = json.loads((d / "seal.json").read_text())
    assert seal["lockbox_csv_sha256"] == lockbox._sha256_file(d / "lockbox.csv")
    assert seal["lockbox_labels_sha256"] == lockbox._sha256_file(d / "lockbox_labels.csv")
    assert seal["lockbox_ids_sha256"] == lockbox._sha256_ids(box["customer_id"])
    st = _state(tiny)
    assert st["stage"] == "sealed" and st["config_hash"] == seal["config_hash"]
    out = capsys.readouterr().out
    assert "lockbox customers=4 pv=2" in out and not any(c in out for c in tiny.ids)

    with pytest.raises(SystemExit, match="already sealed"):
        _seal(tiny)


def test_dir_inside_repository_is_refused(tiny):
    with pytest.raises(SystemExit, match="--dir is inside the repository"):
        _lb("seal", "--dir", lockbox.REPO / "private_x", "--energy", tiny.energy, "--labels", tiny.labels)
    assert not (lockbox.REPO / "private_x").exists()


def test_stage_order_reserved_flags_and_decision_budget(tiny, fake_train, capsys):
    d = tiny.dir
    with pytest.raises(SystemExit, match="run `seal` first"):
        _lb("develop", "--dir", d, "base")
    _seal(tiny)
    with pytest.raises(SystemExit, match="freeze a run before opening"):
        _lb("open", "--dir", d)
    with pytest.raises(SystemExit, match="freeze a run"):
        _lb("final", "--dir", d, "--energy", tiny.energy, "--labels", tiny.labels)
    with pytest.raises(SystemExit, match="no run named"):
        _lb("freeze", "--dir", d, "base")

    assert _lb("develop", "--dir", d, "base") == 0
    assert _lb("develop", "--dir", d, "cand", "--", "--no-e1e2") == 0
    argv = fake_train[0]
    assert "--cv" in argv and _arg(argv, "--folds") == f"{d / 'splits_s17.csv'},{d / 'splits_s42.csv'}"
    assert _arg(argv, "--timeseries") == str(d / "dev.csv") and _arg(argv, "--labels") == str(d / "dev_labels.csv")
    assert fake_train[1][-1] == "--no-e1e2"
    st = _state(tiny)
    assert st["runs"]["base"]["nested"]["ci90"] == [70.0, 90.0] and st["runs"]["cand"]["flags"] == ["--no-e1e2"]
    assert (d / "runs" / "base" / "diagnosis.json").is_file()
    with pytest.raises(SystemExit, match="exists"):
        _lb("develop", "--dir", d, "base")
    with pytest.raises(SystemExit, match="sets"):
        _lb("develop", "--dir", d, "other", "--", "--folds=x.csv")

    capsys.readouterr()
    assert _lb("compare", "--dir", d, "base", "cand") == 0
    out = capsys.readouterr().out
    assert "verdict: cand vs base = " in out and "decisions used 1/1" in out
    assert (d / "runs" / "cand" / "compare_vs_base" / "changes.csv").is_file()
    assert not any(c in out for c in tiny.ids)
    with pytest.raises(SystemExit, match="budget exhausted"):
        _lb("compare", "--dir", d, "base", "cand")
    assert _lb("compare", "--dir", d, "base", "cand", "--force") == 0
    st = _state(tiny)
    assert st["decisions_used"] == 2 and [f["stage"] for f in st["forced"]] == ["compare"]
    assert [x["forced"] for x in st["decisions"]] == [False, True]

    assert _lb("freeze", "--dir", d, "base") == 0
    st = _state(tiny)
    assert st["stage"] == "frozen" and st["frozen"]["run"] == "base" and st["frozen"]["git_commit"]
    for argv in (("develop", "--dir", d, "late"), ("compare", "--dir", d, "base", "cand"), ("freeze", "--dir", d, "cand")):
        with pytest.raises(SystemExit, match="stage is frozen"):
            _lb(*argv)
    with pytest.raises(SystemExit, match="open the lock-box before the final refit"):
        _lb("final", "--dir", d, "--energy", tiny.energy, "--labels", tiny.labels)


def test_open_verifies_seal_looks_once_and_final_refits(tiny, fake_train, fake_inference, capsys):
    d = tiny.dir
    _seal(tiny)
    _lb("develop", "--dir", d, "base")
    _lb("freeze", "--dir", d, "base")

    original = (d / "lockbox.csv").read_bytes()
    (d / "lockbox.csv").write_bytes(original + b"\n")
    with pytest.raises(SystemExit, match="lockbox.csv does not match seal.json"):
        _lb("open", "--dir", d)
    (d / "lockbox.csv").write_bytes(original)
    model = d / "runs" / "base" / "model.json"
    model.write_text("{} ")
    with pytest.raises(SystemExit, match="model.json changed after freeze"):
        _lb("open", "--dir", d)
    model.write_text("{}")

    capsys.readouterr()
    assert _lb("open", "--dir", d) == 0
    out = capsys.readouterr().out
    assert "-> lock-box above" in out and not any(c in out for c in tiny.ids)
    res = json.loads((d / "lockbox_result.json").read_text())
    assert res == json.loads((d / "runs" / "base" / "lockbox_result.json").read_text())
    # A resample without PV customers scores 0 (no true positive), so only the upper end is 100 here.
    assert res["score"] == pytest.approx(100.0) and res["ci90"][1] == pytest.approx(100.0)
    assert (res["customers"], res["pv"], res["vs_nested"], res["forced"], res["opens"]) == (4, 2, "above", False, 1)
    assert res["nested"]["ci90"] == [70.0, 90.0] and res["single_look"] is True
    assert not any(c in json.dumps(res) for c in tiny.ids)
    assert _state(tiny)["stage"] == "opened"
    panel = saved_lockbox(model)
    json.dumps(panel, allow_nan=False)
    assert (panel["score"], panel["vs_nested"], panel["forced"], panel["customers"]) == (res["score"], "above", False, 4)
    assert set(panel) <= set(res) and saved_lockbox(d / "final" / "model.json") is None

    with pytest.raises(SystemExit, match="already opened"):
        _lb("open", "--dir", d)
    assert _lb("open", "--dir", d, "--force") == 0
    st = _state(tiny)
    assert st["opens"] == 2 and [f["stage"] for f in st["forced"]] == ["open"]
    assert json.loads((d / "lockbox_result.json").read_text())["forced"] is True

    capsys.readouterr()
    assert _lb("final", "--dir", d, "--energy", tiny.energy, "--labels", tiny.labels) == 0
    out = capsys.readouterr().out
    assert "scripts/export_submission.sh" in out and (d / "final" / "model.json").is_file()
    argv = fake_train[-1]
    assert _arg(argv, "--timeseries") == str(tiny.energy.resolve()) and "--cv" not in argv
    st = _state(tiny)
    assert st["stage"] == "final" and st["final"]["labelled"] == 12
    assert _lb("status", "--dir", d) == 0
    out = capsys.readouterr().out
    assert "stage: final" in out and "lockbox opened: yes opens=2" in out and "decisions: used=0/1" in out


def test_round_reuses_the_seal_carries_looks_and_keeps_the_split(tiny, fake_train, fake_inference, capsys, tmp_path):
    d, r2 = tiny.dir, tmp_path / "private2"
    _seal(tiny)
    _lb("develop", "--dir", d, "base")
    _lb("freeze", "--dir", d, "base")
    _lb("open", "--dir", d)
    with pytest.raises(SystemExit, match=r"config differs in \['outer_folds'\]"):
        _lb("round", "--dir", r2, "--from", d, "--config", _config(tmp_path, outer_folds=4))
    assert not (r2 / "state.json").exists()
    (d / "lockbox_labels.csv").write_bytes((d / "lockbox_labels.csv").read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="lockbox_labels.csv in --from does not match"):
        _lb("round", "--dir", r2, "--from", d, "--config", _config(tmp_path, decision_budget=5))
    (d / "lockbox_labels.csv").write_bytes((d / "lockbox_labels.csv").read_bytes()[:-1])

    assert _lb("round", "--dir", r2, "--from", d, "--config", _config(tmp_path, decision_budget=5)) == 0
    with pytest.raises(SystemExit, match="already holds a protocol"):
        _lb("round", "--dir", r2, "--from", d, "--config", _config(tmp_path, decision_budget=5))
    st = _state(SimpleNamespace(dir=r2))
    assert (st["stage"], st["round"], st["prior_looks"], st["opens"], st["config"]["decision_budget"]) == ("sealed", 2, 1, 0, 5)
    for name in ("dev.csv", "lockbox.csv", "lockbox_labels.csv", "splits_s17.csv", "splits_s42.csv", "seal.json"):
        assert (r2 / name).read_bytes() == (d / name).read_bytes()

    _lb("develop", "--dir", r2, "base")
    _lb("freeze", "--dir", r2, "base")
    capsys.readouterr()
    assert _lb("open", "--dir", r2) == 0  # the round's first open needs no --force
    out = capsys.readouterr().out
    assert "look 2 (round 2" in out
    res = json.loads((r2 / "lockbox_result.json").read_text())
    assert (res["round"], res["prior_looks"], res["look"], res["opens"], res["single_look"], res["forced"]) == (2, 1, 2, 1, False, False)
    panel = saved_lockbox(r2 / "runs" / "base" / "model.json")
    assert (panel["look"], panel["round"]) == (2, 2)
    with pytest.raises(SystemExit, match="already opened"):
        _lb("open", "--dir", r2)
    assert _lb("status", "--dir", r2) == 0
    assert "round=2 prior_looks=1" in capsys.readouterr().out


def test_ci_position_against_nested_interval():
    nested = {"ci90": [60.0, 70.0]}
    assert [lockbox.vs_nested(s, nested) for s in (59.9, 60.0, 70.0, 70.1)] == ["below", "inside", "inside", "above"]
    assert lockbox.vs_nested(65.0, None) == "unavailable" and lockbox.vs_nested(65.0, {"ci90": [None, 1.0]}) == "unavailable"
    assert lockbox.parse_nested(NESTED) == {
        "score": 80.0, "sd": 1.0, "ci90": [70.0, 90.0], "f1": 0.9, "D": 0.5, "P": 0.5, "outer_informed_decisions": 0
    }
    assert lockbox.parse_nested("cv skipped: fewer than 2 customers in a class") is None


def test_end_to_end_seal_develop_freeze_open_final_on_synthetic(tmp_path, capsys):
    """One real pass with small n: 16 dev-seed customers, a lock-box of 4, one split seed with 2 folds."""
    data = tmp_path / "data"
    write_dataset(data, *generate_dataset(16, seed=5))
    cfg = _config(tmp_path, outer_seeds=[17], outer_folds=2)
    d = tmp_path / "private"
    fast = ["--", "--no-ctune", "--no-offset-tri", "--no-weekly"]
    assert _lb("seal", "--dir", d, "--energy", data / "energy.csv", "--labels", data / "labels.csv", "--config", cfg) == 0
    assert _lb("develop", "--dir", d, "base", *fast) == 0
    assert (d / "runs" / "base" / "oof.csv").is_file() and (d / "runs" / "base" / "diagnosis.json").is_file()
    nested = _state(SimpleNamespace(dir=d))["runs"]["base"]["nested"]
    assert nested is not None and len(nested["ci90"]) == 2
    assert _lb("freeze", "--dir", d, "base") == 0
    assert _lb("open", "--dir", d) == 0
    res = json.loads((d / "runs" / "base" / "lockbox_result.json").read_text())
    assert res["customers"] == 4 and res["pv"] == 2 and res["vs_nested"] in ("inside", "below", "above")
    assert res["ci90"][0] <= res["score"] <= res["ci90"][1]
    assert (d / "lockbox_predictions" / "predictions.csv").is_file()
    assert _lb("final", "--dir", d, "--energy", data / "energy.csv", "--labels", data / "labels.csv") == 0
    assert (d / "final" / "model.json").is_file()
    st = _state(SimpleNamespace(dir=d))
    assert st["stage"] == "final" and st["final"]["labelled"] == 16 and st["forced"] == []
    out = capsys.readouterr().out
    ids = load_labels(data / "labels.csv")["customer_id"]
    assert not any(f";{c};" in out or f" {c} " in out for c in ids)
