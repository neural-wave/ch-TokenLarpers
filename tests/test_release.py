"""Release safety: saved model files (Step 2.5-D) and the audit of an exported tree (Step 2.5-X).

Every fixture is fabricated; the identifier under test is assembled at
runtime so this file itself passes the hygiene check.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pvdetect.procedure as procedure
from pvdetect.model import referenced_files, save_model
from pvdetect.pipeline import build_feature_table
from pvdetect.synthetic import generate_dataset
from tests.helpers import as_loaded
from tests.test_evaluation import typed_labels

REPO = Path(__file__).resolve().parents[1]
RELEASE_SCRIPT = REPO / "scripts" / "check_release.py"
_RELEASE_ALLOW = ["artifacts/model.json", "artifacts/catboost_clf.cbm"]


def _model_relabelled_data():
    ts, labels, _ = generate_dataset(12, seed=5)
    mapping = {cid: f"SYNREL-{i:03d}" for i, cid in enumerate(sorted(labels["customer_id"]))}
    ts = ts.assign(customer_id=ts["customer_id"].map(mapping))
    labels = labels.assign(customer_id=labels["customer_id"].map(mapping))
    return build_feature_table(as_loaded(ts)), typed_labels(labels)


def _model_saved_files(model: dict, path: Path) -> list[Path]:
    save_model(model, path)
    files = [path] + [path.parent / f for f in referenced_files(model)]
    assert all(f.exists() for f in files)
    return files


def _model_leaked_ids(files: list[Path], ids: list[str]) -> list[tuple[str, str]]:
    return [(f.name, cid) for f in files for cid in ids if cid.encode() in f.read_bytes()]


def test_saved_model_files_carry_no_training_customer_ids(tmp_path, monkeypatch):
    """No saved model file (`model.json` or any referenced `.cbm`) contains a training customer_id.

    Pitfalls handled:
    1. Synthetic ids are zero-padded digits (`000003`) that can occur inside JSON floats, so
       customers are relabelled consistently in time series and labels with distinctive ids
       (`SYNREL-007`) before the table is built; a hit is then a real leak, not a digit run.
    2. A 12-customer fit usually selects logreg/ridge and writes no `.cbm`, so a second fit
       forces CatBoost classifier and capacity through the real `fit`/`save_model` path
       (only the inner selection is replaced) and asserts both `.cbm` files exist.

    This check is necessary but not sufficient: Step 2.5-G adds human review and organizer approval.
    """
    table, labels = _model_relabelled_data()
    ids = labels["customer_id"].tolist()
    assert all(cid.startswith("SYNREL-") for cid in ids)

    selected = _model_saved_files(procedure.fit(table, labels, ids), tmp_path / "selected" / "model.json")
    assert _model_leaked_ids(selected, ids) == []

    forced_sel = {"classifier": "catboost", "threshold": 0.5, "capacity": "catboost", "kappa": 1.0, "insufficient_rule": "weak_candidate"}
    monkeypatch.setattr(procedure, "_inner_oof", lambda *args: None)
    monkeypatch.setattr(procedure, "_select", lambda oof: dict(forced_sel))
    forced = procedure.fit(table, labels, ids)
    assert forced["classifier"]["kind"] == "catboost" and forced["capacity"]["kind"] == "catboost"
    files = _model_saved_files(forced, tmp_path / "catboost" / "model.json")
    assert sorted(f.name for f in files if f.suffix == ".cbm") == ["catboost_cap.cbm", "catboost_clf.cbm"]
    assert _model_leaked_ids(files, ids) == []


def _release_tree(dest: Path) -> Path:
    (dest / "pvdetect").mkdir(parents=True)
    (dest / "pvdetect" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (dest / "artifacts").mkdir()
    (dest / "artifacts" / "model.json").write_text('{"version": 2}\n', encoding="utf-8")
    (dest / "artifacts" / "catboost_clf.cbm").write_bytes(b"\x00fabricated model bytes\x01")
    return dest


def _release_check(dest: Path, allow: list[str] = _RELEASE_ALLOW) -> subprocess.CompletedProcess[str]:
    args = [sys.executable, str(RELEASE_SCRIPT), str(dest)]
    for path in allow:
        args += ["--allow", path]
    return subprocess.run(args, capture_output=True, text=True)


def test_release_model_json_and_allowed_cbm_pass(tmp_path: Path):
    result = _release_check(_release_tree(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_release_extra_csv_is_refused(tmp_path: Path):
    dest = _release_tree(tmp_path)
    (dest / "sample.csv").write_text("customer_id;energy\nA;1\n", encoding="utf-8")
    result = _release_check(dest)
    assert result.returncode == 1
    assert "sample.csv" in result.stdout


def test_release_identifier_in_allowed_file_is_refused_without_echoing_it(tmp_path: Path):
    dest = _release_tree(tmp_path)
    identifier = "POD" + "12"
    (dest / "artifacts" / "model.json").write_text(f'{{"ids": ["{identifier}"]}}\n', encoding="utf-8")
    result = _release_check(dest)
    assert result.returncode == 1
    assert "artifacts/model.json" in result.stdout
    assert identifier not in result.stdout + result.stderr


def test_release_cbm_not_on_allow_list_is_refused(tmp_path: Path):
    dest = _release_tree(tmp_path)
    (dest / "artifacts" / "catboost_cap.cbm").write_bytes(b"\x00fabricated\x01")
    result = _release_check(dest)
    assert result.returncode == 1
    assert "artifacts/catboost_cap.cbm" in result.stdout


def test_release_missing_allowed_file_is_refused(tmp_path: Path):
    dest = _release_tree(tmp_path)
    result = _release_check(dest, _RELEASE_ALLOW + ["artifacts/catboost_cap.cbm"])
    assert result.returncode == 1
    assert "artifacts/catboost_cap.cbm" in result.stdout


def test_release_symlink_out_of_the_tree_is_refused(tmp_path: Path):
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "energy.csv").write_text("customer_id;energy\nA;1\n", encoding="utf-8")
    dest = _release_tree(tmp_path / "export")
    (dest / "data").symlink_to(outside, target_is_directory=True)
    (dest / "gone.txt").symlink_to(tmp_path / "missing.txt")
    result = _release_check(dest)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "symlink: data" in result.stdout and "symlink: gone.txt" in result.stdout


def test_release_identifier_in_file_name_is_refused_without_echoing_it(tmp_path: Path):
    dest = _release_tree(tmp_path)
    identifier = "POD" + "12"
    (dest / "review").mkdir()
    (dest / "review" / f"{identifier}.png").write_bytes(b"\x89fabricated image\x00")
    result = _release_check(dest)
    assert result.returncode == 1
    assert "customer identifier in path" in result.stdout
    assert identifier not in result.stdout + result.stderr


def test_release_model_file_listing_mirrors_referenced_files():
    """The export lists model files with check_release.model_files (stdlib only); it must agree with
    pvdetect.model.referenced_files and refuse unsafe names."""
    sys.path.insert(0, str(RELEASE_SCRIPT.parent))
    from check_release import model_files

    both = {"classifier": {"catboost_file": "catboost_clf.cbm"}, "capacity": {"catboost_file": "catboost_cap.cbm"}}
    for model in (both, {"classifier": {"catboost_file": None}, "capacity": {}}, {}):
        assert model_files(model) == referenced_files(model)
    for unsafe in ("../escape.cbm", "/abs.cbm", "a\\b.cbm"):
        try:
            model_files({"classifier": {"catboost_file": unsafe}})
        except ValueError:
            continue
        raise AssertionError(f"unsafe name accepted: {unsafe!r}")


def test_export_runs_without_site_packages(tmp_path: Path):
    """The export works with an interpreter that cannot import numpy or pvdetect's dependencies."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.json").write_text(
        '{"version": 2, "classifier": {"catboost_file": null}, "capacity": {"catboost_file": "catboost_cap.cbm"}}\n',
        encoding="utf-8",
    )
    (model_dir / "catboost_cap.cbm").write_bytes(b"\x00fabricated model bytes\x01")
    bare = tmp_path / "bare_python"
    bare.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n', encoding="utf-8")
    bare.chmod(0o755)
    probe = subprocess.run([str(bare), "-c", "import numpy"], capture_output=True)
    assert probe.returncode != 0, "the bare interpreter must not see site-packages"
    dest = tmp_path / "export"
    result = subprocess.run(
        ["sh", str(REPO / "scripts" / "export_submission.sh"), "HEAD", str(dest), str(model_dir)],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PYTHON": str(bare)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in (dest / "artifacts").iterdir()) == ["catboost_cap.cbm", "model.json"]
