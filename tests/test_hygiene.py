"""The commit-time hygiene check blocks organizer data and real customer ids.

Every fixture below is fabricated; the identifier under test is assembled at
runtime so this file itself passes the check.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_hygiene.py"


def _stage_and_check(repo: Path, relpath: str, content: str) -> subprocess.CompletedProcess[str]:
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "--", relpath], check=True)
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=repo, capture_output=True, text=True)


def test_staged_customer_identifier_is_refused_without_echoing_it(tmp_path: Path):
    identifier = "POD" + "12"
    result = _stage_and_check(tmp_path, "notes.md", f"the series of {identifier} looked odd\n")
    assert result.returncode == 1
    assert "notes.md" in result.stdout
    assert identifier not in result.stdout + result.stderr
    assert "looked odd" not in result.stdout + result.stderr


def test_staged_csv_is_refused(tmp_path: Path):
    result = _stage_and_check(tmp_path, "sample.csv", "customer_id;energy\nA;1\n")
    assert result.returncode == 1
    assert "sample.csv" in result.stdout


def test_staged_model_dump_under_artifacts_is_refused(tmp_path: Path):
    result = _stage_and_check(tmp_path, "artifacts/model.json", "{}\n")
    assert result.returncode == 1
    assert "artifacts/model.json" in result.stdout


def test_clean_python_file_passes(tmp_path: Path):
    result = _stage_and_check(tmp_path, "clean.py", "VALUE = 1\nSYNTHETIC_ID = 'PODSYN001'\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""


def test_whole_tracked_tree_is_clean():
    result = subprocess.run([sys.executable, str(SCRIPT), "--all"], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
