"""Local GUI service: outputs, failures, lifecycle and request-boundary behaviour (synthetic data only)."""

from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pytest

import train
from pvdetect.gui import create_server
from pvdetect.io import load_labels

PRED_HEADER = "customer_id;pv_installed;pv_installation_date;pv_capacity_kwp"
EXT_HEADER = "customer_id;extension_date;pv_capacity_after_extension_kwp"
TERMINAL = {"complete", "failed", "cancelled"}
CONTRACT_CHECKS = ("header", "one_row_per_customer", "unique_ids", "false_blanks", "true_fields", "extensions_header")


class Client:
    def __init__(self, srv):
        self.srv = srv
        self.port = srv.port

    def request(self, method, path, body=None, headers=None, *, token=True, origin=True, host=None):
        h = {"Host": host or f"127.0.0.1:{self.port}"}
        if token:
            h["X-PVD-Token"] = self.srv.token
        if origin and method == "POST":
            h["Origin"] = f"http://127.0.0.1:{self.port}"
        h.update(headers or {})
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        try:
            conn.request(method, path, body=body, headers=h)
            resp = conn.getresponse()
            return resp.status, resp.headers, resp.read()
        finally:
            conn.close()

    def json(self, method, path, **kw):
        status, _, data = self.request(method, path, **kw)
        return status, json.loads(data)

    def upload(self, csv_path: Path, name: str = "energy.csv"):
        headers = {
            "Content-Length": str(csv_path.stat().st_size),
            "Content-Type": "text/csv",
            "X-PVD-Filename": quote(name),
        }
        with open(csv_path, "rb") as f:
            return self.json("POST", "/api/jobs/analysis", body=f, headers=headers)

    def wait(self, timeout: float = 240) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, body = self.json("GET", "/api/jobs/current")
            assert status == 200
            if body["job"]["state"] in TERMINAL:
                return body["job"]
            time.sleep(0.2)
        raise AssertionError("job did not finish in time")

    def file(self, job_id: str, name: str):
        return self.request("GET", f"/api/jobs/{job_id}/files/{name}")


def _start(model_path: Path, session_root: Path, labels_path: Path | None = None):
    srv = create_server(model_path, port=0, session_root=session_root, labels_path=labels_path)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, thread


def _stop(srv, thread) -> None:
    srv.shutdown_and_cleanup()
    thread.join(10)


def _digest(paths: list[Path]) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _model_files(model: Path) -> list[Path]:
    return sorted(p for p in model.parent.iterdir() if p.is_file())


def _session_files(srv) -> list[Path]:
    return [p for p in srv.session_dir.rglob("*") if p.is_file()]


def _assert_valid_predictions(text: str, input_ids: list[str]) -> list[list[str]]:
    lines = text.split("\n")
    assert lines[0] == PRED_HEADER and lines[-1] == ""
    rows = [line.split(";") for line in lines[1:-1]]
    assert all(len(r) == 4 for r in rows)
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)) == len(input_ids)
    assert sorted(ids) == sorted(input_ids)  # exact strings: leading zeros preserved
    for _, flag, date, kwp in rows:
        assert flag in ("TRUE", "FALSE")
        if flag == "FALSE":
            assert date == "" and kwp == ""
        else:
            assert len(date) == 10 and dt.date.fromisoformat(date)
            assert float(kwp) > 0
    return rows


def _assert_sanitized(obj, *private: Path) -> None:
    text = json.dumps(obj)
    for p in private:
        assert str(p) not in text
    assert "Traceback" not in text


@pytest.fixture(scope="module")
def trained(syn_train, tmp_path_factory) -> Path:
    model = tmp_path_factory.mktemp("gui_model") / "model.json"
    rc = train.main(
        ["--timeseries", str(syn_train / "energy.csv"), "--labels", str(syn_train / "labels.csv"), "--out", str(model)]
    )
    assert rc == 0
    return model


@pytest.fixture
def serve(tmp_path):
    started = []

    def _serve(model_path: Path) -> Client:
        srv, thread = _start(model_path, tmp_path / "session")
        started.append((srv, thread))
        return Client(srv)

    yield _serve
    for srv, thread in started:
        _stop(srv, thread)


@pytest.fixture(scope="module")
def analysis_done(trained, syn_test, tmp_path_factory):
    """One completed private analysis of the synthetic test set with the configured synthetic model."""
    srv, thread = _start(trained, tmp_path_factory.mktemp("session"))
    client = Client(srv)
    energy = syn_test / "energy.csv"
    before = _digest([energy, *_model_files(trained)])
    status, body = client.upload(energy, name="../private\x07/dir/energy.csv")
    assert status == 202 and body["job"]["state"] == "running"
    job = client.wait()
    assert job["state"] == "complete", job["error"]
    yield {"client": client, "job": job, "energy": energy, "before": before}
    _stop(srv, thread)


def test_analysis_is_diagnosed_against_launch_labels(trained, syn_test, tmp_path):
    from pvdetect.io import load_labels
    from pvdetect.scoring import main_score

    srv, thread = _start(trained, tmp_path / "session", labels_path=syn_test / "labels.csv")
    try:
        client = Client(srv)
        assert client.upload(syn_test / "energy.csv")[0] == 202
        job = client.wait()
        assert job["state"] == "complete", job["error"]
        status, res = client.json("GET", f"/api/jobs/{job['id']}/result")
        assert status == 200
        dg = res["diagnosis"]
        assert dg["available"] and dg["matched"] == 12 and dg["unlabelled"] == 0
        _, _, pred = client.file(job["id"], "predictions.csv")
        (tmp_path / "p.csv").write_bytes(pred)
        official = main_score(load_labels(tmp_path / "p.csv"), load_labels(syn_test / "labels.csv"))
        assert dg["score"] == pytest.approx(official["score"])
        assert (dg["counts"]["TP"], dg["counts"]["FP"], dg["counts"]["FN"]) == (official["tp"], official["fp"], official["fn"])
    finally:
        _stop(srv, thread)


def test_completed_analysis_exposes_valid_outputs(analysis_done, syn_test, trained):
    client, job = analysis_done["client"], analysis_done["job"]
    assert job["kind"] == "analysis" and job["input_name"] == "energy.csv"
    assert all(s["status"] == "done" for s in job["stages"])
    input_ids = load_labels(syn_test / "labels.csv")["customer_id"].tolist()
    assert any(c.startswith("0") for c in input_ids)

    status, result = client.json("GET", f"/api/jobs/{job['id']}/result")
    assert status == 200
    assert result["job_id"] == job["id"] and result["synthetic"] is None
    assert result["summary"]["customers"] == len(input_ids)
    assert result["summary"]["session"] == "private"
    assert result["summary"]["pv"] + result["summary"]["non_pv"] == len(input_ids)
    passed = {c["key"]: c["passed"] for c in result["checks"]}
    assert all(passed[k] for k in CONTRACT_CHECKS)
    assert result["outputs"]["predictions"] == {"valid": True, "rows": len(input_ids)}
    assert all(c["truth"] is None for c in result["customers"])

    status, headers, data = client.file(job["id"], "predictions.csv")
    assert status == 200
    assert headers["Content-Type"].startswith("text/csv")
    assert headers["Content-Disposition"] == 'attachment; filename="predictions.csv"'
    rows = _assert_valid_predictions(data.decode("utf-8"), input_ids)
    assert [c["customer_id"] for c in result["customers"]] == [r[0] for r in rows]
    for c, (_, flag, date, kwp) in zip(result["customers"], rows):
        assert c["pv_installed"] is (flag == "TRUE")
        assert c["installation_date"] == (date or None)
        assert c["capacity_kwp"] == (float(kwp) if kwp else None)

    status, _, ext = client.file(job["id"], "pv_extensions_predictions.csv")
    assert status == 200
    ext_lines = ext.decode("utf-8").splitlines()
    assert ext_lines[0] == EXT_HEADER
    assert len(ext_lines) - 1 == result["outputs"]["extensions"]["rows"]

    assert _digest([analysis_done["energy"], *_model_files(trained)]) == analysis_done["before"]


def test_diagnostics_for_pv_and_non_pv_customers(analysis_done):
    client, job = analysis_done["client"], analysis_done["job"]
    _, result = client.json("GET", f"/api/jobs/{job['id']}/result")
    pv = next(c for c in result["customers"] if c["pv_installed"])
    non_pv = next(c for c in result["customers"] if not c["pv_installed"])
    for row in (pv, non_pv):
        status, diag = client.json(
            "GET", f"/api/jobs/{job['id']}/customers/{quote(row['customer_id'], safe='')}/diagnostics"
        )
        assert status == 200
        assert diag["customer_id"] == row["customer_id"] and diag["prediction"] == row
        assert diag["usable"] is True and diag["truth"] is None and diag["markers"]["known_date"] is None
        assert diag["markers"]["predicted_date"] == row["installation_date"]
        for block in (diag["series"], diag["weekly"]):
            assert len(block["dates"]) == len(block["total_kwh"]) == len(block["midday_kw"]) > 0
        assert len(diag["series"]["dates"]) <= 1500 or diag["series"]["resolution"] == "weekly"
        assert diag["coverage"]["valid_days"] <= diag["coverage"]["days_with_data"] <= diag["coverage"]["span_days"]
        profile = diag["profile"]
        if profile["available"]:
            assert len(profile["before"]["kw"]) == len(profile["after"]["kw"]) == 96
            assert profile["before"]["days"] >= 14 and profile["after"]["days"] >= 14
        else:
            assert profile["before"] is None and profile["after"] is None and profile["reason"]
    _, diag = client.json("GET", f"/api/jobs/{job['id']}/customers/{non_pv['customer_id']}/diagnostics")
    assert diag["profile"]["available"] is False

    for unknown in ("does-not-exist", quote("../0", safe=""), quote(non_pv["customer_id"] + " ", safe="")):
        status, body = client.json("GET", f"/api/jobs/{job['id']}/customers/{unknown}/diagnostics")
        assert status == 404 and body["error"]["code"] == "unknown_customer"


def test_downloads_accept_only_the_two_output_names(analysis_done):
    client, job = analysis_done["client"], analysis_done["job"]
    for name in (
        "..%2F..%2F..%2Finference.py",
        "%2E%2E%2Fresult.json",
        "result.json",
        "input.csv",
        "..",
        "PREDICTIONS.CSV",
        "predictions.csv%00",
        "diag%2F0.json",
        "..%2Fout%2Fpredictions.csv",
    ):
        status, body = client.json("GET", f"/api/jobs/{job['id']}/files/{name}")
        assert status == 404 and "error" in body, name
    status, _ = client.json("GET", f"/api/jobs/{job['id']}/files/../../../inference.py")
    assert status == 404
    status, body = client.json("GET", f"/api/jobs/{'0' * 32}/files/predictions.csv")
    assert status == 404 and body["error"]["code"] == "unknown_job"


def test_requests_without_local_host_token_or_origin_are_refused(analysis_done):
    client, job = analysis_done["client"], analysis_done["job"]
    port = client.port
    for host in ("evil.example", f"evil.example:{port}", "127.0.0.1:1", f"0.0.0.0:{port}"):
        for path in ("/", "/api/session"):
            status, body = client.json("GET", path, host=host)
            assert status == 403 and body["error"]["code"] == "bad_host"

    status, body = client.json("GET", f"/api/jobs/{job['id']}/result", token=False)
    assert status == 401 and body["error"]["code"] == "bad_token"
    status, body = client.json("GET", "/api/session", token=False, headers={"X-PVD-Token": "x" * 43})
    assert status == 401 and body["error"]["code"] == "bad_token"
    status, _ = client.json("GET", f"/api/jobs/{job['id']}/files/predictions.csv", token=False)
    assert status == 401

    for headers in ({"Origin": "http://evil.example"}, {"Origin": f"http://127.0.0.1:{port + 1}"}, {"Origin": "null"}):
        status, body = client.json("POST", "/api/session/clear", origin=False, headers=headers)
        assert status == 403 and body["error"]["code"] == "bad_origin"
    status, body = client.json("POST", "/api/jobs/current/cancel", origin=False)
    assert status == 403 and body["error"]["code"] == "bad_origin"
    status, body = client.json("PUT", "/api/session")
    assert status == 405
    status, _ = client.json("GET", f"/api/jobs/{job['id']}/result")
    assert status == 200  # none of the refused requests cleared or replaced the job

    status, headers, _ = client.request("GET", "/api/session")
    assert status == 200
    assert headers["Cache-Control"] == "no-store" and headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert not any(k.lower().startswith("access-control-") for k in headers)


def test_static_allowlist_and_token_injection(serve, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "index.html").write_text('<meta name="pvd-token" content="__PVD_TOKEN__">')
    (assets / "app.css").write_text("body{}")
    (assets / "app.js").write_text("void 0;")
    client = serve(tmp_path / "no-model.json")
    client.srv.assets_dir = assets
    status, headers, body = client.request("GET", "/", token=False)
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert client.srv.token.encode() in body and b"__PVD_TOKEN__" not in body
    assert client.request("GET", "/assets/app.js", token=False)[0] == 200
    for path in ("/index.html", "/assets/../gui.py", "/assets/%2e%2e/inference.py", "/assets/x.js", "/inference.py"):
        assert client.request("GET", path, token=False)[0] == 404, path


@pytest.mark.parametrize("kind", ["missing_column", "binary"])
def test_malformed_csv_fails_with_sanitized_error_and_no_files(kind, serve, trained, syn_test, tmp_path):
    bad = tmp_path / "upload" / "bad.csv"
    bad.parent.mkdir()
    if kind == "missing_column":
        raw = pd.read_csv(syn_test / "energy.csv", sep=";", dtype=str, nrows=5000)
        raw.drop(columns=["timestamp_epoch"]).to_csv(bad, sep=";", index=False)
    else:
        bad.write_bytes(bytes(range(256)) * 64)
    client = serve(trained)
    status, _ = client.upload(bad)
    assert status == 202
    job = client.wait()
    assert job["state"] == "failed"
    err = job["error"]
    assert err["title"] and err["detail"] and err["action"]
    if kind == "missing_column":
        assert "timestamp_epoch" in err["detail"]
    _assert_sanitized(job, tmp_path, client.srv.session_dir, trained)
    for name in ("predictions.csv", "pv_extensions_predictions.csv"):
        status, _, _ = client.file(job["id"], name)
        assert status == 409
    assert client.json("GET", f"/api/jobs/{job['id']}/result")[0] == 409
    assert _session_files(client.srv) == []


def test_missing_model_refuses_analysis_and_synthetic_model_is_never_used(serve, syn_test, tmp_path):
    """The single end-to-end synthetic check: runs without a model, yet private analysis stays refused."""
    client = serve(tmp_path / "absent" / "model.json")
    status, session = client.json("GET", "/api/session")
    assert status == 200 and session["model"]["available"] is False and session["model"]["reason"]
    tiny = syn_test / "labels.csv"
    status, body = client.upload(tiny)
    assert status == 409 and body["error"]["code"] == "model_unavailable"
    assert client.json("GET", "/api/jobs/current")[1]["job"] is None

    status, body = client.json("POST", "/api/jobs/synthetic")
    assert status == 202 and body["job"]["kind"] == "synthetic"
    job = client.wait()
    assert job["state"] == "complete", job["error"]
    status, result = client.json("GET", f"/api/jobs/{job['id']}/result")
    assert status == 200 and result["summary"]["session"] == "demo"
    passed = {c["key"]: c["passed"] for c in result["checks"]}
    assert all(passed[k] for k in (*CONTRACT_CHECKS, "determinism", "cli_parity", "score"))
    syn = result["synthetic"]
    assert syn["test_customers"] == len(result["customers"]) and syn["train_customers"] > 0
    assert 0 <= syn["metrics"]["score"] <= 100
    assert all(c["truth"] is not None for c in result["customers"])
    _, _, data = client.file(job["id"], "predictions.csv")
    _assert_valid_predictions(data.decode("utf-8"), [c["customer_id"] for c in result["customers"]])
    pv = next(c for c in result["customers"] if c["pv_installed"] and c["truth"]["pv_installed"])
    _, diag = client.json("GET", f"/api/jobs/{job['id']}/customers/{pv['customer_id']}/diagnostics")
    assert diag["truth"] == pv["truth"] and diag["markers"]["known_date"] == pv["truth"]["installation_date"]

    status, body = client.upload(tiny)
    assert status == 409 and body["error"]["code"] == "model_unavailable"
    assert client.json("GET", "/api/session")[1]["model"]["available"] is False
    assert client.json("GET", f"/api/jobs/{job['id']}/result")[0] == 200  # the refused upload replaced nothing


def test_corrupt_model_is_unavailable_or_fails_sanitized(serve, trained, syn_test, tmp_path):
    truncated = tmp_path / "truncated" / "model.json"
    truncated.parent.mkdir()
    truncated.write_text(trained.read_text()[:200])
    client = serve(truncated)
    assert client.json("GET", "/api/session")[1]["model"]["available"] is False
    status, body = client.upload(syn_test / "labels.csv")
    assert status == 409 and body["error"]["code"] == "model_unavailable"

    broken = tmp_path / "broken" / "model.json"
    broken.parent.mkdir()
    model = json.loads(trained.read_text())
    model["classifier"]["catboost_file"] = "catboost_clf.cbm"
    broken.write_text(json.dumps(model))
    (broken.parent / "catboost_clf.cbm").write_bytes(b"not a catboost model" * 50)
    client = serve(broken)
    assert client.json("GET", "/api/session")[1]["model"]["available"] is True
    status, _ = client.upload(syn_test / "energy.csv")
    assert status == 202
    job = client.wait()
    assert job["state"] == "failed" and job["error"]["title"] and job["error"]["action"]
    _assert_sanitized(job, tmp_path, client.srv.session_dir)
    assert client.file(job["id"], "predictions.csv")[0] == 409
    assert _session_files(client.srv) == []


def test_start_cancel_clear_invalidate_jobs_and_remove_only_session_files(serve, trained, syn_test):
    energy = syn_test / "energy.csv"
    before = _digest([energy, *_model_files(trained)])
    client = serve(trained)

    status, body = client.upload(energy)
    assert status == 202
    first = body["job"]["id"]
    assert client.json("POST", "/api/jobs/synthetic")[1]["error"]["code"] == "job_active"
    assert client.upload(syn_test / "labels.csv")[1]["error"]["code"] == "job_active"
    status, body = client.json("POST", "/api/jobs/current/cancel")
    assert status == 200 and body["job"]["state"] == "cancelled" and body["job"]["id"] == first
    assert client.json("GET", f"/api/jobs/{first}/result")[0] == 409
    assert client.file(first, "predictions.csv")[0] == 409
    assert _session_files(client.srv) == []
    assert client.json("POST", "/api/jobs/current/cancel")[1]["error"]["code"] == "no_active_job"

    second = client.upload(energy)[1]["job"]["id"]
    assert client.wait()["state"] == "complete"
    assert client.json("GET", f"/api/jobs/{first}/result")[0] == 404
    assert client.file(second, "predictions.csv")[0] == 200

    third = client.upload(energy)[1]["job"]["id"]
    assert client.wait()["state"] == "complete"
    assert client.json("GET", f"/api/jobs/{second}/result")[0] == 404
    assert client.file(second, "predictions.csv")[0] == 404
    assert client.file(third, "predictions.csv")[0] == 200

    status, body = client.json("POST", "/api/session/clear")
    assert status == 200 and body == {"cleared": True}
    assert client.json("GET", "/api/jobs/current")[1]["job"] is None
    assert client.json("GET", f"/api/jobs/{third}/result")[0] == 404
    assert client.file(third, "predictions.csv")[0] == 404
    assert _session_files(client.srv) == []
    assert _digest([energy, *_model_files(trained)]) == before
    assert client.json("GET", "/api/session")[1]["model"]["available"] is True


def test_shutdown_stops_the_worker_and_deletes_the_session(trained, syn_test, tmp_path):
    srv, thread = _start(trained, tmp_path / "session")
    client = Client(srv)
    assert client.upload(syn_test / "energy.csv")[0] == 202
    session_dir = srv.session_dir
    _stop(srv, thread)
    assert not thread.is_alive()
    assert not session_dir.exists()
    assert os.listdir(tmp_path / "session") == []
