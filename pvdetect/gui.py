"""Optional local GUI service: loopback HTTP API, single-job lifecycle, isolated worker and bounded diagnostics.

The submission interface stays `python inference.py path/to/test.csv`; this module only wraps the same
`inference.run_inference` call. The server binds 127.0.0.1, checks Host/token/Origin on every request,
never logs bodies, IDs, values, paths or raw exceptions, and keeps all session data in one 0700 temporary
directory that clear and shutdown delete. Abnormal termination can leave that directory behind.
"""

from __future__ import annotations

import atexit
import csv
import datetime as dt
import hmac
import http.server
import json
import logging
import math
import multiprocessing
import os
import queue as queue_mod
import re
import secrets
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from multiprocessing import connection as mp_connection
from pathlib import Path
from urllib.parse import unquote

from pvdetect.io import REQUIRED
from pvdetect.model import CURRENT_VERSION, referenced_files
from pvdetect.output import EXT_HEADER, PRED_HEADER

REPO = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(__file__).resolve().parent / "gui_assets"
TOKEN_PLACEHOLDER = b"__PVD_TOKEN__"
OUTPUT_FILES = ("predictions.csv", "pv_extensions_predictions.csv")
MAX_UPLOAD = 4 * 1024**3
UPLOAD_CHUNK = 1 << 20
SMALL_BODY = 64 * 1024  # bodies of non-upload requests are skipped up to this size, then the connection closes
SYNTHETIC_TRAIN = (24, 101)  # (customers, dev seed < 1000), variant A
SYNTHETIC_TEST = (12, 102)
DAILY_SERIES_MAX_DAYS = 1500
PROFILE_WINDOW_DAYS = 90
PROFILE_GAP_DAYS = 14
PROFILE_MIN_DAYS = 14
PROFILE_MIN_SLOT_SHARE = 0.8
CLI_TIMEOUT_S = 900

_LOG = logging.getLogger(__name__)

STAGES = {
    "analysis": (
        ("receive", "Receiving input"),
        ("load", "Validating and loading data"),
        ("predict", "Building features and predicting"),
        ("write", "Writing outputs"),
        ("check", "Checking outputs"),
    ),
    "synthetic": (
        ("prepare", "Preparing synthetic data and model"),
        ("load", "Validating and loading data"),
        ("predict", "Building features and predicting"),
        ("write", "Writing outputs"),
        ("check", "Checking outputs, repeatability and CLI parity"),
    ),
}
STEP_STAGE = {"model": "load", "data": "load", "predict": "predict", "write": "write"}

REGIMES = {
    "event": ("Change detected", "A step in midday withdrawal was found and dated."),
    "present_at_start": (
        "Present from start",
        "PV appears to be present when the data begins, so the date is estimated from the first day.",
    ),
    "insufficient": ("Weak evidence", "No reliable step was found; the date follows the model's fallback rule."),
}

INPUT_FORMAT = {
    "required_columns": list(REQUIRED),
    "separators": "semicolon or comma",
    "decimal": "point or comma",
    "timestamps": "epoch seconds or milliseconds",
}

SECURITY_HEADERS = (
    ("Cache-Control", "no-store"),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Frame-Options", "DENY"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    (
        "Content-Security-Policy",
        (
            "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; "
            "connect-src 'self'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        ),
    ),
)
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/assets/app.css": ("app.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/assets/inter.woff2": ("inter.woff2", "font/woff2"),
}
JSON_TYPE = "application/json; charset=utf-8"

_RESTART_HINT = "Open the address printed in the terminal where the GUI was started."
HTTP_ERRORS = {
    "bad_request": (400, "Request not understood", "The request was malformed.", "Reload the page and try again."),
    "upload_incomplete": (
        400,
        "Upload incomplete",
        "The file was not received completely.",
        "Choose the file again and retry.",
    ),
    "bad_token": (
        401,
        "Session token missing or invalid",
        "The request did not carry this session's token.",
        _RESTART_HINT,
    ),
    "bad_host": (403, "Request refused", "The request did not use this app's local address.", _RESTART_HINT),
    "bad_origin": (403, "Request refused", "The request did not come from this app's page.", _RESTART_HINT),
    "not_found": (404, "Not found", "There is nothing at this address.", _RESTART_HINT),
    "unknown_job": (
        404,
        "Result no longer available",
        "This job was replaced, cancelled or cleared.",
        "Start a new synthetic check or analysis.",
    ),
    "unknown_customer": (
        404,
        "Customer not found",
        "This customer is not part of the current result.",
        "Choose a customer from the results table.",
    ),
    "unknown_file": (
        404,
        "File not available",
        "Only predictions.csv and pv_extensions_predictions.csv can be downloaded.",
        "Use the download buttons.",
    ),
    "method_not_allowed": (
        405,
        "Method not allowed",
        "This address does not accept that request method.",
        "Use the app page instead of calling the address directly.",
    ),
    "not_complete": (
        409,
        "Result not ready",
        "The job has not completed successfully.",
        "Wait for the job to complete, or start a new one.",
    ),
    "job_active": (
        409,
        "A job is already running",
        "Only one job can run at a time.",
        "Wait for the current job to finish or cancel it.",
    ),
    "model_unavailable": (
        409,
        "No usable model",
        "Local CSV analysis uses only the model configured at launch, and it is not available.",
        "Restart the GUI with --model pointing to a trained model.json, or run the synthetic check.",
    ),
    "no_active_job": (409, "Nothing to cancel", "No job is receiving or running.", "No action is needed."),
    "job_cancelled": (
        409,
        "Upload stopped",
        "The job was cancelled or cleared while the file was being received.",
        "Start the analysis again if needed.",
    ),
    "length_required": (
        411,
        "File size missing",
        "The upload did not state its size.",
        "Upload the file from the app page.",
    ),
    "too_large": (413, "File too large", "Uploads are limited to 4 GiB.", "Split the input into smaller files."),
    "uri_too_long": (414, "Address too long", "The request address was too long.", _RESTART_HINT),
    "headers_too_large": (431, "Request too large", "The request headers were too large.", _RESTART_HINT),
    "internal_error": (
        500,
        "Unexpected server error",
        "The request could not be completed.",
        "Try again; if it keeps failing, restart the GUI.",
    ),
    "unavailable": (503, "Shutting down", "The GUI is shutting down.", "Restart the GUI."),
}
_STATUS_ERRORS = {400: "bad_request", 408: "bad_request", 414: "uri_too_long", 431: "headers_too_large"}

WORKER_FAILURES = {
    "model_missing": (
        "Model not found",
        "The configured model file, or a file it references, is missing.",
        "Restart the GUI with --model pointing to a trained model.json, or run the synthetic check.",
    ),
    "model_invalid": (
        "Model could not be loaded",
        "The configured model file or a file it references is corrupt or was written by an incompatible version.",
        "Retrain the model with train.py and restart the GUI with --model pointing to it.",
    ),
    "unparsable_csv": (
        "The CSV could not be read",
        "The file is not a UTF-8 CSV that the loader can parse (semicolon or comma separators, decimal point or comma).",
        "Check that you selected the time-series CSV export saved as UTF-8 text, then try again.",
    ),
    "upload_incomplete": (
        "Upload incomplete",
        "The file was not received completely; nothing was analysed.",
        "Choose the file again and retry.",
    ),
}


class _ApiError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class _WorkerFailure(Exception):
    def __init__(self, payload: dict):
        super().__init__("worker failure")
        self.payload = payload


def _failure(code: str) -> dict:
    title, detail, action = WORKER_FAILURES[code]
    return {"title": title, "detail": detail, "action": action}


def _missing_columns_failure(names: list[str]) -> dict:
    plural = "s" if len(names) > 1 else ""
    return {
        "title": "Required columns missing",
        "detail": f"The CSV has no {', '.join(names)} column{plural}.",
        "action": "Export the input with the columns " + ", ".join(REQUIRED) + ", separated by semicolons or commas.",
    }


def _generic_failure(stage_label: str) -> dict:
    return {
        "title": "The job failed",
        "detail": f"An unexpected error occurred during \u201c{stage_label}\u201d. No output files were kept.",
        "action": "Check the input format and try again. The synthetic check shows whether the installation works.",
    }


def model_status(path: Path) -> tuple[bool, str | None]:
    """Cheap availability check without importing CatBoost: JSON, version and referenced sibling files."""
    unreadable = "The model file could not be read as model JSON."
    try:
        if not path.is_file():
            return False, "No model file was found at the configured location."
        model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, unreadable
    if not isinstance(model, dict):
        return False, unreadable
    if model.get("version") != CURRENT_VERSION:
        return False, "The model file was written by an incompatible version; retrain it with train.py."
    try:
        refs = referenced_files(model)
    except (AttributeError, TypeError):
        return False, unreadable
    if not all(isinstance(f, str) and (path.parent / f).is_file() for f in refs):
        return False, "A file referenced by the model is missing next to model.json."
    return True, None


def saved_evaluation(model_path: Path) -> dict | None:
    """Aggregate error analysis (`pvdetect.diagnose --out-dir`) saved next to the configured model, if any.

    Only the aggregate summary is read; the per-customer `errors.csv` is never served."""
    from pvdetect.diagnose import DIAGNOSIS_FILE, DIAGNOSIS_SCHEMA

    path = model_path.parent / DIAGNOSIS_FILE
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        stale = path.stat().st_mtime < model_path.stat().st_mtime - 1
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != DIAGNOSIS_SCHEMA:
        return None
    keep = ("source", "out_of_fold", "missing_predictions", "failure_modes", "counts", "precision", "recall", "f1",
            "D", "P", "score", "lost", "largest", "dates", "capacity", "by_size", "by_season", "by_year")
    return {**{k: doc.get(k) for k in keep}, "stale": stale}


def _display_name(raw: str | None) -> str | None:
    """Display-only file name: percent-decoded, no path parts or control characters, at most 120 characters."""
    if not raw:
        return None
    name = re.split(r"[/\\]", unquote(raw))[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip()[:120]
    return name or None


def _dumps(obj) -> bytes:
    return json.dumps(obj, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _empty_dir(path: Path) -> None:
    for child in list(path.iterdir()) if path.is_dir() else []:
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _exited(proc) -> bool:
    """True once the worker has exited, without reaping it (only one thread ever joins)."""
    return bool(mp_connection.wait([proc.sentinel], timeout=0))


def _signal_worker(proc, sig: int) -> None:
    """Signal the worker's process group (it includes the CLI-parity subprocess); fall back to the pid."""
    try:
        os.killpg(proc.pid, sig)
    except (AttributeError, OSError):
        try:
            os.kill(proc.pid, sig)
        except OSError:
            pass


# --------------------------------------------------------------------------------------------- job state


class _Job:
    def __init__(self, kind: str, input_name: str | None, jobs_dir: Path):
        self.id = secrets.token_hex(16)
        self.kind = kind
        self.state = "receiving" if kind == "analysis" else "running"
        self.stages = [{"key": k, "label": label, "status": "pending"} for k, label in STAGES[kind]]
        self.labels = dict(STAGES[kind])
        self.started = time.monotonic()
        self.ended: float | None = None
        self.input_name = input_name
        self.error: dict | None = None
        self.dir = jobs_dir / self.id
        self.proc = None
        self.queue = None
        self.result: dict | None = None
        self.diag_index: dict[str, int] = {}

    @property
    def active(self) -> bool:
        return self.state in ("receiving", "running")

    def active_stage(self) -> str:
        for s in self.stages:
            if s["status"] == "active":
                return s["key"]
        return self.stages[-1]["key"]

    def set_active(self, key: str) -> None:
        if key not in self.labels:
            return
        before = True
        for s in self.stages:
            if s["key"] == key:
                s["status"], before = "active", False
            elif before:
                s["status"] = "done"

    def finish(self, state: str, error: dict | None = None) -> None:
        self.state, self.error, self.ended = state, error, time.monotonic()
        for s in self.stages:
            if state == "complete":
                s["status"] = "done"
            elif s["status"] == "active":
                s["status"] = "failed"

    def elapsed(self) -> float:
        return (self.ended if self.ended is not None else time.monotonic()) - self.started

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "stages": [dict(s) for s in self.stages],
            "elapsed_s": round(self.elapsed(), 2),
            "input_name": self.input_name,
            "error": dict(self.error) if self.error else None,
        }


# --------------------------------------------------------------------------------------------- server


class _LoopbackHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, gui: GuiServer):
        self.gui = gui
        super().__init__(("127.0.0.1", port), _Handler)

    def server_bind(self) -> None:  # skip HTTPServer's reverse-DNS lookup of the bind address
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address) -> None:  # never print tracebacks (they may hold paths)
        _LOG.debug("request handler error")


class GuiServer:
    """One loopback HTTP service with at most one job; see `create_server`."""

    def __init__(
        self, model_path: Path, port: int = 8765, session_root: Path | None = None, labels_path: Path | None = None
    ):
        import inference

        self.model_path = Path(model_path).expanduser().absolute()
        self.model_source = "default" if self.model_path.resolve() == inference.MODEL_PATH.resolve() else "configured"
        self.labels_path = Path(labels_path).expanduser().absolute() if labels_path else None
        self.token = secrets.token_urlsafe(32)
        self.assets_dir = ASSETS_DIR
        if session_root is not None:
            Path(session_root).mkdir(parents=True, exist_ok=True)
        self.session_dir = Path(tempfile.mkdtemp(prefix="pvdetect-gui-", dir=session_root))
        self._lock = threading.RLock()  # job state
        self._lifecycle = threading.Lock()  # start / upload hand-off / cancel / clear / shutdown
        self._job: _Job | None = None
        self._closed = False
        self._serve_thread: threading.Thread | None = None
        self._ctx = multiprocessing.get_context("spawn")
        try:
            self._httpd = _LoopbackHTTPServer(port, self)
        except BaseException:
            shutil.rmtree(self.session_dir, ignore_errors=True)
            raise
        self.port = self._httpd.server_address[1]
        self.hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        self.origins = {f"http://{h}" for h in self.hosts}
        atexit.register(self.shutdown_and_cleanup)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def serve_forever(self) -> None:
        with self._lifecycle:
            if self._closed:
                return
            self._serve_thread = threading.current_thread()
        try:
            self._httpd.serve_forever(poll_interval=0.2)
        finally:
            self._serve_thread = None

    def shutdown_and_cleanup(self) -> None:
        """Stop serving, terminate any worker, delete the session directory. Idempotent."""
        with self._lifecycle:
            if self._closed:
                return
            self._closed = True
            serving = self._serve_thread
            if serving is not None and serving is not threading.current_thread():
                self._httpd.shutdown()
            with self._lock:
                job, self._job = self._job, None
                if job is not None and job.active:
                    job.finish("cancelled")
            if job is not None:
                self._stop(job)
            shutil.rmtree(self.session_dir, ignore_errors=True)
            self._httpd.server_close()
        atexit.unregister(self.shutdown_and_cleanup)

    # ---- session and job queries

    def session_json(self) -> dict:
        available, reason = model_status(self.model_path)
        with self._lock:
            job = self._job.to_json() if self._job else None
        return {
            "model": {
                "available": available,
                "label": self.model_path.name,
                "source": self.model_source,
                "reason": reason,
            },
            "evaluation": saved_evaluation(self.model_path) if available else None,
            "labels": {"configured": self.labels_path is not None, "label": self.labels_path.name}
            if self.labels_path
            else {"configured": False, "label": None},
            "input": dict(INPUT_FORMAT),
            "outputs": list(OUTPUT_FILES),
            "job": job,
        }

    def current_json(self) -> dict | None:
        with self._lock:
            return self._job.to_json() if self._job else None

    def _job_for(self, job_id: str) -> _Job:
        job = self._job
        if job is None or job.id != job_id:
            raise _ApiError("unknown_job")
        return job

    def result_json(self, job_id: str) -> dict:
        with self._lock:
            job = self._job_for(job_id)
            if job.state != "complete":
                raise _ApiError("not_complete")
            res = job.result
            summary = {**res["summary"], "elapsed_s": round(job.elapsed(), 2)}
            return {"job_id": job.id, "kind": job.kind, **res, "summary": summary}

    def diagnostics_bytes(self, job_id: str, customer_id: str) -> bytes:
        with self._lock:
            job = self._job_for(job_id)
            if job.state != "complete":
                raise _ApiError("not_complete")
            idx = job.diag_index.get(customer_id)
            if idx is None:
                raise _ApiError("unknown_customer")
            path = job.dir / "diag" / f"{idx}.json"
        try:
            return path.read_bytes()
        except OSError:
            raise _ApiError("unknown_job") from None

    def output_path(self, job_id: str, name: str) -> Path:
        with self._lock:
            job = self._job_for(job_id)
            if name not in OUTPUT_FILES:
                raise _ApiError("unknown_file")
            if job.state != "complete":
                raise _ApiError("not_complete")
            return job.dir / "out" / name

    # ---- lifecycle (callers never hold self._lock across process operations)

    def _ensure_open_and_idle(self) -> None:
        if self._closed:
            raise _ApiError("unavailable")
        with self._lock:
            if self._job is not None and self._job.active:
                raise _ApiError("job_active")

    def _replace_job(self, kind: str, input_name: str | None) -> _Job:
        """Delete the previous (inactive) job, invalidating its id, and register a new one."""
        with self._lock:
            old, self._job = self._job, None
        if old is not None:
            shutil.rmtree(old.dir, ignore_errors=True)
        job = _Job(kind, input_name, self.session_dir / "jobs")
        job.dir.mkdir(parents=True)
        with self._lock:
            self._job = job
        return job

    def analysis_precheck(self, headers) -> str | None:
        """Checks that need no body: size header, job slot and model availability."""
        raw = headers.get("Content-Length")
        if raw is None or "Transfer-Encoding" in headers:
            return "length_required"
        if not re.fullmatch(r"[0-9]{1,20}", raw.strip()):
            return "bad_request"
        if int(raw) > MAX_UPLOAD:
            return "too_large"
        if self._job is not None and self._job.active:
            return "job_active"
        if not model_status(self.model_path)[0]:
            return "model_unavailable"
        return None

    def start_synthetic(self) -> dict:
        with self._lifecycle:
            self._ensure_open_and_idle()
            job = self._replace_job("synthetic", None)
            spec = {"kind": "synthetic", "job_dir": str(job.dir), "synthetic_dir": str(self.session_dir / "synthetic")}
            self._spawn(job, spec, "prepare")
            with self._lock:
                return job.to_json()

    def begin_analysis(self, input_name: str | None) -> tuple[_Job, int]:
        """Register a receiving analysis job and open its 0600 upload file; the model is checked first."""
        with self._lifecycle:
            self._ensure_open_and_idle()
            if not model_status(self.model_path)[0]:
                raise _ApiError("model_unavailable")
            job = self._replace_job("analysis", input_name)
            with self._lock:
                job.set_active("receive")
            try:
                fd = os.open(job.dir / "input.csv", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except OSError:
                with self._lock:
                    job.finish("failed", _failure("upload_incomplete"))
                shutil.rmtree(job.dir, ignore_errors=True)
                raise
            return job, fd

    def upload_received(self, job: _Job) -> dict | None:
        with self._lifecycle:
            with self._lock:
                if self._job is not job or job.state != "receiving":
                    return None
                job.state = "running"
            spec = {
                "kind": "analysis",
                "job_dir": str(job.dir),
                "input": str(job.dir / "input.csv"),
                "model": str(self.model_path),
                "model_label": self.model_path.name,
                "labels": str(self.labels_path) if self.labels_path else None,
            }
            self._spawn(job, spec, "load")
            with self._lock:
                return job.to_json()

    def upload_failed(self, job: _Job) -> None:
        with self._lifecycle:
            with self._lock:
                if self._job is job and job.state == "receiving":
                    job.finish("failed", _failure("upload_incomplete"))
            shutil.rmtree(job.dir, ignore_errors=True)

    def cancel(self) -> dict:
        with self._lifecycle:
            with self._lock:
                job = self._job
                if job is None or not job.active:
                    raise _ApiError("no_active_job")
                job.finish("cancelled")
            self._stop(job)
            with self._lock:
                return job.to_json()

    def clear(self) -> None:
        """Cancel any job and delete every session file (uploads, outputs, diagnostics, synthetic assets)."""
        with self._lifecycle:
            with self._lock:
                job, self._job = self._job, None
                if job is not None and job.active:
                    job.finish("cancelled")
            if job is not None:
                self._stop(job)
            _empty_dir(self.session_dir)

    def _stop(self, job: _Job) -> None:
        proc = job.proc
        if proc is not None:
            if not _exited(proc):
                _signal_worker(proc, signal.SIGTERM)
                proc.join(10)
                if proc.exitcode is None:
                    _signal_worker(proc, signal.SIGKILL)
                    proc.join(5)
            else:
                proc.join(5)
        shutil.rmtree(job.dir, ignore_errors=True)

    def _spawn(self, job: _Job, spec: dict, first_stage: str) -> None:
        with self._lock:
            job.set_active(first_stage)
            try:
                job.queue = self._ctx.Queue()
                job.proc = self._ctx.Process(
                    target=_worker_main, args=(spec, job.queue), daemon=True, name="pvdetect-gui-worker"
                )
                job.proc.start()
            except Exception:
                job.proc = None
                job.finish("failed", _generic_failure(job.labels[first_stage]))
                shutil.rmtree(job.dir, ignore_errors=True)
                return
        threading.Thread(target=self._monitor, args=(job,), daemon=True, name="pvdetect-gui-monitor").start()

    # ---- worker monitoring

    def _is_current_running(self, job: _Job) -> bool:
        return self._job is job and job.state == "running"

    def _monitor(self, job: _Job) -> None:
        q, proc = job.queue, job.proc
        outcome = "continue"
        try:
            while outcome == "continue":
                try:
                    msg = q.get(timeout=0.2)
                except queue_mod.Empty:
                    with self._lock:
                        if not self._is_current_running(job):
                            return
                    if not _exited(proc):
                        continue
                    try:
                        msg = q.get(timeout=1.0)
                    except queue_mod.Empty:
                        msg = ("crashed",)
                except (EOFError, OSError, ValueError):
                    msg = ("crashed",)
                outcome = self._apply(job, msg)
            if outcome == "finished":
                proc.join(10)
        finally:
            q.close()

    def _apply(self, job: _Job, msg: tuple) -> str:
        kind = msg[0]
        if kind == "stage":
            with self._lock:
                if not self._is_current_running(job):
                    return "abandoned"
                job.set_active(msg[1])
                return "continue"
        if kind == "done":
            try:
                result = json.loads((job.dir / "result.json").read_text(encoding="utf-8"))
                index = {c["customer_id"]: i for i, c in enumerate(result["customers"])}
            except (OSError, ValueError, KeyError, TypeError):
                return self._fail(job, None, None)
            with self._lock:
                if not self._is_current_running(job):
                    return "abandoned"
                job.result, job.diag_index = result, index
                job.finish("complete")
            (job.dir / "input.csv").unlink(missing_ok=True)
            return "finished"
        if kind == "error":
            return self._fail(job, msg[1], msg[2])
        return self._fail(job, None, None)

    def _fail(self, job: _Job, stage: str | None, payload: dict | None) -> str:
        with self._lock:
            if not self._is_current_running(job):
                return "abandoned"
            if stage is not None:
                job.set_active(stage)
            job.finish("failed", payload or _generic_failure(job.labels[job.active_stage()]))
        shutil.rmtree(job.dir, ignore_errors=True)  # nothing partial stays downloadable
        return "finished"


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "pvdetect-gui"
    sys_version = ""
    timeout = 60
    _route = "-"
    _pending = 0  # declared request-body bytes not yet read
    _responded = False

    @property
    def gui(self) -> GuiServer:
        return self.server.gui

    # ---- logging: method, route template and status only

    def log_request(self, code="-", size="-") -> None:
        _LOG.debug("%s %s %s", self.command or "-", self._route, getattr(code, "value", code))

    def log_message(self, format, *args) -> None:  # stdlib signature; request lines and paths are never logged
        pass

    # ---- request parsing and checks

    def parse_request(self) -> bool:
        self._route, self._pending, self._responded = "-", 0, False
        if not super().parse_request():
            return False
        if self.command in ("GET", "POST"):
            return True
        self.close_connection = True
        self._send_api_error(self._check_request() or "method_not_allowed")
        return False

    def handle_expect_100(self) -> bool:
        err = self._check_request()
        if err is None and self.command == "POST" and self._raw_path() == "/api/jobs/analysis":
            err = self.gui.analysis_precheck(self.headers)
        if err is not None:
            self.close_connection = True  # the body was never sent; refuse it before it is
            self._send_api_error(err)
            return False
        return super().handle_expect_100()

    def _raw_path(self) -> str:
        return self.path.split("?", 1)[0].split("#", 1)[0]

    def _check_request(self) -> str | None:
        gui = self.gui
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1 or hosts[0].strip().lower() not in gui.hosts:
            return "bad_host"
        path = self._raw_path()
        if path == "/api" or path.startswith("/api/"):
            given = (self.headers.get("X-PVD-Token") or "").encode("utf-8", "surrogateescape")
            if not hmac.compare_digest(given, gui.token.encode("ascii")):
                return "bad_token"
        if self.command == "POST":
            if (self.headers.get("Origin") or "") not in gui.origins:
                return "bad_origin"
        elif self.command != "GET":
            return "method_not_allowed"
        return None

    def _declared_length(self) -> int:
        if "Transfer-Encoding" in self.headers:
            self.close_connection = True  # chunked bodies are not accepted; never parse them
            return 0
        raw = self.headers.get("Content-Length")
        if raw is None:
            return 0
        if not re.fullmatch(r"[0-9]{1,20}", raw.strip()):
            self.close_connection = True
            raise _ApiError("bad_request")
        return int(raw)

    # ---- responses

    def _start(self, status: int, ctype: str, length: int, extra: tuple = ()) -> None:
        self._responded = True
        self.send_response(status)
        for k, v in SECURITY_HEADERS:
            self.send_header(k, v)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        for k, v in extra:
            self.send_header(k, v)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()

    def _send_bytes(self, status: int, ctype: str, body: bytes, extra: tuple = ()) -> None:
        self._start(status, ctype, len(body), extra)
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, obj) -> None:
        self._send_bytes(status, JSON_TYPE, _dumps(obj))

    def _send_api_error(self, code: str) -> None:
        status, title, detail, action = HTTP_ERRORS[code]
        if self._pending:
            self.close_connection = True
        self._send_json(status, {"error": {"code": code, "title": title, "detail": detail, "action": action}})
        if self._pending:
            self._drain()

    def send_error(self, code, message=None, explain=None) -> None:
        """Malformed requests (bad request line, oversized headers, timeouts) get the same JSON error shape."""
        self.close_connection = True
        key = _STATUS_ERRORS.get(int(code), "bad_request")
        status, title, detail, action = HTTP_ERRORS[key]
        body = _dumps({"error": {"code": key, "title": title, "detail": detail, "action": action}})
        try:
            self._start(int(code) if int(code) >= 400 else status, JSON_TYPE, len(body))
            if self.command != "HEAD" and int(code) >= 200 and int(code) not in (204, 304):
                self.wfile.write(body)
        except OSError:
            pass

    def _drain(self) -> None:
        """Discard a bounded amount of an unread body so the error response is not lost to a TCP reset."""
        left = min(self._pending, SMALL_BODY)
        self._pending = 0
        try:
            self.connection.settimeout(1.0)
            while left > 0:
                chunk = self.rfile.read1(min(left, SMALL_BODY))
                if not chunk:
                    break
                left -= len(chunk)
        except OSError:
            pass

    def _skip_small_body(self) -> None:
        if self._pending > SMALL_BODY:
            self.close_connection = True
        elif self._pending:
            self.rfile.read(self._pending)
            self._pending = 0

    # ---- dispatch

    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def _dispatch(self) -> None:
        try:
            self._pending = self._declared_length()
            err = self._check_request()
            if err is not None:
                raise _ApiError(err)
            path = self._raw_path()
            if path.startswith("/api/"):
                self._api(path)
            else:
                self._static(path)
        except _ApiError as e:
            if self._responded:
                self.close_connection = True
            else:
                self._send_api_error(e.code)
        except OSError:  # client went away or timed out
            self.close_connection = True
        except Exception:
            self.close_connection = True
            if not self._responded:
                try:
                    self._send_api_error("internal_error")
                except OSError:
                    pass

    def _static(self, path: str) -> None:
        self._route = "/assets/{name}" if path.startswith("/assets/") else path if path == "/" else "/*"
        entry = STATIC_FILES.get(path)
        if entry is None:
            raise _ApiError("not_found")
        if self.command != "GET":
            raise _ApiError("method_not_allowed")
        self._skip_small_body()
        name, ctype = entry
        try:
            body = (self.gui.assets_dir / name).read_bytes()
        except OSError:
            raise _ApiError("not_found") from None
        if name == "index.html":
            body = body.replace(TOKEN_PLACEHOLDER, self.gui.token.encode("ascii"))
        self._send_bytes(200, ctype, body)

    def _need(self, method: str, route: str) -> None:
        self._route = route
        if self.command != method:
            raise _ApiError("method_not_allowed")
        if route != "/api/jobs/analysis":
            self._skip_small_body()

    def _api(self, path: str) -> None:
        gui = self.gui
        segs = [unquote(s) for s in path[len("/api/") :].split("/")]
        match segs:
            case ["session"]:
                self._need("GET", "/api/session")
                self._send_json(200, gui.session_json())
            case ["session", "clear"]:
                self._need("POST", "/api/session/clear")
                gui.clear()
                self._send_json(200, {"cleared": True})
            case ["jobs", "synthetic"]:
                self._need("POST", "/api/jobs/synthetic")
                self._send_json(202, {"job": gui.start_synthetic()})
            case ["jobs", "analysis"]:
                self._need("POST", "/api/jobs/analysis")
                self._post_analysis()
            case ["jobs", "current"]:
                self._need("GET", "/api/jobs/current")
                self._send_json(200, {"job": gui.current_json()})
            case ["jobs", "current", "cancel"]:
                self._need("POST", "/api/jobs/current/cancel")
                self._send_json(200, {"job": gui.cancel()})
            case ["jobs", job_id, "result"]:
                self._need("GET", "/api/jobs/{id}/result")
                self._send_json(200, gui.result_json(job_id))
            case ["jobs", job_id, "customers", customer_id, "diagnostics"]:
                self._need("GET", "/api/jobs/{id}/customers/{customer_id}/diagnostics")
                self._send_bytes(200, JSON_TYPE, gui.diagnostics_bytes(job_id, customer_id))
            case ["jobs", job_id, "files", name]:
                self._need("GET", "/api/jobs/{id}/files/{name}")
                self._send_output(job_id, name)
            case _:
                self._route = "/api/*"
                raise _ApiError("not_found")

    def _send_output(self, job_id: str, name: str) -> None:
        path = self.gui.output_path(job_id, name)
        try:
            f = open(path, "rb")
        except OSError:
            raise _ApiError("unknown_job") from None
        with f:
            size = os.fstat(f.fileno()).st_size
            extra = (("Content-Disposition", f'attachment; filename="{name}"'),)
            self._start(200, "text/csv; charset=utf-8", size, extra)
            shutil.copyfileobj(f, self.wfile, UPLOAD_CHUNK)

    def _post_analysis(self) -> None:
        gui = self.gui
        err = gui.analysis_precheck(self.headers)
        if err in ("length_required", "bad_request", "too_large"):
            raise _ApiError(err)
        job, fd = gui.begin_analysis(_display_name(self.headers.get("X-PVD-Filename")))
        try:
            outcome = self._receive(job, fd)
        except BaseException:
            gui.upload_failed(job)
            raise
        if outcome == "cancelled":
            raise _ApiError("job_cancelled")
        if outcome == "incomplete":
            gui.upload_failed(job)
            raise _ApiError("upload_incomplete")
        job_json = gui.upload_received(job)
        if job_json is None:
            raise _ApiError("job_cancelled")
        self._send_json(202, {"job": job_json})

    def _receive(self, job: _Job, fd: int) -> str:
        """Stream the body to the job's upload file in chunks; stop as soon as the job is cancelled."""
        try:
            with os.fdopen(fd, "wb") as f:
                while self._pending:
                    if job.state != "receiving":
                        return "cancelled"
                    chunk = self.rfile.read(min(UPLOAD_CHUNK, self._pending))
                    if not chunk:
                        self.close_connection = True
                        return "incomplete"
                    f.write(chunk)
                    self._pending -= len(chunk)
        except OSError:
            self.close_connection = True
            return "incomplete"
        return "received" if job.state == "receiving" else "cancelled"


def create_server(
    model_path: Path, port: int = 8765, session_root: Path | None = None, labels_path: Path | None = None
) -> GuiServer:
    """Bind 127.0.0.1:`port` (0 = ephemeral). Analysis uses only `model_path`; session data lives under a
    fresh 0700 directory (inside `session_root` when given, else the system temp directory). With
    `labels_path`, every completed analysis is diagnosed against those labels."""
    return GuiServer(model_path, port, session_root, labels_path)


# --------------------------------------------------------------------------------------------- worker


def _worker_main(spec: dict, q) -> None:
    """Spawned worker. Reports only ("stage", key), ("done",) or ("error", stage, sanitized payload)."""
    try:
        os.setpgrp()  # own process group, so cancelling also stops the CLI-parity subprocess
    except (AttributeError, OSError):
        pass
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)  # nothing a library prints (warnings, tracebacks) may reach the terminal
    os.dup2(devnull, 2)
    import warnings

    warnings.simplefilter("ignore")
    labels = dict(STAGES[spec["kind"]])
    current = {"stage": "prepare" if spec["kind"] == "synthetic" else "load"}  # already active on the server

    def stage(key: str) -> None:
        if current["stage"] != key:
            current["stage"] = key
            q.put(("stage", key))

    try:
        if spec["kind"] == "synthetic":
            _run_synthetic(spec, stage)
        else:
            _run_analysis(spec, stage)
    except _WorkerFailure as f:
        q.put(("error", current["stage"], f.payload))
    except BaseException:
        q.put(("error", current["stage"], _generic_failure(labels[current["stage"]])))
    else:
        q.put(("done",))
    finally:
        q.close()
        q.join_thread()


def _classify(exc: BaseException, step: str | None) -> dict | None:
    """Sanitized payload for known failure kinds; None means unexpected (generic message with stage)."""
    if step == "model":
        return _failure("model_missing" if isinstance(exc, FileNotFoundError) else "model_invalid")
    if step == "data":
        msg = str(exc)
        if isinstance(exc, ValueError) and msg.startswith("missing required columns"):
            names = [c for c in REQUIRED if repr(c) in msg]
            if names:
                return _missing_columns_failure(names)
        if isinstance(exc, (ValueError, KeyError, IndexError, TypeError, csv.Error)):
            return _failure("unparsable_csv")
    return None


def _infer(test_csv: Path, model_path: Path, out_dir: Path, stage):
    from inference import run_inference

    step = {"name": None}

    def on_step(name: str) -> None:
        step["name"] = name
        stage(STEP_STAGE[name])

    try:
        return run_inference(test_csv, model_path, out_dir, on_step=on_step)
    except Exception as e:
        payload = _classify(e, step["name"])
        if payload is None:
            raise
        raise _WorkerFailure(payload) from None


def _write_json(path: Path, obj) -> None:
    path.write_bytes(_dumps(obj))


def _floats(values, nd: int = 4) -> list:
    import numpy as np

    return [round(float(v), nd) if np.isfinite(v) else None for v in values]


def _iso(d) -> str | None:
    return None if d is None else str(d)[:10]


def _read_rows(path: Path) -> tuple[list[str] | None, list[list[str]]]:
    """(header, rows) of a written submission file; header None when it is not UTF-8 or empty."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, []
    lines = list(csv.reader(text.splitlines(), delimiter=";"))
    if not lines:
        return None, []
    return lines[0], lines[1:]


def _customer_rows(pred_csv: Path, run, truth: dict | None) -> list[dict]:
    """One entry per written predictions.csv row (file order), values exactly as written."""
    regimes = dict(zip(run.pred["customer_id"].astype(str), run.pred["regime"]))
    _, rows = _read_rows(pred_csv)
    out = []
    for r in rows:
        if not r:
            continue
        cid = r[0]
        date = r[2] if len(r) > 2 and r[2] else None
        try:
            kwp = float(r[3]) if len(r) > 3 and r[3] else None
        except ValueError:
            kwp = None
        cd = run.table.customers.get(cid)
        out.append(
            {
                "customer_id": cid,
                "pv_installed": len(r) > 1 and r[1] == "TRUE",
                "installation_date": date,
                "capacity_kwp": kwp,
                "regime": regimes.get(cid, "insufficient"),
                "usable": cd is not None and cd.n_valid > 0,
                "truth": truth.get(cid) if truth is not None else None,
            }
        )
    return out


def _output_checks(out: Path, ids: list[str]) -> list[dict]:
    """Submission-contract checks on the written files; details carry counts only, never IDs or values."""
    header, rows = _read_rows(out / "predictions.csv")
    ext_header, ext_rows = _read_rows(out / "pv_extensions_predictions.csv")
    file_ids = [r[0] for r in rows if r]
    leading = sum(1 for c in ids if len(c) > 1 and c.startswith("0"))

    bad_shape = sum(1 for r in rows if len(r) != len(PRED_HEADER))
    header_ok = header == PRED_HEADER and bad_shape == 0
    one_row = len(rows) == len(ids) and set(file_ids) == set(ids)
    unique = len(set(file_ids)) == len(file_ids) and sorted(file_ids) == sorted(ids)

    flags = [r[1] if len(r) > 1 else "" for r in rows]
    false_rows = [r for r, f in zip(rows, flags) if f == "FALSE"]
    true_rows = [r for r, f in zip(rows, flags) if f == "TRUE"]
    bad_flags = len(rows) - len(false_rows) - len(true_rows)
    false_ok = bad_flags == 0 and all(len(r) == 4 and r[2] == "" and r[3] == "" for r in false_rows)

    def true_ok(r: list[str]) -> bool:
        if len(r) != 4 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", r[2]):
            return False
        try:
            dt.date.fromisoformat(r[2])
            kwp = float(r[3])
        except ValueError:
            return False
        return math.isfinite(kwp) and kwp > 0

    bad_true = sum(1 for r in true_rows if not true_ok(r))
    ext_ok = ext_header == EXT_HEADER and all(len(r) == len(EXT_HEADER) for r in ext_rows)

    def check(key: str, label: str, passed: bool, detail: str) -> dict:
        return {"key": key, "label": label, "passed": bool(passed), "detail": detail}

    return [
        check(
            "header",
            "Header and semicolons",
            header_ok,
            "Exact header; every row has 4 semicolon-separated fields."
            if header_ok
            else f"Header {'matches' if header == PRED_HEADER else 'does not match'}; {bad_shape} rows have the wrong number of fields.",
        ),
        check(
            "one_row_per_customer",
            "One row per input customer",
            one_row,
            f"{len(rows)} rows for {len(ids)} input customers.",
        ),
        check(
            "unique_ids",
            "Unique string customer IDs",
            unique,
            f"{len(set(file_ids))} distinct IDs, identical to the input IDs as text"
            f" ({leading} with leading zeros preserved)."
            if unique
            else f"{len(file_ids) - len(set(file_ids))} duplicate IDs; IDs {'match' if set(file_ids) == set(ids) else 'differ from'} the input.",
        ),
        check(
            "false_blanks",
            "FALSE rows leave date and capacity empty",
            false_ok,
            f"{len(false_rows)} FALSE rows, all blank."
            if false_ok
            else f"{len(false_rows)} FALSE rows; {bad_flags} rows have neither TRUE nor FALSE or carry values.",
        ),
        check(
            "true_fields",
            "TRUE rows have a date and positive capacity",
            bad_true == 0,
            f"{len(true_rows)} TRUE rows, all with a valid YYYY-MM-DD date and capacity above 0."
            if bad_true == 0
            else f"{bad_true} of {len(true_rows)} TRUE rows lack a valid date or positive capacity.",
        ),
        check(
            "extensions_header",
            "Extension file header",
            ext_ok,
            f"Exact header; {len(ext_rows)} extension rows."
            if ext_ok
            else "The extension file header or rows do not match.",
        ),
    ]


def _summary(run, customers: list[dict], inference_s: float, model_label: str, session: str) -> dict:
    rep = run.report
    cds = [run.table.customers.get(c) for c in run.customer_ids]
    present = [cd for cd in cds if cd is not None]
    pv = sum(1 for c in customers if c["pv_installed"])
    return {
        "customers": len(run.customer_ids),
        "pv": pv,
        "non_pv": len(customers) - pv,
        "date_range": {
            "start": _iso(min(cd.dates[0] for cd in present)) if present else None,
            "end": _iso(max(cd.dates[-1] for cd in present)) if present else None,
        },
        "quality": {
            "rows_total": int(rep["rows_total"]),
            "rows_bad_epoch": int(rep["rows_bad_epoch"]),
            "rows_bad_energy": int(rep["rows_bad_energy"]),
            "rows_duplicate": int(rep["rows_duplicate"]),
            "customers_signed": int(rep["customers_signed"]),
            "day_length_counts": {k: int(v) for k, v in rep["day_length_counts"].items()},
            "customers_without_usable_history": sum(1 for cd in cds if cd is None or cd.n_valid == 0),
        },
        "elapsed_s": 0.0,  # the server fills in the job's wall time
        "inference_s": round(inference_s, 2),
        "model_label": model_label,
        "session": session,
    }


def _outputs(checks: list[dict], out: Path) -> dict:
    by = {c["key"]: c["passed"] for c in checks}
    _, rows = _read_rows(out / "predictions.csv")
    _, ext_rows = _read_rows(out / "pv_extensions_predictions.csv")
    main_keys = ("header", "one_row_per_customer", "unique_ids", "false_blanks", "true_fields")
    if ext_rows:
        note = f"{len(ext_rows)} extension rows were written."
    else:
        note = (
            "Extension detection is not enabled in the current model, so this file contains only the header. "
            "An empty file is not evidence that no extensions exist."
        )
    return {
        "predictions": {"valid": all(by[k] for k in main_keys), "rows": len(rows)},
        "extensions": {"valid": by["extensions_header"], "rows": len(ext_rows), "note": note},
    }


def _week_starts(day_numbers):
    return day_numbers - (day_numbers + 3) % 7  # 1970-01-01 was a Thursday; Monday-start weeks


def _weekly(dates, total, mid_kw) -> dict:
    import numpy as np
    import pandas as pd

    days = dates.astype("int64")
    g = pd.DataFrame({"w": _week_starts(days), "t": total, "m": mid_kw}).groupby("w", sort=True).mean()
    weeks = np.asarray(g.index.to_numpy(), dtype="int64").astype("datetime64[D]")
    return {"dates": [str(d) for d in weeks], "total_kwh": _floats(g["t"]), "midday_kw": _floats(g["m"])}


def _no_profile(reason: str) -> dict:
    return {"available": False, "reason": reason, "window_days": PROFILE_WINDOW_DAYS, "before": None, "after": None}


def _profile(cd, predicted: str | None) -> dict:
    import numpy as np

    if predicted is None:
        return _no_profile("No installation date is predicted for this customer.")
    d = np.datetime64(predicted, "D")
    good = (~np.isnan(cd.M)).sum(axis=1) >= PROFILE_MIN_SLOT_SHARE * cd.M.shape[1]

    def side(lo, hi):
        in_window = (cd.dates >= lo) & (cd.dates <= hi)
        use = in_window & good
        if int(use.sum()) < PROFILE_MIN_DAYS:
            return None
        span = cd.dates[in_window]
        kw = np.nanmean(cd.M[use], axis=0) * 4
        return {"start": str(span[0]), "end": str(span[-1]), "days": int(use.sum()), "kw": _floats(kw)}

    w = np.timedelta64(PROFILE_WINDOW_DAYS, "D")
    gap = np.timedelta64(PROFILE_GAP_DAYS, "D")
    one = np.timedelta64(1, "D")
    before = side(d - w, d - one)
    after = side(d + gap, d + gap + w - one)
    if before is None or after is None:
        missing = [
            name
            for name, part in (
                (f"the {PROFILE_WINDOW_DAYS} days before the estimated date", before),
                (f"the {PROFILE_WINDOW_DAYS} days starting {PROFILE_GAP_DAYS} days after it", after),
            )
            if part is None
        ]
        return _no_profile(
            f"Fewer than {PROFILE_MIN_DAYS} days with at least {int(PROFILE_MIN_SLOT_SHARE * 100)} % of readings in "
            + " and in ".join(missing)
            + "."
        )
    return {
        "available": True,
        "reason": None,
        "window_days": PROFILE_WINDOW_DAYS,
        "before": before,
        "after": after,
    }


def _diagnostics(row: dict, cd, truth: dict | None) -> dict:
    """Bounded per-customer view from the run's CustomerData (same day matrix and daily_signal)."""
    import numpy as np

    key = row["regime"] if row["regime"] in REGIMES else "insufficient"
    label, explanation = REGIMES[key]
    empty = {"dates": [], "total_kwh": [], "midday_kw": []}
    base = {
        "customer_id": row["customer_id"],
        "prediction": row,
        "regime": {"key": key, "label": label, "explanation": explanation},
        "markers": {
            "predicted_date": row["installation_date"],
            "known_date": truth["installation_date"] if truth else None,
        },
        "truth": truth,
    }
    if cd is None:
        return {
            **base,
            "usable": False,
            "message": "The input has no readable rows for this customer, so there is nothing to show.",
            "coverage": {
                "first_date": None,
                "last_date": None,
                "span_days": 0,
                "days_with_data": 0,
                "valid_days": 0,
                "missing_slot_share": 1.0,
            },
            "series": {"resolution": "daily", **empty},
            "weekly": dict(empty),
            "profile": _no_profile("This customer has no usable history."),
        }
    has = ~np.isnan(cd.M)
    coverage = {
        "first_date": str(cd.dates[0]),
        "last_date": str(cd.dates[-1]),
        "span_days": len(cd.dates),
        "days_with_data": int((has.sum(axis=1) > 0).sum()),
        "valid_days": int(cd.n_valid),
        "missing_slot_share": round(float(1.0 - has.mean()), 4),
    }
    if cd.n_valid == 0:
        return {
            **base,
            "usable": False,
            "message": "No day has enough midday and night readings (16 of 20 slots each) to analyse this customer.",
            "coverage": coverage,
            "series": {"resolution": "daily", **empty},
            "weekly": dict(empty),
            "profile": _no_profile("This customer has no usable history."),
        }
    total = cd.daily["total"].to_numpy(float)
    mid_kw = cd.daily["mid"].to_numpy(float) * 4
    weekly = _weekly(cd.dates, total, mid_kw)
    if len(cd.dates) <= DAILY_SERIES_MAX_DAYS:
        series = {
            "resolution": "daily",
            "dates": [str(d) for d in cd.dates],
            "total_kwh": _floats(total),
            "midday_kw": _floats(mid_kw),
        }
    else:
        series = {"resolution": "weekly", **weekly}
    return {
        **base,
        "usable": True,
        "message": None,
        "coverage": coverage,
        "series": series,
        "weekly": weekly,
        "profile": _profile(cd, row["installation_date"] if row["pv_installed"] else None),
    }


def _write_diagnostics(diag_dir: Path, run, customers: list[dict], truth: dict | None) -> None:
    diag_dir.mkdir()
    for i, row in enumerate(customers):
        t = truth.get(row["customer_id"]) if truth is not None else None
        _write_json(diag_dir / f"{i}.json", _diagnostics(row, run.table.customers.get(row["customer_id"]), t))


def _run_analysis(spec: dict, stage) -> None:
    job_dir = Path(spec["job_dir"])
    out = job_dir / "out"
    out.mkdir()
    t0 = time.perf_counter()
    run = _infer(Path(spec["input"]), Path(spec["model"]), out, stage)
    inference_s = time.perf_counter() - t0
    customers = _customer_rows(out / "predictions.csv", run, None)
    _write_diagnostics(job_dir / "diag", run, customers, None)
    summary = _summary(run, customers, inference_s, spec["model_label"], "private")
    ids = list(run.customer_ids)
    del run  # release the day matrices before checking
    stage("check")
    checks = _output_checks(out, ids)
    result = {
        "summary": summary,
        "outputs": _outputs(checks, out),
        "checks": checks,
        "synthetic": None,
        "diagnosis": _diagnosis(out / "predictions.csv", Path(spec["labels"]), ids) if spec.get("labels") else None,
        "customers": customers,
    }
    _write_json(job_dir / "result.json", result)


def _diagnosis(pred_csv: Path, labels_csv: Path, ids: list[str]) -> dict:
    """Aggregate error analysis of this job's predictions against the labels of the customers it contains.

    Customers of the CSV without a label are left out; the result is in-sample when the model was trained
    on these customers (the frontend says so). Never raises: failures become a sanitized reason."""
    from pvdetect.diagnose import _json_safe, error_table, failure_modes, load_predictions, summarize
    from pvdetect.io import load_labels

    try:
        truth = load_labels(labels_csv)
    except (OSError, ValueError, KeyError):
        return {"available": False, "reason": "The labels file could not be read as a labels CSV."}
    truth = truth[truth["customer_id"].isin(set(ids))]
    if truth.empty:
        return {"available": False, "reason": "No customer in this CSV appears in the labels file."}
    pred, _ = load_predictions(pred_csv, None)
    s = summarize(error_table(pred, truth))
    doc = {"available": True, "matched": len(truth), "unlabelled": len(ids) - len(truth), "failure_modes": failure_modes(s)}
    return _json_safe({**doc, **s})


def _build_synthetic(build: Path, final: Path) -> None:
    """Variant-A train/test sets with dev seeds and a model fitted like `train.py` without --cv."""
    from pvdetect.io import load_labels, load_timeseries
    from pvdetect.model import save_model
    from pvdetect.pipeline import build_feature_table
    from pvdetect.procedure import fit
    from pvdetect.synthetic import generate_dataset, write_dataset

    shutil.rmtree(build, ignore_errors=True)
    write_dataset(build / "train", *generate_dataset(SYNTHETIC_TRAIN[0], seed=SYNTHETIC_TRAIN[1]), sep=";")
    write_dataset(build / "test", *generate_dataset(SYNTHETIC_TEST[0], seed=SYNTHETIC_TEST[1]), sep=";")
    labels = load_labels(build / "train" / "labels.csv")
    table = build_feature_table(load_timeseries(build / "train" / "energy.csv"))
    model = fit(table, labels, labels["customer_id"].tolist())
    save_model(model, build / "model" / "model.json")
    (build / "ready").write_text("")
    shutil.rmtree(final, ignore_errors=True)
    os.rename(build, final)


def _truth(labels_csv: Path) -> dict:
    import pandas as pd

    from pvdetect.io import load_labels

    lab = load_labels(labels_csv)
    out = {}
    for r in lab.itertuples(index=False):
        on = bool(r.pv_installed)
        has_date = on and pd.notna(r.pv_installation_date)
        has_kwp = on and pd.notna(r.pv_capacity_kwp)
        out[str(r.customer_id)] = {
            "pv_installed": on,
            "installation_date": r.pv_installation_date.strftime("%Y-%m-%d") if has_date else None,
            "capacity_kwp": float(r.pv_capacity_kwp) if has_kwp else None,
        }
    return out


def _same_outputs(a: Path, b: Path) -> bool:
    return all((a / n).read_bytes() == (b / n).read_bytes() for n in OUTPUT_FILES)


def _determinism_check(test_csv: Path, model: Path, out: Path, rerun: Path) -> dict:
    from inference import run_inference

    rerun.mkdir()
    try:
        run_inference(test_csv, model, rerun)
        same = _same_outputs(out, rerun)
        detail = (
            "A second inference produced byte-identical files."
            if same
            else "A second inference produced different files."
        )
    except Exception:
        same, detail = False, "The repeat inference failed."
    return {"key": "determinism", "label": "Repeat run is byte-identical", "passed": same, "detail": detail}


def _cli_parity_check(job_dir: Path, test_csv: Path, model_json: Path, out: Path) -> dict:
    """Run `python inference.py test.csv` from an isolated app copy with the model in artifacts/."""
    app, run_dir = job_dir / "app", job_dir / "run"
    label = "Matches command-line inference.py output"
    try:
        app.mkdir()
        shutil.copy2(REPO / "inference.py", app / "inference.py")
        shutil.copytree(REPO / "pvdetect", app / "pvdetect", ignore=shutil.ignore_patterns("__pycache__"))
        artifacts = app / "artifacts"
        artifacts.mkdir()
        shutil.copy2(model_json, artifacts / "model.json")
        for name in referenced_files(json.loads(model_json.read_text(encoding="utf-8"))):
            shutil.copy2(model_json.parent / name, artifacts / name)
        run_dir.mkdir()
        proc = subprocess.run(
            [sys.executable, str(app / "inference.py"), str(test_csv)],
            cwd=run_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=CLI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"key": "cli_parity", "label": label, "passed": False, "detail": "The CLI run could not be started."}
    if proc.returncode != 0:
        return {"key": "cli_parity", "label": label, "passed": False, "detail": "The CLI run exited with an error."}
    try:
        same = _same_outputs(out, run_dir)
    except OSError:
        return {"key": "cli_parity", "label": label, "passed": False, "detail": "The CLI run wrote no output files."}
    detail = (
        "Both CSVs written by the CLI in a separate app copy are byte-identical to the GUI's."
        if same
        else "The CLI wrote different bytes."
    )
    return {"key": "cli_parity", "label": label, "passed": same, "detail": detail}


def _run_synthetic(spec: dict, stage) -> None:
    from pvdetect.io import load_labels
    from pvdetect.scoring import main_score

    stage("prepare")
    job_dir, syn = Path(spec["job_dir"]), Path(spec["synthetic_dir"])
    if not (syn / "ready").is_file():
        _build_synthetic(job_dir / "synthetic-build", syn)
    test_csv, labels_csv, model = syn / "test" / "energy.csv", syn / "test" / "labels.csv", syn / "model" / "model.json"
    out = job_dir / "out"
    out.mkdir()
    t0 = time.perf_counter()
    run = _infer(test_csv, model, out, stage)
    inference_s = time.perf_counter() - t0
    truth = _truth(labels_csv)
    customers = _customer_rows(out / "predictions.csv", run, truth)
    _write_diagnostics(job_dir / "diag", run, customers, truth)
    summary = _summary(run, customers, inference_s, "Synthetic demonstration model", "demo")
    ids = list(run.customer_ids)
    del run
    stage("check")
    checks = _output_checks(out, ids)
    checks.append(_determinism_check(test_csv, model, out, job_dir / "rerun"))
    checks.append(_cli_parity_check(job_dir, test_csv, model, out))
    m = main_score(load_labels(out / "predictions.csv"), load_labels(labels_csv))
    metrics = {k: float(m[k]) for k in ("score", "f1", "D", "P")} | {k: int(m[k]) for k in ("tp", "fp", "fn")}
    checks.append(
        {
            "key": "score",
            "label": "Synthetic score (regression demonstration only)",
            "passed": True,
            "detail": (
                f"Synthetic score {metrics['score']:.1f} (F1 {metrics['f1']:.3f}, D {metrics['D']:.3f}, "
                f"P {metrics['P']:.3f}) on {len(ids)} synthetic test customers. Synthetic regression "
                "demonstration \u2014 not evidence of real-world accuracy."
            ),
        }
    )
    for d in ("rerun", "app", "run"):
        shutil.rmtree(job_dir / d, ignore_errors=True)
    result = {
        "summary": summary,
        "outputs": _outputs(checks, out),
        "checks": checks,
        "synthetic": {
            "metrics": metrics,
            "train_customers": SYNTHETIC_TRAIN[0],
            "test_customers": SYNTHETIC_TEST[0],
            "seeds": {"train": SYNTHETIC_TRAIN[1], "test": SYNTHETIC_TEST[1]},
        },
        "diagnosis": _diagnosis(out / "predictions.csv", labels_csv, ids),
        "customers": customers,
    }
    _write_json(job_dir / "result.json", result)
