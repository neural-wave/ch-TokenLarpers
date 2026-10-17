/* PV Detect local GUI. Vanilla JS, no external resources.
   Untrusted text (customer IDs, file names, server messages) is only ever written through
   textContent / attribute setters, never parsed as markup. */
(() => {
  "use strict";

  const TOKEN = document.querySelector('meta[name="pvd-token"]').getAttribute("content");
  const SVG_NS = "http://www.w3.org/2000/svg";
  const PAGE_SIZE = 20;
  const POLL_MS = 600;
  const REGIMES = {
    event: "Change detected",
    present_at_start: "Present from start",
    insufficient: "Weak evidence",
  };
  const ANALYSIS_STAGES = [
    ["receive", "Receiving input"],
    ["load", "Validating and loading data"],
    ["predict", "Building features and predicting"],
    ["write", "Writing outputs"],
    ["check", "Checking outputs"],
  ];

  const state = {
    session: null,
    job: null,
    jobSyncedAt: 0,
    result: null,
    resultFor: null,
    file: null,
    upload: null,
    xhr: null,
    filter: "all",
    query: "",
    page: 0,
    selected: null,
    diag: null,
    diagCache: new Map(),
    metric: "total",
    resolution: "weekly",
    busy: false,
    fatal: null,
  };
  let pollTimer = null;
  let tickTimer = null;

  // ---------------------------------------------------------------- DOM helpers

  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    applyAttrs(el, attrs);
    appendAll(el, children);
    return el;
  }

  function s(tag, attrs, ...children) {
    const el = document.createElementNS(SVG_NS, tag);
    applyAttrs(el, attrs);
    appendAll(el, children);
    return el;
  }

  function applyAttrs(el, attrs) {
    if (!attrs) return;
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "text") el.textContent = String(v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else el.setAttribute(k === "className" ? "class" : k, v === true ? "" : String(v));
    }
  }

  function appendAll(el, children) {
    for (const c of children.flat(Infinity)) {
      if (c === null || c === undefined || c === false) continue;
      el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
    }
  }

  const $ = (id) => document.getElementById(id);

  function icon(name, size = 14) {
    const paths = {
      check: ["M5 12l5 5L20 7"],
      x: ["M6 6l12 12", "M18 6L6 18"],
      alert: ["M12 3l10 18H2z", "M12 10v4", "M12 17.5v.5"],
      info: ["M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z", "M12 11v6", "M12 7.5v.5"],
      file: ["M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z", "M14 3v5h5"],
      download: ["M12 4v11", "M7 10l5 5 5-5", "M5 20h14"],
      search: ["M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14z", "M20 20l-3.5-3.5"],
      flask: ["M9 3h6", "M10 3v6L4.5 18.5A2 2 0 0 0 6.2 21h11.6a2 2 0 0 0 1.7-2.5L14 9V3", "M7 15h10"],
      lock: ["M6 11h12v10H6z", "M8.5 11V7.5a3.5 3.5 0 0 1 7 0V11"],
      left: ["M15 18l-6-6 6-6"],
      right: ["M9 18l6-6-6-6"],
      sun: ["M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z", "M12 2v2", "M12 20v2", "M4.9 4.9l1.4 1.4", "M17.7 17.7l1.4 1.4", "M2 12h2", "M20 12h2", "M4.9 19.1l1.4-1.4", "M17.7 6.3l1.4-1.4"],
      stop: ["M7 7h10v10H7z"],
    };
    return s(
      "svg",
      { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", focusable: "false" },
      (paths[name] || []).map((d) => s("path", { d })),
    );
  }

  // ---------------------------------------------------------------- formatting

  const nf = new Intl.NumberFormat("en-US");
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

  const fmtInt = (n) => (n === null || n === undefined ? "—" : nf.format(n));
  const fmtKwp = (v) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(2)} kWp`);
  const fmtPct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;

  function fmtElapsed(sec) {
    if (!Number.isFinite(sec)) return "—";
    if (sec < 60) return `${sec.toFixed(1)} s`;
    const m = Math.floor(sec / 60);
    const r = Math.floor(sec - m * 60);
    return `${m}:${String(r).padStart(2, "0")} min`;
  }

  function fmtBytes(b) {
    if (b < 1024) return `${b} B`;
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
    if (b < 1024 ** 3) return `${(b / 1024 / 1024).toFixed(1)} MB`;
    return `${(b / 1024 ** 3).toFixed(2)} GB`;
  }

  function dateMs(iso) {
    const [y, m, d] = iso.split("-").map(Number);
    return Date.UTC(y, m - 1, d);
  }

  function fmtDateLong(ms) {
    const d = new Date(ms);
    return `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
  }

  function fmtDate(iso) {
    if (!iso) return "—";
    const d = new Date(dateMs(iso));
    return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
  }

  function fmtHour(hh) {
    const mins = Math.round(hh * 60);
    return `${String(Math.floor(mins / 60) % 24).padStart(2, "0")}:${String(mins % 60).padStart(2, "0")}`;
  }

  function fmtVal(v, unit) {
    if (v === null || v === undefined || !Number.isFinite(v)) return "no data";
    const a = Math.abs(v);
    const d = a >= 100 ? 0 : a >= 10 ? 1 : 2;
    return `${v.toFixed(d)} ${unit}`;
  }

  // ---------------------------------------------------------------- API

  class ApiError extends Error {
    constructor(status, err) {
      super((err && err.title) || `Request failed (${status})`);
      this.status = status;
      this.err = err || { title: "Request failed", detail: `The local service answered with status ${status}.`, action: "Retry, or restart the app." };
    }
  }

  async function api(path, opts = {}) {
    let res;
    try {
      res = await fetch(path, {
        method: opts.method || "GET",
        headers: { "X-PVD-Token": TOKEN, ...(opts.headers || {}) },
        body: opts.body,
        cache: "no-store",
        credentials: "same-origin",
        redirect: "error",
      });
    } catch (_) {
      throw new ApiError(0, {
        title: "Local service unreachable",
        detail: "The PV Detect process is not answering on this address.",
        action: "Check the terminal where you started gui.py, restart it, then reload this page.",
      });
    }
    if (opts.raw) {
      if (!res.ok) throw new ApiError(res.status, await readError(res));
      return res;
    }
    const body = await res.json().catch(() => null);
    if (!res.ok) throw new ApiError(res.status, body && body.error);
    return body;
  }

  async function readError(res) {
    const body = await res.json().catch(() => null);
    return body && body.error;
  }

  function handleApiError(e, context) {
    if (!(e instanceof ApiError)) throw e;
    if (e.status === 401 || e.status === 403) {
      state.fatal = {
        title: "This page belongs to an earlier session",
        detail: "The local service was restarted or the page was opened from another origin.",
        action: "Reload the page from the address printed in the terminal.",
      };
      render();
      return;
    }
    toast(`${context}: ${e.err.title}`);
  }

  // ---------------------------------------------------------------- session & jobs

  async function refreshSession() {
    try {
      state.session = await api("/api/session");
      renderRail();
      applyJob(state.session.job);
    } catch (e) {
      handleApiError(e, "Could not read the session");
      renderRail();
    }
  }

  const isActive = (job) => !!job && (job.state === "running" || job.state === "receiving");

  function applyJob(job) {
    const prev = state.job;
    state.job = job || null;
    state.jobSyncedAt = performance.now();
    if (!job || (state.resultFor && state.resultFor !== job.id)) resetResults();
    if (isActive(job)) {
      schedulePoll();
      startTicker();
    } else {
      stopTicker();
    }
    if (job && job.state === "complete" && state.resultFor !== job.id) {
      loadResult(job.id);
      return;
    }
    const changed = !prev || !job || prev.id !== job.id || prev.state !== job.state || stageSig(prev) !== stageSig(job);
    if (changed) render();
    else updateElapsed();
  }

  const stageSig = (job) => (job && job.stages ? job.stages.map((st) => st.status).join(",") : "");

  function schedulePoll() {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      if (state.upload) return;
      try {
        const { job } = await api("/api/jobs/current");
        applyJob(job);
      } catch (e) {
        handleApiError(e, "Status update failed");
        schedulePoll();
      }
    }, POLL_MS);
  }

  function startTicker() {
    if (tickTimer) return;
    tickTimer = setInterval(updateElapsed, 100);
  }

  function stopTicker() {
    clearInterval(tickTimer);
    tickTimer = null;
  }

  function currentElapsed() {
    const job = state.job;
    if (!job) return 0;
    const base = job.elapsed_s || 0;
    return isActive(job) ? base + (performance.now() - state.jobSyncedAt) / 1000 : base;
  }

  function updateElapsed() {
    const el = $("elapsed-value");
    if (el) el.textContent = fmtElapsed(currentElapsed());
  }

  function resetResults() {
    state.result = null;
    state.resultFor = null;
    state.selected = null;
    state.diag = null;
    state.diagCache = new Map();
    state.page = 0;
    state.query = "";
    state.filter = "all";
  }

  async function loadResult(jobId) {
    try {
      const result = await api(`/api/jobs/${encodeURIComponent(jobId)}/result`);
      if (!state.job || state.job.id !== jobId) return;
      state.result = result;
      state.resultFor = jobId;
      const firstPv = result.customers.find((c) => c.pv_installed) || result.customers[0];
      render();
      if (firstPv) selectCustomer(firstPv.customer_id, false);
      refreshSessionQuietly();
    } catch (e) {
      handleApiError(e, "Could not load results");
    }
  }

  async function refreshSessionQuietly() {
    try {
      state.session = await api("/api/session");
      renderRail();
    } catch (_) {
      /* the next explicit action reports errors */
    }
  }

  async function startSynthetic() {
    if (state.busy || isActive(state.job)) return;
    state.busy = true;
    renderRail();
    try {
      resetResults();
      const { job } = await api("/api/jobs/synthetic", { method: "POST" });
      applyJob(job);
    } catch (e) {
      handleApiError(e, "Could not start the synthetic check");
    } finally {
      state.busy = false;
      renderRail();
    }
  }

  function startAnalysis() {
    const file = state.file;
    if (!file || state.busy || isActive(state.job)) return;
    resetResults();
    state.busy = true;
    state.upload = { loaded: 0, total: file.size };
    applyJob({
      id: null,
      kind: "analysis",
      state: "receiving",
      stages: ANALYSIS_STAGES.map(([key, label], i) => ({ key, label, status: i === 0 ? "active" : "pending" })),
      elapsed_s: 0,
      input_name: file.name,
      error: null,
    });
    clearTimeout(pollTimer);
    renderRail();

    const xhr = new XMLHttpRequest();
    state.xhr = xhr;
    xhr.open("POST", "/api/jobs/analysis");
    xhr.setRequestHeader("X-PVD-Token", TOKEN);
    xhr.setRequestHeader("Content-Type", "text/csv");
    xhr.setRequestHeader("X-PVD-Filename", encodeURIComponent(file.name));
    xhr.upload.addEventListener("progress", (ev) => {
      if (!state.upload) return;
      state.upload.loaded = ev.loaded;
      if (ev.lengthComputable) state.upload.total = ev.total;
      updateUpload();
    });
    const finish = () => {
      state.upload = null;
      state.xhr = null;
      state.busy = false;
      renderRail();
    };
    xhr.addEventListener("load", () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch (_) {
        body = null;
      }
      finish();
      if (xhr.status >= 200 && xhr.status < 300 && body && body.job) {
        applyJob(body.job);
      } else {
        const err = (body && body.error) || { title: "Upload failed", detail: `The local service answered with status ${xhr.status}.`, action: "Check the file and try again." };
        if (xhr.status === 401 || xhr.status === 403) {
          handleApiError(new ApiError(xhr.status, err), "Upload failed");
          return;
        }
        applyJob({ ...state.job, state: "failed", error: err, stages: state.job.stages.map((st) => (st.status === "active" ? { ...st, status: "failed" } : st)) });
      }
    });
    xhr.addEventListener("error", () => {
      finish();
      applyJob({
        ...state.job,
        state: "failed",
        error: { title: "Upload interrupted", detail: "The connection to the local service closed while sending the file.", action: "Check that gui.py is still running and try again." },
        stages: state.job.stages.map((st) => (st.status === "active" ? { ...st, status: "failed" } : st)),
      });
    });
    xhr.addEventListener("abort", () => {
      finish();
      applyJob({ ...state.job, state: "cancelled", error: null });
    });
    xhr.send(file);
  }

  async function cancelJob() {
    if (state.xhr) state.xhr.abort();
    try {
      const { job } = await api("/api/jobs/current/cancel", { method: "POST" });
      applyJob(job);
      toast("Job cancelled. No outputs were kept.");
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        refreshSession();
        return;
      }
      handleApiError(e, "Cancel failed");
    }
  }

  async function clearSession() {
    if (state.xhr) state.xhr.abort();
    try {
      await api("/api/session/clear", { method: "POST" });
      state.file = null;
      $("file-input").value = "";
      resetResults();
      applyJob(null);
      toast("Session data cleared. Your original files and configured model were not touched.");
      refreshSessionQuietly();
    } catch (e) {
      handleApiError(e, "Clear failed");
    }
  }

  async function download(name) {
    if (!state.result) return;
    try {
      const res = await api(`/api/jobs/${encodeURIComponent(state.result.job_id)}/files/${encodeURIComponent(name)}`, { raw: true });
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = h("a", { href: url, download: name, hidden: true });
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      toast(`${name} saved`);
    } catch (e) {
      handleApiError(e, `Could not download ${name}`);
    }
  }

  // ---------------------------------------------------------------- rail

  function renderRail() {
    const sess = state.session;
    const model = sess && sess.model;
    const active = isActive(state.job) || state.busy;
    const dot = $("model-dot");
    dot.className = "dot " + (model ? (model.available ? "ok" : "warn") : "");
    if (!model) {
      $("model-title").textContent = state.fatal ? "Service unavailable" : "Checking model…";
      $("model-detail").textContent = "";
    } else if (model.available) {
      $("model-title").textContent = "Trained model ready";
      $("model-detail").textContent = `${model.label || "model"} · ${model.source === "configured" ? "set at launch" : "default artifact location"}`;
    } else {
      $("model-title").textContent = "No trained model";
      $("model-detail").textContent = model.reason || "Launch with --model PATH to analyze your own CSV. The synthetic check still works.";
    }

    $("btn-synthetic").disabled = active || !!state.fatal;
    $("btn-choose").disabled = active || !!state.fatal;
    const analyze = $("btn-analyze");
    analyze.hidden = !state.file;
    analyze.disabled = active || !state.file || !(model && model.available) || !!state.fatal;
    $("analyze-hint").textContent =
      model && !model.available
        ? "Private analysis needs a trained model. The demo model from the synthetic check is never used for your files."
        : "Uses only the model configured when the app was launched.";

    const chip = $("file-chip");
    chip.hidden = !state.file;
    if (state.file) {
      $("file-name").textContent = state.file.name;
      $("file-size").textContent = fmtBytes(state.file.size);
    }

    if (sess) {
      const cols = $("req-columns");
      cols.replaceChildren(...sess.input.required_columns.map((c) => h("li", null, h("code", { text: c }))));
      $("input-formats").replaceChildren(
        h("dt", { text: "Separator" }), h("dd", { text: sess.input.separators }),
        h("dt", { text: "Decimal" }), h("dd", { text: sess.input.decimal }),
        h("dt", { text: "Timestamps" }), h("dd", { text: sess.input.timestamps }),
      );
      $("out-files").replaceChildren(...sess.outputs.map((f) => h("li", null, h("code", { text: f }))));
    }
  }

  // ---------------------------------------------------------------- top-level render

  function render() {
    renderRail();
    renderTopbar();
    const view = $("view");
    if (state.fatal) {
      view.replaceChildren(h("div", { className: "card" }, h("div", { className: "card-body" }, errorNote(state.fatal))));
      return;
    }
    const job = state.job;
    if (!job) {
      view.replaceChildren(welcomeCard());
      return;
    }
    if (job.state === "complete" && state.result && state.resultFor === job.id) {
      view.replaceChildren(resultsView());
      if (state.selected) renderInspect();
      return;
    }
    view.replaceChildren(jobCard(job));
    updateElapsed();
    updateUpload();
  }

  function renderTopbar() {
    const job = state.job;
    const title = $("view-title");
    const sub = $("view-sub");
    const chip = $("session-chip");
    if (!job) {
      title.textContent = "Workspace";
      sub.textContent = "Run the synthetic check or analyze a local CSV.";
      chip.hidden = true;
      return;
    }
    const demo = job.kind === "synthetic";
    title.textContent = demo ? "Synthetic check" : `Analysis · ${job.input_name || "local CSV"}`;
    chip.hidden = false;
    chip.replaceChildren(icon(demo ? "flask" : "lock"), demo ? "Demo session" : "Private session");
    const r = state.result;
    if (job.state === "complete" && r) {
      sub.textContent = `Completed in ${fmtElapsed(r.summary.elapsed_s)} · model inference ${fmtElapsed(r.summary.inference_s)} · ${r.summary.model_label}`;
    } else if (job.state === "complete") {
      sub.textContent = "Loading results…";
    } else if (job.state === "failed") {
      sub.textContent = "The job stopped. Nothing was made downloadable.";
    } else if (job.state === "cancelled") {
      sub.textContent = "Cancelled. Partial outputs were discarded.";
    } else {
      sub.textContent = demo ? "Fabricated customers only — no private data involved." : "Running on this computer with the configured model.";
    }
  }

  // ---------------------------------------------------------------- welcome

  function welcomeCard() {
    const step = (n, title, text) => h("li", { className: "step" }, h("span", { className: "step-n", text: n }), h("span", { className: "step-title", text: title }), h("p", { text }));
    return h(
      "section",
      { className: "card welcome", "aria-labelledby": "welcome-h" },
      h("span", { className: "badge badge-solar" }, icon("sun", 12), "Offline PV detector"),
      h("h2", { id: "welcome-h", text: "Find rooftop solar, its start date and size from 15-minute grid withdrawal." }),
      h("p", { className: "welcome-lede", text: "Everything runs on this computer. Start with the synthetic check to see the full workflow on fabricated customers, then analyze your own CSV with the trained model." }),
      h(
        "ol",
        { className: "steps" },
        step(1, "Run the synthetic check", "Fits a small demo model on fabricated customers and runs the official CLI next to the GUI."),
        step(2, "Inspect a customer", "See withdrawal over time, the estimated installation date and the typical day before and after."),
        step(3, "Download both CSVs", "predictions.csv and pv_extensions_predictions.csv, byte-identical to the CLI output."),
      ),
      h(
        "div",
        { className: "welcome-actions" },
        h("button", { type: "button", className: "btn btn-primary", onclick: startSynthetic, disabled: state.busy || !!state.fatal }, icon("flask"), "Run synthetic check"),
        h("button", { type: "button", className: "btn btn-secondary", onclick: () => $("file-input").click(), disabled: state.busy || !!state.fatal }, icon("file"), "Choose a CSV"),
      ),
      h(
        "p",
        { className: "note" },
        icon("info"),
        h("span", null, h("strong", { text: "What the check establishes. " }), "GUI and CLI write byte-identical, repeatable, schema-valid files for the same model and input. Its score on fabricated data is a regression signal, not evidence of real-world accuracy."),
      ),
    );
  }

  // ---------------------------------------------------------------- job card

  function jobCard(job) {
    const stages = h(
      "ol",
      { className: "stages", "aria-label": "Job stages" },
      job.stages.map((st) =>
        h(
          "li",
          { className: "stage", "data-status": job.state === "cancelled" && st.status === "failed" ? "stopped" : st.status },
          h(
            "span",
            { className: "stage-icon" },
            st.status === "done" ? icon("check", 12) : st.status === "failed" ? icon("x", 12) : st.status === "active" ? h("span", { className: "stage-pulse" }) : null,
          ),
          h("span", { className: "stage-label", text: st.label + (st.status === "active" ? "…" : job.state === "cancelled" && st.status === "failed" ? " — stopped" : "") }),
        ),
      ),
    );
    const active = isActive(job);
    const head = h(
      "div",
      { className: "card-head" },
      h(
        "div",
        null,
        h("h2", { className: "card-title", id: "job-h", text: job.kind === "synthetic" ? "Synthetic check" : "Analysis" }),
        h("p", { className: "card-sub", text: active ? "Only one job runs at a time. Elapsed time is measured, not estimated." : job.state === "failed" ? "Failed" : job.state === "cancelled" ? "Cancelled" : "Finished" }),
      ),
      active ? h("button", { type: "button", className: "btn btn-quiet btn-sm", onclick: cancelJob }, icon("stop", 12), "Cancel") : null,
    );
    const body = h(
      "div",
      { className: "card-body" },
      h(
        "div",
        { className: "job-grid" },
        stages,
        h("div", { className: "elapsed" }, h("span", { className: "elapsed-value num", id: "elapsed-value", text: fmtElapsed(currentElapsed()) }), h("span", { className: "elapsed-label", text: "Elapsed" })),
      ),
      job.state === "receiving" && state.upload
        ? h(
            "div",
            null,
            h("div", { className: "progress", role: "progressbar", "aria-label": "Upload", "aria-valuemin": 0, "aria-valuemax": 100, id: "upload-bar" }, h("div", { className: "progress-bar", id: "upload-fill" })),
            h("p", { className: "caption num", id: "upload-text" }),
          )
        : null,
      job.state === "failed" && job.error ? errorNote(job.error) : null,
      job.state === "cancelled" ? h("p", { className: "note" }, icon("info"), h("span", { text: "The worker was stopped and its files were deleted. Nothing from this run can be downloaded." })) : null,
      !active
        ? h(
            "div",
            { className: "welcome-actions" },
            h("button", { type: "button", className: "btn btn-primary btn-sm", onclick: startSynthetic }, "Run synthetic check"),
            h("button", { type: "button", className: "btn btn-secondary btn-sm", onclick: () => $("file-input").click() }, "Choose a CSV"),
          )
        : null,
    );
    return h("section", { className: "card", "aria-labelledby": "job-h" }, head, body);
  }

  function errorNote(err) {
    return h(
      "div",
      { className: "note note-danger", role: "alert" },
      icon("alert"),
      h("div", null, h("strong", { text: err.title }), h("p", { text: err.detail }), err.action ? h("p", null, h("strong", { text: "What to do: " }), err.action) : null),
    );
  }

  function updateUpload() {
    const u = state.upload;
    const fill = $("upload-fill");
    if (!u || !fill) return;
    const frac = u.total ? Math.min(1, u.loaded / u.total) : 0;
    fill.style.width = `${(frac * 100).toFixed(1)}%`;
    $("upload-bar").setAttribute("aria-valuenow", String(Math.round(frac * 100)));
    $("upload-text").textContent = `${fmtBytes(u.loaded)} of ${fmtBytes(u.total)} sent`;
  }

  // ---------------------------------------------------------------- results

  function resultsView() {
    const r = state.result;
    return h(
      "div",
      { className: "workspace" },
      h("div", { className: "col" }, summaryCard(r), r.synthetic ? syntheticCard(r) : null, outputsCard(r), qualityCard(r)),
      h("div", { className: "col" }, customersCard(r), h("section", { className: "card", id: "inspect-card", "aria-label": "Customer inspection" }, inspectPlaceholder())),
    );
  }

  function summaryCard(r) {
    const sm = r.summary;
    const share = sm.customers ? sm.pv / sm.customers : 0;
    const pvRows = r.customers.filter((c) => c.pv_installed);
    const regimeCounts = { event: 0, present_at_start: 0, insufficient: 0 };
    for (const c of pvRows) regimeCounts[c.regime] = (regimeCounts[c.regime] || 0) + 1;
    const maxR = Math.max(1, ...Object.values(regimeCounts));
    const split = h("div", { className: "split", role: "img", "aria-label": `${fmtPct(share)} of customers predicted with PV` }, h("div", { className: "split-pv" }));
    split.firstChild.style.width = fmtPct(share, 2);

    return h(
      "section",
      { className: "card", "aria-labelledby": "sum-h" },
      h(
        "div",
        { className: "card-head" },
        h("div", null, h("h2", { className: "card-title", id: "sum-h", text: "Detection summary" }), h("p", { className: "card-sub", text: sm.date_range.start ? `Input ${fmtDate(sm.date_range.start)} – ${fmtDate(sm.date_range.end)}` : "No dated input" })),
        h("span", { className: `badge ${r.kind === "synthetic" ? "" : "badge-info"}` }, icon(r.kind === "synthetic" ? "flask" : "lock", 12), r.kind === "synthetic" ? "Synthetic" : "Private"),
      ),
      h(
        "div",
        { className: "card-body" },
        h(
          "div",
          { className: "kpis" },
          kpi(fmtInt(sm.customers), "Customers processed"),
          kpi(fmtInt(sm.pv), "PV detected", "kpi-solar"),
          kpi(fmtInt(sm.non_pv), "No PV detected"),
        ),
        h("div", null, split),
        h("div", { className: "split-legend" }, h("span", null, h("span", { className: "swatch swatch-pv" }), `PV ${fmtPct(share)}`), h("span", null, h("span", { className: "swatch swatch-none" }), `No PV ${fmtPct(1 - share)}`)),
        h("hr", { className: "card-divider" }),
        h("p", { className: "section-label", text: "How the installation date was found (PV customers)" }),
        h(
          "div",
          { className: "bars" },
          Object.entries(REGIMES).map(([key, label]) => {
            const fill = h("div", { className: `bar-fill r-${key}` });
            fill.style.width = `${((regimeCounts[key] || 0) / maxR) * 100}%`;
            return [h("span", { className: "bar-label", text: label }), h("div", { className: "bar-track" }, fill), h("span", { className: "bar-value num", text: fmtInt(regimeCounts[key] || 0) })];
          }),
        ),
      ),
    );
  }

  const kpi = (value, label, extra = "") => h("div", { className: `kpi ${extra}` }, h("span", { className: "kpi-value num", text: value }), h("span", { className: "kpi-label", text: label }));

  function syntheticCard(r) {
    const m = r.synthetic.metrics;
    const metric = (v, label) => h("div", { className: "metric" }, h("span", { className: "metric-value num", text: v }), h("span", { className: "metric-label", text: label }));
    return h(
      "section",
      { className: "card", "aria-labelledby": "syn-h" },
      h("div", { className: "card-head" }, h("div", null, h("h2", { className: "card-title", id: "syn-h", text: "Synthetic regression demonstration" }), h("p", { className: "card-sub", text: `${r.synthetic.train_customers} training · ${r.synthetic.test_customers} test customers · seeds ${r.synthetic.seeds.train}/${r.synthetic.seeds.test}` }))),
      h(
        "div",
        { className: "card-body" },
        h("div", { className: "synthetic-banner", role: "note" }, icon("alert"), h("div", null, h("strong", { text: "Not evidence of real-world accuracy" }), h("span", { text: "Fabricated customers from the development generator. This score only guards against regressions." }))),
        h("div", { className: "metrics" }, metric(m.score.toFixed(1), "Full score"), metric(m.f1.toFixed(3), "F1"), metric(m.D.toFixed(3), "Date D"), metric(m.P.toFixed(3), "Capacity P")),
        h("p", { className: "caption num", text: `TP ${m.tp} · FP ${m.fp} · FN ${m.fn}` }),
      ),
    );
  }

  function outputsCard(r) {
    const fileRow = (name, info) =>
      h(
        "div",
        { className: "file-row" },
        h("span", { className: "file-icon" }, icon("file", 16)),
        h(
          "div",
          { className: "file-info" },
          h("div", { className: "file-name", text: name }),
          h("div", { className: "file-meta" }, h("span", { className: "num", text: `${fmtInt(info.rows)} ${info.rows === 1 ? "row" : "rows"}` }), h("span", { className: `badge ${info.valid ? "badge-success" : "badge-danger"}` }, icon(info.valid ? "check" : "x", 12), info.valid ? "Valid" : "Invalid")),
        ),
        h("button", { type: "button", className: "btn btn-quiet btn-sm", onclick: () => download(name), "aria-label": `Download ${name}` }, icon("download", 14), "Download"),
      );
    const failed = r.checks.filter((c) => !c.passed).length;
    return h(
      "section",
      { className: "card", "aria-labelledby": "out-h2" },
      h(
        "div",
        { className: "card-head" },
        h("div", null, h("h2", { className: "card-title", id: "out-h2", text: "Outputs" }), h("p", { className: "card-sub", text: "Exact files written by the submission writers" })),
        h("span", { className: `badge ${failed ? "badge-danger" : "badge-success"}` }, icon(failed ? "x" : "check", 12), failed ? `${failed} check${failed > 1 ? "s" : ""} failed` : `${r.checks.length} checks passed`),
      ),
      h(
        "div",
        { className: "card-body" },
        h("div", { className: "files" }, fileRow("predictions.csv", r.outputs.predictions), fileRow("pv_extensions_predictions.csv", r.outputs.extensions)),
        r.outputs.extensions.note ? h("p", { className: "note" }, icon("info"), h("span", { text: r.outputs.extensions.note })) : null,
        h(
          "ul",
          { className: "checks", "aria-label": "Output checks" },
          r.checks.map((c) =>
            h(
              "li",
              { className: "check" },
              h("span", { className: `check-icon ${c.passed ? "pass" : "fail"}`, "aria-label": c.passed ? "passed" : "failed", role: "img" }, icon(c.passed ? "check" : "x", 12)),
              h("div", null, h("div", { className: "check-label", text: c.label }), c.detail ? h("div", { className: "check-detail", text: c.detail }) : null),
            ),
          ),
        ),
      ),
    );
  }

  function qualityCard(r) {
    const q = r.summary.quality;
    const dl = q.day_length_counts || {};
    const item = (label, value) => h("div", null, h("dt", { text: label }), h("dd", { className: "num", text: value }));
    return h(
      "section",
      { className: "card", "aria-labelledby": "q-h" },
      h("div", { className: "card-head" }, h("div", null, h("h2", { className: "card-title", id: "q-h", text: "Input quality" }), h("p", { className: "card-sub", text: "Counts reported by the loader" }))),
      h(
        "div",
        { className: "card-body" },
        h(
          "dl",
          { className: "facts" },
          item("Rows read", fmtInt(q.rows_total)),
          item("Duplicate timestamps dropped", fmtInt(q.rows_duplicate)),
          item("Rows with invalid timestamp", fmtInt(q.rows_bad_epoch)),
          item("Rows with invalid energy", fmtInt(q.rows_bad_energy)),
          item("Customer-days with 96 slots", fmtInt(dl["96"])),
          item("DST days (92 / 100 slots)", `${fmtInt(dl["92"])} / ${fmtInt(dl["100"])}`),
          item("Partial customer-days", fmtInt(dl.other)),
          item("Customers with export (signed) readings", fmtInt(q.customers_signed)),
          item("Customers without usable history", fmtInt(q.customers_without_usable_history)),
        ),
      ),
    );
  }

  // ---------------------------------------------------------------- customer table

  function filteredCustomers() {
    const q = state.query.trim().toLowerCase();
    return state.result.customers.filter((c) => {
      if (state.filter === "pv" && !c.pv_installed) return false;
      if (state.filter === "nopv" && c.pv_installed) return false;
      return !q || c.customer_id.toLowerCase().includes(q);
    });
  }

  function customersCard(r) {
    const pv = r.customers.filter((c) => c.pv_installed).length;
    const search = h("input", {
      type: "search",
      placeholder: "Search customer ID",
      "aria-label": "Search customer ID",
      value: state.query,
      autocomplete: "off",
      spellcheck: "false",
      oninput: (ev) => {
        state.query = ev.target.value;
        state.page = 0;
        renderTable();
      },
    });
    const seg = segmented(
      "Filter customers",
      [
        ["all", `All ${r.customers.length}`],
        ["pv", `PV ${pv}`],
        ["nopv", `No PV ${r.customers.length - pv}`],
      ],
      state.filter,
      (v) => {
        state.filter = v;
        state.page = 0;
        renderTable();
      },
      "",
      "filter-seg",
    );
    return h(
      "section",
      { className: "card", "aria-labelledby": "cust-h" },
      h("div", { className: "card-head" }, h("div", null, h("h2", { className: "card-title", id: "cust-h", text: "Customers" }), h("p", { className: "card-sub", text: "Select a row to inspect it. Downloads always contain every customer." }))),
      h("div", { className: "card-body" }, h("div", { className: "toolbar" }, h("label", { className: "search" }, icon("search", 16), search), seg), h("div", { id: "table-slot" }, tableEl())),
    );
  }

  function segmented(label, options, current, onPick, extra = "", id = null) {
    const wrap = h("div", { className: `segmented ${extra}`, role: "group", "aria-label": label, id });
    for (const [value, text, disabled] of options) {
      wrap.appendChild(
        h("button", {
          type: "button",
          "aria-pressed": String(value === current),
          disabled: !!disabled,
          text,
          onclick: (ev) => {
            const btn = ev.currentTarget;
            if (btn.getAttribute("aria-pressed") === "true") return;
            for (const b of wrap.children) b.setAttribute("aria-pressed", String(b === btn));
            onPick(value);
          },
        }),
      );
    }
    return wrap;
  }

  function renderTable() {
    const slot = $("table-slot");
    if (slot) slot.replaceChildren(tableEl());
    const seg = $("filter-seg");
    if (seg) {
      const keys = ["all", "pv", "nopv"];
      [...seg.children].forEach((b, i) => b.setAttribute("aria-pressed", String(keys[i] === state.filter)));
    }
  }

  function tableEl() {
    const r = state.result;
    const synthetic = r.kind === "synthetic";
    const rows = filteredCustomers();
    const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    state.page = Math.min(state.page, pages - 1);
    const start = state.page * PAGE_SIZE;
    const pageRows = rows.slice(start, start + PAGE_SIZE);

    const head = h(
      "tr",
      null,
      h("th", { scope: "col", text: "Customer ID" }),
      h("th", { scope: "col", text: "PV detected" }),
      h("th", { scope: "col", text: "Installation date" }),
      h("th", { scope: "col", className: "r", text: "Capacity (kWp)" }),
      h("th", { scope: "col", text: "Date regime" }),
      synthetic ? h("th", { scope: "col", text: "Known (synthetic)" }) : null,
    );
    const body = h("tbody", { onkeydown: onTableKey });
    if (!pageRows.length) {
      body.appendChild(h("tr", { className: "empty-row" }, h("td", { colspan: synthetic ? 6 : 5, text: "No customers match this search." })));
    }
    for (const c of pageRows) {
      const sel = c.customer_id === state.selected;
      const tr = h(
        "tr",
        { "aria-selected": String(sel), "data-cid": c.customer_id, onclick: () => selectCustomer(c.customer_id, true) },
        h("td", null, h("button", { type: "button", className: "id-btn", text: c.customer_id, "aria-label": `Inspect customer ${c.customer_id}` })),
        h("td", null, h("span", { className: c.pv_installed ? "yes" : "no", text: c.pv_installed ? "Yes" : "No" })),
        h("td", { className: "num" }, c.installation_date ? c.installation_date : h("span", { className: "dash", text: "—", "aria-label": "empty" })),
        h("td", { className: "r num" }, c.capacity_kwp !== null && c.capacity_kwp !== undefined ? c.capacity_kwp.toFixed(2) : h("span", { className: "dash", text: "—", "aria-label": "empty" })),
        h("td", { className: "regime-cell", text: c.pv_installed ? REGIMES[c.regime] || c.regime : "" }, c.pv_installed ? null : h("span", { className: "dash", text: "—", "aria-label": "not applicable" })),
        synthetic ? h("td", null, truthCell(c)) : null,
      );
      body.appendChild(tr);
    }
    const pager = h(
      "div",
      { className: "pager" },
      h("span", { className: "num", text: rows.length ? `${fmtInt(start + 1)}–${fmtInt(start + pageRows.length)} of ${fmtInt(rows.length)}` : "0 results" }),
      h(
        "div",
        { className: "pager-btns" },
        h("button", { type: "button", className: "icon-btn", "aria-label": "Previous page", disabled: state.page === 0, onclick: () => { state.page -= 1; renderTable(); } }, icon("left", 14)),
        h("button", { type: "button", className: "icon-btn", "aria-label": "Next page", disabled: state.page >= pages - 1, onclick: () => { state.page += 1; renderTable(); } }, icon("right", 14)),
      ),
    );
    return h("div", { className: "col" }, h("div", { className: "table-wrap" }, h("table", { className: "data" }, h("caption", { className: "visually-hidden", text: "Predictions per customer" }), h("thead", null, head), body)), pager);
  }

  function truthCell(c) {
    const t = c.truth;
    if (!t) return h("span", { className: "dash", text: "—" });
    const agree = t.pv_installed === c.pv_installed;
    return h("span", { className: agree ? "match" : "mismatch", text: `${t.pv_installed ? "PV" : "No PV"} ${agree ? "✓" : "✗"}` });
  }

  function onTableKey(ev) {
    if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp") return;
    const tr = ev.target.closest("tr[data-cid]");
    if (!tr) return;
    const next = ev.key === "ArrowDown" ? tr.nextElementSibling : tr.previousElementSibling;
    if (!next || !next.dataset.cid) return;
    ev.preventDefault();
    selectCustomer(next.dataset.cid, false);
    const btn = next.querySelector(".id-btn");
    if (btn) btn.focus();
  }

  // ---------------------------------------------------------------- inspection

  function inspectPlaceholder() {
    return h("div", { className: "card-body" }, h("div", { className: "empty" }, h("strong", { text: "No customer selected" }), h("span", { text: "Choose a row in the table to see its withdrawal history and the estimated installation." })));
  }

  async function selectCustomer(cid, scroll) {
    if (!state.result) return;
    state.selected = cid;
    for (const tr of document.querySelectorAll("tr[data-cid]")) tr.setAttribute("aria-selected", String(tr.dataset.cid === cid));
    const jobId = state.result.job_id;
    let diag = state.diagCache.get(cid);
    if (!diag) {
      const card = $("inspect-card");
      if (card) card.setAttribute("aria-busy", "true");
      try {
        diag = await api(`/api/jobs/${encodeURIComponent(jobId)}/customers/${encodeURIComponent(cid)}/diagnostics`);
        state.diagCache.set(cid, diag);
      } catch (e) {
        if (card) card.removeAttribute("aria-busy");
        handleApiError(e, "Could not load the customer view");
        return;
      }
    }
    if (!state.result || state.result.job_id !== jobId || state.selected !== cid) return;
    state.diag = diag;
    renderInspect();
    if (scroll) {
      const card = $("inspect-card");
      if (card && card.getBoundingClientRect().top > window.innerHeight * 0.6) card.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function renderInspect() {
    const card = $("inspect-card");
    const d = state.diag;
    if (!card) return;
    card.removeAttribute("aria-busy");
    if (!d || d.customer_id !== state.selected) {
      card.replaceChildren(inspectPlaceholder());
      return;
    }
    const p = d.prediction;
    const synthetic = !!d.truth;
    const head = h(
      "div",
      { className: "card-head" },
      h("div", null, h("p", { className: "card-sub", text: "Customer" }), h("h2", { className: "inspect-id", text: d.customer_id })),
      h(
        "div",
        { className: "inspect-chips" },
        h("span", { className: `badge ${p.pv_installed ? "badge-solar" : ""}` }, icon(p.pv_installed ? "sun" : "x", 12), p.pv_installed ? "PV detected" : "No PV detected"),
        p.pv_installed ? h("span", { className: "badge", text: d.regime.label }) : null,
        synthetic ? h("span", { className: "badge" }, icon("flask", 12), "Synthetic") : null,
      ),
    );

    const body = h("div", { className: "card-body", id: "inspect-body" });
    body.appendChild(predictionTiles(d));

    if (!d.usable) {
      body.appendChild(h("div", { className: "empty" }, h("strong", { text: "No usable history" }), h("span", { text: d.message || "This customer has no valid days in the input, so there is nothing to plot." })));
      card.replaceChildren(head, body);
      return;
    }

    const weeklyOnly = d.series.resolution === "weekly";
    const metricSeg = segmented(
      "Quantity",
      [
        ["total", "Daily withdrawal"],
        ["midday", "Midday power"],
      ],
      state.metric,
      (v) => {
        state.metric = v;
        renderInspect();
      },
      "segmented-sm",
    );
    const resSeg = segmented(
      "Resolution",
      [
        ["weekly", "Weekly"],
        ["daily", "Daily", weeklyOnly],
      ],
      weeklyOnly ? "weekly" : state.resolution,
      (v) => {
        state.resolution = v;
        renderInspect();
      },
      "segmented-sm",
    );
    const unit = state.metric === "total" ? "kWh/day" : "kW";
    const chartTitle = state.metric === "total" ? "Grid withdrawal per day" : "Mean withdrawal 10:00–15:00";
    const legend = h(
      "div",
      { className: "legend" },
      h("span", { className: "lg-series" }, h("span", { className: "legend-line" }), `${state.resolution === "daily" && !weeklyOnly ? "Daily" : "Weekly mean"} (${unit})`),
      d.markers.predicted_date ? h("span", { className: "lg-pred" }, h("span", { className: "legend-line dashed" }), "Estimated installation") : null,
      d.markers.known_date ? h("span", { className: "lg-known" }, h("span", { className: "legend-line" }), "Known installation (synthetic)") : null,
    );
    const timeBox = h("div", { className: "chart-box", id: "time-chart" });
    body.append(
      h("hr", { className: "card-divider" }),
      h("div", { className: "chart-head" }, h("div", null, h("div", { className: "chart-title", text: chartTitle }), h("div", { className: "chart-units", text: state.metric === "total" ? "kWh per day · fixed UTC+1 days · gaps are missing data" : "kW average over 10:00–14:59 UTC+1 · where PV shows first" })), h("div", { className: "toolbar" }, metricSeg, resSeg)),
      timeBox,
      legend,
    );

    const prof = d.profile;
    const profBox = h("div", { className: "chart-box", id: "profile-chart" });
    body.append(
      h("hr", { className: "card-divider" }),
      h("div", { className: "chart-head" }, h("div", null, h("div", { className: "chart-title", text: "Typical day before vs after the estimated date" }), h("div", { className: "chart-units", text: `kW per 15-minute slot · up to ${prof.window_days} days each side · observed withdrawal, not recovered generation` }))),
    );
    if (prof.available) {
      body.append(
        profBox,
        h(
          "div",
          { className: "legend" },
          h("span", { className: "lg-before" }, h("span", { className: "legend-line" }), `Before · ${prof.before.days} days (${fmtDate(prof.before.start)} – ${fmtDate(prof.before.end)})`),
          h("span", { className: "lg-after" }, h("span", { className: "legend-line" }), `After · ${prof.after.days} days (${fmtDate(prof.after.start)} – ${fmtDate(prof.after.end)})`),
        ),
        h("p", { className: "caption", text: "Descriptive only: the two windows fall in different seasons, so this is not seasonally adjusted evidence of PV by itself." }),
      );
    } else {
      body.append(h("p", { className: "note" }, icon("info"), h("span", { text: prof.reason || "Not enough coverage on both sides of the date for a fair comparison." })));
    }

    const cov = d.coverage;
    body.append(
      h("hr", { className: "card-divider" }),
      h(
        "dl",
        { className: "facts" },
        fact("Data from", cov.first_date ? `${fmtDate(cov.first_date)} – ${fmtDate(cov.last_date)}` : "—"),
        fact("Span", `${fmtInt(cov.span_days)} days`),
        fact("Days with data", `${fmtInt(cov.days_with_data)} (${fmtInt(cov.valid_days)} valid)`),
        fact("Missing 15-minute slots", fmtPct(cov.missing_slot_share, 1)),
      ),
      p.pv_installed
        ? h("p", { className: "note" }, icon("info"), h("span", null, h("strong", { text: `${d.regime.label}. ` }), d.regime.explanation))
        : h("p", { className: "note" }, icon("info"), h("span", { text: "No PV detected: installation date and capacity stay blank, exactly as in predictions.csv." })),
    );

    card.replaceChildren(head, body);
    drawCharts();
  }

  const fact = (label, value) => h("div", null, h("dt", { text: label }), h("dd", { className: "num", text: value }));

  function predictionTiles(d) {
    const p = d.prediction;
    const t = d.truth;
    const tile = (label, value, known) => h("div", { className: "tile" }, h("div", { className: "tile-label", text: label }), h("div", { className: "tile-value num", text: value }), known ? h("div", { className: "tile-known num", text: known }) : null);
    let knownDate = null;
    let knownCap = null;
    let knownPv = null;
    if (t) {
      knownPv = `Known: ${t.pv_installed ? "PV" : "no PV"}`;
      if (t.installation_date) {
        knownDate = `Known ${fmtDate(t.installation_date)}`;
        if (p.installation_date) knownDate += ` · Δ ${Math.round((dateMs(p.installation_date) - dateMs(t.installation_date)) / 86400000)} d`;
      }
      if (t.capacity_kwp !== null && t.capacity_kwp !== undefined) {
        knownCap = `Known ${Number(t.capacity_kwp).toFixed(2)} kWp`;
        if (p.capacity_kwp) knownCap += ` · Δ ${(((p.capacity_kwp - t.capacity_kwp) / t.capacity_kwp) * 100).toFixed(0)}%`;
      }
    }
    return h(
      "div",
      { className: "tiles" },
      tile("PV detected", p.pv_installed ? "Yes" : "No", knownPv),
      tile("Installation date", p.installation_date ? fmtDate(p.installation_date) : "—", knownDate),
      tile("Capacity", p.capacity_kwp !== null && p.capacity_kwp !== undefined ? fmtKwp(p.capacity_kwp) : "—", knownCap),
    );
  }

  function drawCharts() {
    const d = state.diag;
    if (!d || !d.usable) return;
    const tbox = $("time-chart");
    if (tbox) {
      const src = state.resolution === "daily" && d.series.resolution === "daily" ? d.series : d.weekly;
      const ys = state.metric === "total" ? src.total_kwh : src.midday_kw;
      const unit = state.metric === "total" ? "kWh/day" : "kW";
      const xs = src.dates.map(dateMs);
      const markers = [];
      if (d.markers.predicted_date) markers.push({ x: dateMs(d.markers.predicted_date), cls: "marker-pred", label: "Est. install" });
      if (d.markers.known_date) markers.push({ x: dateMs(d.markers.known_date), cls: "marker-known", label: "Known" });
      const bands = [];
      if (d.profile.available) {
        bands.push({ x0: dateMs(d.profile.before.start), x1: dateMs(d.profile.before.end) + 86400000, cls: "window-band" });
        bands.push({ x0: dateMs(d.profile.after.start), x1: dateMs(d.profile.after.end) + 86400000, cls: "window-band after" });
      }
      lineChart(tbox, {
        xs,
        series: [{ ys, cls: "series-main", area: true }],
        xKind: "date",
        markers,
        bands,
        height: 230,
        ariaLabel: `${state.metric === "total" ? "Daily withdrawal" : "Midday power"} for customer ${d.customer_id}`,
        tip: (i) => [src === d.weekly ? `Week of ${fmtDateLong(xs[i])}` : fmtDateLong(xs[i]), fmtVal(ys[i], unit)],
      });
    }
    const pbox = $("profile-chart");
    if (pbox && d.profile.available) {
      const xs = Array.from({ length: 96 }, (_, i) => i * 0.25 + 0.125);
      const before = d.profile.before.kw;
      const after = d.profile.after.kw;
      lineChart(pbox, {
        xs,
        series: [
          { ys: before, cls: "series-before" },
          { ys: after, cls: "series-after" },
        ],
        xKind: "hour",
        xDomain: [0, 24],
        markers: [],
        bands: [],
        height: 200,
        ariaLabel: `Average daily withdrawal profile before and after the estimated installation for customer ${d.customer_id}`,
        tip: (i) => [`${fmtHour(i * 0.25)}–${fmtHour(i * 0.25 + 0.25)}`, `Before ${fmtVal(before[i], "kW")}`, `After ${fmtVal(after[i], "kW")}`],
      });
    }
  }

  // ---------------------------------------------------------------- charts

  function niceStep(span, count) {
    const raw = span / Math.max(1, count);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const norm = raw / mag;
    return (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  }

  function yTicks(lo, hi, count) {
    if (!(hi > lo)) hi = lo + 1;
    const step = niceStep(hi - lo, count);
    const start = Math.floor(lo / step) * step;
    const end = Math.ceil(hi / step) * step;
    const ticks = [];
    for (let v = start; v <= end + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return { ticks, lo: start, hi: end, step };
  }

  function dateTicks(x0, x1, maxTicks) {
    const d0 = new Date(x0);
    const months = (new Date(x1).getUTCFullYear() - d0.getUTCFullYear()) * 12 + new Date(x1).getUTCMonth() - d0.getUTCMonth() + 1;
    const step = [1, 2, 3, 4, 6, 12, 24, 36].find((st) => months / st <= maxTicks) || 48;
    const ticks = [];
    let y = d0.getUTCFullYear();
    let m = d0.getUTCMonth();
    m = Math.ceil(m / step) * step;
    if (step >= 12) {
      m = 0;
      y = Math.ceil((y + (d0.getUTCMonth() > 0 || d0.getUTCDate() > 1 ? 1 : 0)) / (step / 12)) * (step / 12);
    }
    for (let guard = 0; guard < 200; guard++) {
      const t = Date.UTC(y + Math.floor(m / 12), m % 12, 1);
      if (t > x1) break;
      if (t >= x0) {
        const dt = new Date(t);
        ticks.push({ x: t, label: step >= 12 ? String(dt.getUTCFullYear()) : dt.getUTCMonth() === 0 || ticks.length === 0 ? `${MONTHS[dt.getUTCMonth()]} ${dt.getUTCFullYear()}` : MONTHS[dt.getUTCMonth()] });
      }
      m += step;
    }
    return ticks;
  }

  function fmtTick(v, step) {
    const mag = Math.floor(Math.log10(step));
    const fractional = (step / 10 ** mag) % 1 !== 0;
    const d = Math.min(3, Math.max(0, -mag) + (fractional ? 1 : 0));
    return v.toFixed(d);
  }

  function lineChart(box, o) {
    const W = Math.max(300, Math.round(box.clientWidth || 600));
    const H = o.height;
    const m = { l: 44, r: 12, t: 24, b: 26 };
    const pw = W - m.l - m.r;
    const ph = H - m.t - m.b;
    const xs = o.xs;
    const finite = o.series.flatMap((se) => se.ys.filter((v) => v !== null && Number.isFinite(v)));

    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": o.ariaLabel });
    if (!xs.length || !finite.length) {
      box.replaceChildren(h("div", { className: "empty" }, h("strong", { text: "No data points" }), h("span", { text: "Every point in this window is missing." })));
      return;
    }
    const x0 = o.xDomain ? o.xDomain[0] : xs[0];
    const x1 = o.xDomain ? o.xDomain[1] : Math.max(xs[xs.length - 1], xs[0] + 1);
    let lo = Math.min(0, ...finite);
    let hi = Math.max(...finite);
    if (hi === lo) hi = lo + 1;
    const yt = yTicks(lo, hi + (hi - lo) * 0.05, Math.max(3, Math.floor(ph / 45)));
    lo = yt.lo;
    hi = yt.hi;
    const X = (v) => m.l + ((v - x0) / (x1 - x0)) * pw;
    const Y = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;

    const grid = s("g", { class: "grid" });
    const axis = s("g", { class: "axis" });
    for (const t of yt.ticks) {
      grid.appendChild(s("line", { x1: m.l, x2: m.l + pw, y1: Y(t), y2: Y(t) }));
      axis.appendChild(s("text", { x: m.l - 8, y: Y(t) + 3.5, "text-anchor": "end", text: fmtTick(t, yt.step) }));
    }
    const xticks =
      o.xKind === "hour"
        ? [0, 3, 6, 9, 12, 15, 18, 21, 24].filter((hh) => pw > 420 || hh % 6 === 0).map((hh) => ({ x: hh, label: fmtHour(hh) }))
        : dateTicks(x0, x1, Math.max(2, Math.floor(pw / 78)));
    for (const t of xticks) {
      const px = X(t.x);
      grid.appendChild(s("line", { x1: px, x2: px, y1: m.t, y2: m.t + ph, "stroke-dasharray": "2 3" }));
      axis.appendChild(s("text", { x: px, y: H - 8, "text-anchor": "middle", text: t.label }));
    }
    svg.appendChild(grid);

    for (const b of o.bands) {
      const a = Math.max(m.l, X(b.x0));
      const z = Math.min(m.l + pw, X(b.x1));
      if (z > a) svg.appendChild(s("rect", { class: b.cls, x: a, y: m.t, width: z - a, height: ph }));
    }

    if (lo < 0) svg.appendChild(s("line", { class: "axis-base", x1: m.l, x2: m.l + pw, y1: Y(0), y2: Y(0) }));
    svg.appendChild(s("line", { class: "axis-base", x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph }));

    for (const se of o.series) {
      let line = "";
      let area = "";
      let runStart = null;
      let prevX = null;
      for (let i = 0; i < xs.length; i++) {
        const v = se.ys[i];
        const px = X(xs[i]).toFixed(1);
        if (v === null || !Number.isFinite(v)) {
          if (runStart !== null && se.area) area += `L${prevX},${Y(Math.max(lo, 0)).toFixed(1)}L${runStart},${Y(Math.max(lo, 0)).toFixed(1)}Z`;
          runStart = null;
          continue;
        }
        const py = Y(v).toFixed(1);
        if (runStart === null) {
          line += `M${px},${py}`;
          if (se.area) area += `M${px},${py}`;
          runStart = px;
        } else {
          line += `L${px},${py}`;
          if (se.area) area += `L${px},${py}`;
        }
        prevX = px;
      }
      if (runStart !== null && se.area) area += `L${prevX},${Y(Math.max(lo, 0)).toFixed(1)}L${runStart},${Y(Math.max(lo, 0)).toFixed(1)}Z`;
      if (se.area && area) svg.appendChild(s("path", { class: "area-main", d: area }));
      svg.appendChild(s("path", { class: `series ${se.cls}`, d: line }));
    }

    let lastLabelX = -Infinity;
    let row = 0;
    for (const mk of o.markers) {
      const px = X(mk.x);
      if (px < m.l - 0.5 || px > m.l + pw + 0.5) continue;
      row = Math.abs(px - lastLabelX) < 80 ? row + 1 : 0;
      lastLabelX = px;
      const g = s("g", { class: mk.cls });
      g.appendChild(s("line", { x1: px, x2: px, y1: m.t - 4 + row * 14, y2: m.t + ph }));
      const label = s("text", { class: "marker-label", x: px + 4, y: m.t - 8 + row * 14, text: mk.label });
      if (px > m.l + pw - 80) {
        label.setAttribute("x", String(px - 4));
        label.setAttribute("text-anchor", "end");
      }
      g.appendChild(label);
      svg.appendChild(g);
    }
    svg.appendChild(axis);

    const hoverLine = s("line", { class: "hover-line", y1: m.t, y2: m.t + ph, visibility: "hidden" });
    const dots = o.series.map(() => s("circle", { class: "hover-dot", r: 3.5, visibility: "hidden" }));
    const overlay = s("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" });
    svg.append(hoverLine, ...dots, overlay);

    const tip = h("div", { className: "tooltip", hidden: true });
    const hide = () => {
      tip.hidden = true;
      hoverLine.setAttribute("visibility", "hidden");
      dots.forEach((dd) => dd.setAttribute("visibility", "hidden"));
    };
    overlay.addEventListener("pointerleave", hide);
    overlay.addEventListener("pointermove", (ev) => {
      const rect = svg.getBoundingClientRect();
      const scale = W / rect.width;
      const vx = (ev.clientX - rect.left) * scale;
      const xv = x0 + ((vx - m.l) / pw) * (x1 - x0);
      let a = 0;
      let b = xs.length - 1;
      while (b - a > 1) {
        const mid = (a + b) >> 1;
        if (xs[mid] < xv) a = mid;
        else b = mid;
      }
      const i = Math.abs(xs[a] - xv) <= Math.abs(xs[b] - xv) ? a : b;
      const px = X(xs[i]);
      hoverLine.setAttribute("x1", px);
      hoverLine.setAttribute("x2", px);
      hoverLine.setAttribute("visibility", "visible");
      let topY = m.t + ph;
      o.series.forEach((se, k) => {
        const v = se.ys[i];
        if (v === null || !Number.isFinite(v)) {
          dots[k].setAttribute("visibility", "hidden");
          return;
        }
        dots[k].setAttribute("cx", px);
        dots[k].setAttribute("cy", Y(v));
        dots[k].setAttribute("visibility", "visible");
        topY = Math.min(topY, Y(v));
      });
      const lines = o.tip(i);
      tip.replaceChildren(h("div", { text: lines[0] }), ...lines.slice(1).map((t) => h("div", { className: "tt-sub", text: t })));
      tip.hidden = false;
      tip.style.left = `${(px / W) * rect.width}px`;
      tip.style.top = `${(Math.max(topY, m.t + 30) / H) * rect.height}px`;
    });

    box.replaceChildren(svg, tip);
  }

  // ---------------------------------------------------------------- toasts

  function toast(text) {
    const el = h("div", { className: "toast" }, icon("info"), h("span", { text }));
    $("toasts").appendChild(el);
    setTimeout(() => el.remove(), 4200);
  }

  // ---------------------------------------------------------------- wiring

  function init() {
    $("btn-synthetic").addEventListener("click", startSynthetic);
    $("btn-choose").addEventListener("click", () => $("file-input").click());
    $("btn-analyze").addEventListener("click", startAnalysis);
    $("btn-clear").addEventListener("click", clearSession);
    $("file-input").addEventListener("change", (ev) => {
      const f = ev.target.files && ev.target.files[0];
      state.file = f || null;
      renderRail();
      if (f) $("btn-analyze").focus();
    });

    let resizeRaf = 0;
    let lastWidth = 0;
    new ResizeObserver((entries) => {
      const w = Math.round(entries[0].contentRect.width);
      if (w === lastWidth) return;
      lastWidth = w;
      cancelAnimationFrame(resizeRaf);
      resizeRaf = requestAnimationFrame(drawCharts);
    }).observe($("view"));

    window.addEventListener("focus", () => {
      if (!isActive(state.job) && !state.fatal) refreshSessionQuietly();
    });

    render();
    refreshSession();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
