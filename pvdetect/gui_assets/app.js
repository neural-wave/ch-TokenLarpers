/* PV Detect local GUI. Vanilla JS, no external resources.
   Layout and chart styling ported from Open Climate Fix's Quartz Energy app (MIT).
   Untrusted text (customer IDs, file names, server messages) is only written through
   textContent / attribute setters, never parsed as markup. */
(() => {
  "use strict";

  const TOKEN = document.querySelector('meta[name="pvd-token"]').getAttribute("content");
  const SVG_NS = "http://www.w3.org/2000/svg";
  const POLL_MS = 600;
  const C = {
    yellow: "#FFD053",
    yellow100: "#FFDDB1",
    blue: "#53D6FF",
    blue100: "#C7DCF2",
    mint: "#B5F5AA",
    orange: "#FF9736",
    stripeA: "#191A1A",
    stripeB: "#2C2C2C",
  };
  const REGIME = { event: "Change detected", present_at_start: "Present from start", insufficient: "Weak evidence" };
  const ANALYSIS_STAGES = [
    ["receive", "Receiving input"],
    ["load", "Validating and loading data"],
    ["predict", "Building features and predicting"],
    ["write", "Writing outputs"],
    ["check", "Checking outputs"],
  ];
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const nf = new Intl.NumberFormat("en-US");

  const state = {
    session: null,
    job: null,
    jobSyncedAt: 0,
    result: null,
    resultFor: null,
    upload: null,
    xhr: null,
    query: "",
    filter: "all",
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

  // ------------------------------------------------------------------ DOM

  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    setAttrs(el, attrs);
    append(el, children);
    return el;
  }

  function s(tag, attrs, ...children) {
    const el = document.createElementNS(SVG_NS, tag);
    setAttrs(el, attrs);
    append(el, children);
    return el;
  }

  function setAttrs(el, attrs) {
    if (!attrs) return;
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "text") el.textContent = String(v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else el.setAttribute(k === "className" ? "class" : k, v === true ? "" : String(v));
    }
  }

  function append(el, children) {
    for (const c of children.flat(Infinity)) {
      if (c === null || c === undefined || c === false) continue;
      el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
    }
  }

  const $ = (id) => document.getElementById(id);

  // Icons: Quartz Energy icon set (quartz-frontend, MIT) plus a few simple glyphs.
  const ICONS = {
    clock: ["0 0 16 16", "M8 2.667a5.333 5.333 0 100 10.666A5.333 5.333 0 008 2.667zM1.333 8a6.667 6.667 0 1113.334 0A6.667 6.667 0 011.333 8zM8 4c.368 0 .667.299.667.667V8A.667.667 0 018 8.667H6a.667.667 0 010-1.333h1.333V4.667C7.333 4.299 7.632 4 8 4z"],
    power: ["0 0 32 32", "M15.519 19.027v-.865H6.902l7.752-13.957v8.768c0 .23.09.45.255.61.16.16.385.255.61.255h8.617l-7.752 13.957v-8.768a.875.875 0 00-.865-.865v.865h-.865v12.106a.867.867 0 001.622.424l10.087-18.16a.883.883 0 00-.009-.86.87.87 0 00-.743-.428h-9.223V.867a.866.866 0 00-1.621-.424L4.679 18.603a.864.864 0 00.752 1.288H15.52v-.864h-.865.865z"],
    solar: ["0 0 32 33", "M15.069.714v4.932h1.862V.714h-1.863zM15.069 27.783v4.931h1.862v-4.931h-1.863zM15.945 24.824a8.055 8.055 0 100-16.11 8.055 8.055 0 000 16.11zm0-1.863a6.192 6.192 0 100-12.384 6.192 6.192 0 000 12.384zM27.069 15.783H32v1.863h-4.931v-1.863zM0 15.783h4.932v1.863H0v-1.863zM23.142 8.255l3.487-3.487 1.317 1.317-3.487 3.487-1.317-1.317zM4.002 27.395l3.487-3.487 1.317 1.318-3.487 3.487-1.317-1.318zM27.946 27.343l-3.487-3.487-1.317 1.318 3.487 3.487 1.317-1.318zM8.806 8.203L5.319 4.716 4.002 6.033 7.489 9.52l1.317-1.317z"],
    download: ["0 0 24 24", "m16.75 8.96-4.01 4.01-.707.708-.708-.707-4.01-4.01 1.414-1.415 2.304 2.303V2h2v7.85l2.303-2.304zM1 20.34v-9h6v2H3v5h18v-5h-4v-2h6v9H1"],
    calendar: ["0 0 24 24", "M7 2h2v2h6V2h2v2h3a1 1 0 011 1v15a1 1 0 01-1 1H4a1 1 0 01-1-1V5a1 1 0 011-1h3zm12 8H5v9h14zM5 6v2h14V6z"],
    users: ["0 0 24 24", "M9 11a4 4 0 110-8 4 4 0 010 8zm0-2a2 2 0 100-4 2 2 0 000 4zm-7 12v-2a5 5 0 015-5h4a5 5 0 015 5v2h-2v-2a3 3 0 00-3-3H7a3 3 0 00-3 3v2zm16.5-8a3.5 3.5 0 110-7v2a1.5 1.5 0 100 3zM22 21h-2v-2a3 3 0 00-2-2.83V14.1A5 5 0 0122 19z"],
    flask: ["0 0 24 24", "M9 2h6v2h-1v5.3l5.6 9.7A2 2 0 0117.9 22H6.1a2 2 0 01-1.7-3L10 9.3V4H9zm3 9.2L8.8 17h6.4z"],
    lock: ["0 0 24 24", "M6 10V8a6 6 0 1112 0v2h1a1 1 0 011 1v10a1 1 0 01-1 1H5a1 1 0 01-1-1V11a1 1 0 011-1zm2 0h8V8a4 4 0 00-8 0z"],
  };

  function icon(name, size = 16, fill = "currentColor") {
    const [vb, d] = ICONS[name];
    return s("svg", { width: size, height: size, viewBox: vb, "aria-hidden": "true", focusable: "false" }, s("path", { d, fill, "fill-rule": "evenodd" }));
  }

  function mark(kind) {
    if (kind === "ok") return s("svg", { width: 16, height: 16, viewBox: "0 0 16 16", "aria-hidden": "true" }, s("path", { d: "M3 8.5l3 3 7-7", fill: "none", stroke: C.mint, "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" }));
    return s("svg", { width: 16, height: 16, viewBox: "0 0 16 16", "aria-hidden": "true" }, s("path", { d: "M4 4l8 8M12 4l-8 8", fill: "none", stroke: C.orange, "stroke-width": 2, "stroke-linecap": "round" }));
  }

  // Quartz Spinner (quartz-frontend icons.tsx).
  function spinner(size) {
    return s(
      "svg",
      { class: "spinner", width: size, height: size, viewBox: "0 0 100 101", "aria-hidden": "true" },
      s("path", { fill: "#444", d: "M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" }),
      s("path", { fill: C.yellow, d: "M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" }),
    );
  }

  function info(text) {
    return h(
      "button",
      { type: "button", className: "info", "aria-label": text },
      s("svg", { width: 18, height: 18, viewBox: "0 0 24 24", "aria-hidden": "true" }, s("path", { d: "M12 21a9 9 0 100-18 9 9 0 000 18zM12 8h.01M11 12h1v4h1", fill: "none", stroke: "currentColor", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" })),
      h("span", { className: "info-pop", role: "tooltip", text }),
    );
  }

  // ------------------------------------------------------------------ formatting

  const fmtInt = (n) => (n === null || n === undefined ? "–" : nf.format(n));
  const fmtPct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;

  function fmtElapsed(sec) {
    if (!Number.isFinite(sec)) return "-.--";
    if (sec < 60) return `${sec.toFixed(1)} s`;
    return `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
  }

  function fmtBytes(b) {
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`;
    if (b < 1024 ** 3) return `${(b / 1024 / 1024).toFixed(1)} MB`;
    return `${(b / 1024 ** 3).toFixed(2)} GB`;
  }

  function dateMs(iso) {
    const [y, m, d] = iso.split("-").map(Number);
    return Date.UTC(y, m - 1, d);
  }

  function fmtDate(iso) {
    if (!iso) return "–";
    const d = new Date(dateMs(iso));
    return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
  }

  function fmtDay(ms) {
    const d = new Date(ms);
    return `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
  }

  function fmtHour(hh) {
    const mins = Math.round(hh * 60);
    return `${String(Math.floor(mins / 60) % 24).padStart(2, "0")}:${String(mins % 60).padStart(2, "0")}`;
  }

  function fmtVal(v, unit) {
    if (v === null || v === undefined || !Number.isFinite(v)) return "No data";
    const a = Math.abs(v);
    return `${v.toFixed(a >= 100 ? 0 : a >= 10 ? 1 : 2)} ${unit}`;
  }

  // ------------------------------------------------------------------ API

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
      res = await fetch(path, { method: opts.method || "GET", headers: { "X-PVD-Token": TOKEN }, cache: "no-store", credentials: "same-origin", redirect: "error" });
    } catch (_) {
      throw new ApiError(0, { title: "Local service unreachable", detail: "The PV Detect process is not answering.", action: "Restart gui.py, then reload this page." });
    }
    if (opts.raw) {
      if (!res.ok) throw new ApiError(res.status, ((await res.json().catch(() => null)) || {}).error);
      return res;
    }
    const body = await res.json().catch(() => null);
    if (!res.ok) throw new ApiError(res.status, body && body.error);
    return body;
  }

  function handleApiError(e, context) {
    if (!(e instanceof ApiError)) throw e;
    if (e.status === 401 || e.status === 403) {
      state.fatal = { title: "This page belongs to an earlier session", detail: "The local service was restarted.", action: "Reload the page from the address printed in the terminal." };
      render();
      return;
    }
    toast(`${context}: ${e.err.title}`);
  }

  // ------------------------------------------------------------------ jobs

  const isActive = (job) => !!job && (job.state === "running" || job.state === "receiving");

  async function refreshSession() {
    try {
      state.session = await api("/api/session");
      applyJob(state.session.job, true);
    } catch (e) {
      handleApiError(e, "Could not read the session");
    }
    renderHeader();
  }

  function applyJob(job, force = false) {
    const prev = state.job;
    state.job = job || null;
    state.jobSyncedAt = performance.now();
    if (!job || (state.resultFor && state.resultFor !== job.id)) resetResults();
    if (isActive(job)) {
      schedulePoll();
      if (!tickTimer) tickTimer = setInterval(updateElapsed, 100);
    } else {
      clearInterval(tickTimer);
      tickTimer = null;
    }
    if (job && job.state === "complete" && state.resultFor !== job.id) {
      loadResult(job.id);
      return;
    }
    const sig = (j) => (j ? `${j.id}|${j.state}|${j.stages.map((x) => x.status).join("")}` : "");
    if (force || sig(prev) !== sig(job)) render();
    else updateElapsed();
  }

  function schedulePoll() {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      if (state.upload) return;
      try {
        applyJob((await api("/api/jobs/current")).job);
      } catch (e) {
        handleApiError(e, "Status update failed");
        schedulePoll();
      }
    }, POLL_MS);
  }

  function currentElapsed() {
    const job = state.job;
    if (!job) return NaN;
    return (job.elapsed_s || 0) + (isActive(job) ? (performance.now() - state.jobSyncedAt) / 1000 : 0);
  }

  function updateElapsed() {
    const el = $("elapsed");
    if (el) el.textContent = fmtElapsed(currentElapsed());
  }

  function resetResults() {
    state.result = null;
    state.resultFor = null;
    state.selected = null;
    state.diag = null;
    state.diagCache = new Map();
    state.query = "";
    state.filter = "all";
    $("search").value = "";
    $("sel-filter").value = "all";
  }

  async function loadResult(jobId) {
    try {
      const result = await api(`/api/jobs/${encodeURIComponent(jobId)}/result`);
      if (!state.job || state.job.id !== jobId) return;
      state.result = result;
      state.resultFor = jobId;
      render();
      const first = result.customers.find((c) => c.pv_installed) || result.customers[0];
      if (first) selectCustomer(first.customer_id);
      api("/api/session").then((sess) => { state.session = sess; renderHeader(); }).catch(() => {});
    } catch (e) {
      handleApiError(e, "Could not load results");
    }
  }

  async function startSynthetic() {
    if (state.busy || isActive(state.job)) return;
    state.busy = true;
    renderHeader();
    try {
      resetResults();
      applyJob((await api("/api/jobs/synthetic", { method: "POST" })).job);
    } catch (e) {
      handleApiError(e, "Could not start the synthetic check");
    } finally {
      state.busy = false;
      renderHeader();
    }
  }

  function startAnalysis(file) {
    if (!file || state.busy || isActive(state.job)) return;
    resetResults();
    state.busy = true;
    state.upload = { loaded: 0, total: file.size };
    applyJob({ id: null, kind: "analysis", state: "receiving", elapsed_s: 0, input_name: file.name, error: null, stages: ANALYSIS_STAGES.map(([key, label], i) => ({ key, label, status: i === 0 ? "active" : "pending" })) });
    clearTimeout(pollTimer);

    const xhr = new XMLHttpRequest();
    state.xhr = xhr;
    xhr.open("POST", "/api/jobs/analysis");
    xhr.setRequestHeader("X-PVD-Token", TOKEN);
    xhr.setRequestHeader("Content-Type", "text/csv");
    xhr.setRequestHeader("X-PVD-Filename", encodeURIComponent(file.name));
    xhr.upload.addEventListener("progress", (ev) => {
      if (!state.upload) return;
      state.upload.loaded = ev.loaded;
      updateUpload();
    });
    const done = () => {
      state.upload = null;
      state.xhr = null;
      state.busy = false;
      $("file-input").value = "";
    };
    const failActive = (err) => ({ ...state.job, state: "failed", error: err, stages: state.job.stages.map((x) => (x.status === "active" ? { ...x, status: "failed" } : x)) });
    xhr.addEventListener("load", () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch (_) { body = null; }
      done();
      if (xhr.status >= 200 && xhr.status < 300 && body && body.job) return applyJob(body.job);
      const err = (body && body.error) || { title: "Upload failed", detail: `The local service answered with status ${xhr.status}.`, action: "Check the file and try again." };
      if (xhr.status === 401 || xhr.status === 403) return handleApiError(new ApiError(xhr.status, err), "Upload failed");
      applyJob(failActive(err));
    });
    xhr.addEventListener("error", () => {
      done();
      applyJob(failActive({ title: "Upload interrupted", detail: "The connection to the local service closed while sending the file.", action: "Check that gui.py is still running and try again." }));
    });
    xhr.addEventListener("abort", () => {
      done();
      applyJob({ ...state.job, state: "cancelled", error: null });
    });
    xhr.send(file);
  }

  async function cancelJob() {
    if (state.xhr) state.xhr.abort();
    try {
      applyJob((await api("/api/jobs/current/cancel", { method: "POST" })).job);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) return refreshSession();
      handleApiError(e, "Cancel failed");
    }
  }

  async function clearSession() {
    if (state.xhr) state.xhr.abort();
    try {
      await api("/api/session/clear", { method: "POST" });
      resetResults();
      applyJob(null, true);
      toast("Session data cleared. Original files and the configured model were not touched.");
      refreshSession();
    } catch (e) {
      handleApiError(e, "Clear failed");
    }
  }

  async function download(name) {
    if (!state.result) return;
    closeMenu();
    try {
      const res = await api(`/api/jobs/${encodeURIComponent(state.result.job_id)}/files/${encodeURIComponent(name)}`, { raw: true });
      const url = URL.createObjectURL(await res.blob());
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

  // ------------------------------------------------------------------ render

  function render() {
    renderHeader();
    renderSide();
    renderStatus();
    renderOverlay();
    renderTable();
    renderCustomer();
  }

  function renderHeader() {
    const model = state.session && state.session.model;
    const active = isActive(state.job) || state.busy || !!state.fatal;
    $("model-dot").className = `dot ${model ? (model.available ? "ok" : "off") : ""}`;
    $("model-label").textContent = model ? (model.available ? model.label || "ready" : "none configured") : "checking…";
    $("btn-synthetic").disabled = active;
    const up = $("btn-upload");
    up.disabled = active || !(model && model.available);
    up.title = model && !model.available ? "No trained model configured. Launch with --model PATH. The synthetic check still works." : "Analyze a local CSV with the configured model";
    $("btn-download").disabled = !state.result;
    const kind = state.job && state.job.kind;
    $("tab-demo").setAttribute("aria-current", String(kind === "synthetic"));
    $("tab-private").setAttribute("aria-current", String(kind === "analysis"));
  }

  function renderStatus() {
    const job = state.job;
    let pill;
    if (state.fatal) pill = h("span", { className: "pill err", text: "Session expired" });
    else if (!job) pill = h("span", { className: "pill", text: "No run yet" });
    else if (isActive(job)) {
      const st = job.stages.find((x) => x.status === "active");
      pill = h("span", { className: "pill" }, `${st ? st.label : "Running"}`, spinner(14));
    } else if (job.state === "complete") pill = h("span", { className: "pill", text: state.result ? "Data loaded" : "Loading results" });
    else if (job.state === "failed") pill = h("span", { className: "pill err", text: "Run failed" });
    else pill = h("span", { className: "pill", text: "Cancelled" });
    $("status").replaceChildren(pill);
    const usable = !!(state.diag && state.diag.usable);
    $("sel-metric").disabled = !usable;
    $("sel-res").disabled = !usable;
    const resDaily = $("sel-res").querySelector('option[value="daily"]');
    resDaily.disabled = !!(state.diag && state.diag.series.resolution === "weekly");
  }

  // ---- sidebar (Quartz WideCard layout)

  function wcard(label, iconEl, left, right, extra) {
    return h(
      "div",
      { className: "wcard" },
      h("div", { className: "wcard-top" }, h("span"), h("span", { className: "wcard-label" }, iconEl, label)),
      h("div", { className: "wcard-row" }, left, right),
      extra,
    );
  }

  function fig(value, cap, color, right = false, unit = null) {
    return h("div", { className: `fig ${right ? "right" : ""}` }, h("span", { className: `fig-val num ${color}` }, value, unit ? h("small", { text: unit }) : null), h("span", { className: "fig-cap", text: cap }));
  }

  // Aggregate error analysis saved by `pvdetect.diagnose` next to the configured model (out-of-fold when
  // produced by scripts/train_and_launch.sh). Shown before any run: it describes the model, not an input.
  function evaluationCards(ev) {
    const n = (v, d = 3) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
    const pct = (v) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`);
    const c = ev.counts || {};
    const lost = ev.lost || {};
    const dt = ev.dates || {};
    const cap = ev.capacity || {};
    const label = ev.out_of_fold ? "Cross-validated (out-of-fold)" : "In-sample (optimistic)";
    const lostRow = h(
      "p",
      { className: "metric-line num" },
      "Points lost · detection ",
      h("b", { text: n(lost.detection, 1) }),
      " · dates ",
      h("b", { text: n(lost.dates, 1) }),
      " · capacity ",
      h("b", { text: n(lost.capacity, 1) }),
    );
    const sizes = Object.entries(ev.by_size || {});
    return [
      h("hr"),
      h(
        "div",
        { className: "wcard" },
        h("div", { className: "wcard-top" }, info(`${label} evaluation of this model on its labelled customers, saved by pvdetect.diagnose (${ev.source}). Each customer is scored by a model that never saw it when out-of-fold.`), h("span", { className: "wcard-label" }, icon("flask", 30), "Model score")),
        h("div", { className: "wcard-row" }, fig(n(ev.score, 1), "score / 100", "c-mint"), fig(n(ev.f1), "F1", "c-mint", true)),
        lostRow,
        h("span", { className: `tag-demo ${ev.stale ? "c-orange" : ""}`, text: ev.stale ? "Stale: model is newer than this evaluation" : label }),
      ),
      h("hr"),
      wcard("Detection", icon("solar", 32), fig(n(ev.precision, 2), `precision · ${c.FP ?? "–"} false alarms`, "c-yellow"), fig(n(ev.recall, 2), `recall · ${c.FN ?? "–"} missed`, "c-yellow", true)),
      h("hr"),
      wcard(
        "Dates",
        icon("calendar", 30),
        fig(n(ev.D, 2), `D · median ${n(dt.median_abs, 0)} d off`, "c-blue"),
        fig(pct(dt.within && dt.within["15"]), "within 15 days", "c-blue", true),
        h("p", { className: "metric-line num" }, `Median signed error ${n(dt.median_signed, 0)} d · late share ${pct(dt.late_share)}`),
      ),
      h("hr"),
      wcard(
        "Capacity",
        icon("power", 32),
        fig(n(ev.P, 2), `P · median ${pct(cap.median_ape)} off`, "c-yellow-100"),
        fig(n(cap.median_ratio, 2), "predicted / true", "c-yellow-100", true),
        h("p", { className: "metric-line num" }, `Within 25%: ${pct(cap.within && cap.within["25"])} · underestimated: ${pct(cap.under_share)} · at 500 kWp clip: ${cap.at_ceiling ?? "–"}`),
      ),
      h(
        "details",
        { className: "fold", open: true },
        h("summary", null, "Where it fails", h("span", { className: "fold-meta c-orange", text: ev.largest ? `fix ${ev.largest}` : "" })),
        h(
          "div",
          { className: "fold-body" },
          (ev.failure_modes || []).map((m) => h("div", { className: "check" }, mark("bad"), h("div", { text: m }))),
          sizes.length
            ? h(
                "dl",
                { className: "kv num" },
                sizes.map(([bin, v]) => [h("dt", { text: `${bin} kWp · n ${v.n}` }), h("dd", { text: `missed ${v.missed} · D ${n(v.D, 2)} · P ${n(v.P, 2)}` })]),
              )
            : null,
        ),
      ),
    ];
  }

  function renderSide() {
    const side = $("side");
    const job = state.job;
    const r = state.result;
    const title = !job ? "No run yet" : job.kind === "synthetic" ? "Synthetic check" : job.input_name || "Local CSV";
    const today = new Date();
    const nodes = [
      h("div", { className: "side-head" }, h("span", { className: "side-title", text: title, title }), h("span", { className: "side-date", text: `${DAYS[today.getDay()]}, ${today.getDate()} ${MONTHS[today.getMonth()]} ${today.getFullYear()}` })),
      h("hr"),
      h(
        "div",
        { className: "times" },
        h("div", { className: "tl" }, h("span", { className: "tl-tag", text: "Elapsed" }), h("span", { className: "tl-val num" }, icon("clock", 16), h("span", { id: "elapsed", text: fmtElapsed(currentElapsed()) }))),
        h("div", { className: "tl right" }, h("span", { className: "tl-tag", text: "Model inference" }), h("span", { className: "tl-val num" }, icon("clock", 16), r ? fmtElapsed(r.summary.inference_s) : "-.--")),
      ),
    ];
    const ev = state.session && state.session.evaluation;
    if (ev) nodes.push(...evaluationCards(ev));

    const pv = r ? r.customers.filter((c) => c.pv_installed) : [];
    const caps = pv.map((c) => c.capacity_kwp).filter((v) => v !== null).sort((a, b) => a - b);
    const median = caps.length ? (caps.length % 2 ? caps[(caps.length - 1) / 2] : (caps[caps.length / 2 - 1] + caps[caps.length / 2]) / 2) : null;
    const total = caps.reduce((a, b) => a + b, 0);
    const reg = { event: 0, present_at_start: 0, insufficient: 0 };
    for (const c of pv) reg[c.regime] = (reg[c.regime] || 0) + 1;
    const dash = "-.--";
    const sm = r && r.summary;

    nodes.push(
      wcard("Customers", icon("users", 32), fig(sm ? fmtInt(sm.customers) : dash, "processed", sm ? "c-mint" : "c-muted"), fig(sm ? fmtInt(sm.quality.customers_without_usable_history) : dash, "no usable history", sm ? "c-mint" : "c-muted", true)),
      h("hr"),
      wcard("PV detected", icon("solar", 32), fig(sm ? fmtInt(sm.pv) : dash, "customers", sm ? "c-yellow" : "c-muted"), fig(sm ? (sm.customers ? fmtPct(sm.pv / sm.customers) : "0%") : dash, `of ${sm ? fmtInt(sm.customers) : "all"}`, sm ? "c-yellow" : "c-muted", true)),
      h("hr"),
      wcard("Capacity", icon("power", 32), fig(median !== null ? median.toFixed(1) : dash, "median", median !== null ? "c-yellow-100" : "c-muted", false, median !== null ? "kWp" : null), fig(caps.length ? total.toFixed(1) : dash, "total", caps.length ? "c-yellow-100" : "c-muted", true, caps.length ? "kWp" : null)),
      h("hr"),
      wcard(
        "Install date",
        icon("calendar", 30),
        fig(r ? fmtInt(reg.event) : dash, "change detected", r ? "c-blue" : "c-muted"),
        fig(r ? fmtInt(reg.present_at_start) : dash, "present from start", r ? "c-blue" : "c-muted", true),
        r ? h("p", { className: "metric-line" }, h("b", { text: fmtInt(reg.insufficient) }), " weak evidence (date from the model's fallback rule)") : null,
      ),
    );

    if (r && r.synthetic) {
      const m = r.synthetic.metrics;
      nodes.push(
        h("hr"),
        h(
          "div",
          { className: "wcard" },
          h("div", { className: "wcard-top" }, info(`Fabricated customers from the development generator (train seed ${r.synthetic.seeds.train}, ${r.synthetic.train_customers} customers; test seed ${r.synthetic.seeds.test}, ${r.synthetic.test_customers} customers). This guards against regressions only; it is not evidence of real-world accuracy.`), h("span", { className: "wcard-label" }, icon("flask", 30), "Synthetic score")),
          h("div", { className: "wcard-row" }, fig(m.score.toFixed(1), "full score", "c-mint"), fig(m.f1.toFixed(3), "F1", "c-mint", true)),
          h("p", { className: "metric-line num" }, "Date D ", h("b", { text: m.D.toFixed(3) }), " · Capacity P ", h("b", { text: m.P.toFixed(3) }), ` · TP ${m.tp} · FP ${m.fp} · FN ${m.fn}`),
          h("span", { className: "tag-demo", text: "Regression demo · not real-world accuracy" }),
        ),
      );
    }

    if (r) {
      const passed = r.checks.filter((c) => c.passed).length;
      nodes.push(
        h(
          "details",
          { className: "fold" },
          h("summary", null, "Output checks", h("span", { className: `fold-meta num ${passed === r.checks.length ? "c-mint" : "c-orange"}`, text: `${passed}/${r.checks.length}` })),
          h("div", { className: "fold-body" }, r.checks.map((c) => h("div", { className: "check" }, mark(c.passed ? "ok" : "bad"), h("div", null, c.label, c.detail ? h("small", { text: c.detail }) : null)))),
        ),
        h(
          "details",
          { className: "fold", open: true },
          h("summary", null, "Output files"),
          h(
            "div",
            { className: "fold-body" },
            [["predictions.csv", r.outputs.predictions], ["pv_extensions_predictions.csv", r.outputs.extensions]].map(([name, o]) =>
              h(
                "div",
                { className: "file-row" },
                h("div", null, h("code", { text: name }), h("small", { text: `${fmtInt(o.rows)} ${o.rows === 1 ? "row" : "rows"} · ${o.valid ? "valid" : "invalid"}` })),
                h("button", { type: "button", className: "icon-btn", "aria-label": `Download ${name}`, title: `Download ${name}`, onclick: () => download(name) }, icon("download", 20)),
              ),
            ),
            r.outputs.extensions.note ? h("p", { className: "metric-line", text: r.outputs.extensions.note }) : null,
          ),
        ),
      );
      const q = sm.quality;
      const dl = q.day_length_counts || {};
      nodes.push(
        h(
          "details",
          { className: "fold" },
          h("summary", null, "Input quality"),
          h(
            "dl",
            { className: "kv num fold-body" },
            [
              ["Date range", sm.date_range.start ? `${fmtDate(sm.date_range.start)} – ${fmtDate(sm.date_range.end)}` : "–"],
              ["Rows read", fmtInt(q.rows_total)],
              ["Duplicate timestamps dropped", fmtInt(q.rows_duplicate)],
              ["Invalid timestamp / energy", `${fmtInt(q.rows_bad_epoch)} / ${fmtInt(q.rows_bad_energy)}`],
              ["Customer-days with 96 slots", fmtInt(dl["96"])],
              ["DST days (92 / 100 slots)", `${fmtInt(dl["92"])} / ${fmtInt(dl["100"])}`],
              ["Partial customer-days", fmtInt(dl.other)],
              ["Customers with export readings", fmtInt(q.customers_signed)],
            ].map(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })]),
          ),
        ),
      );
    } else if (state.session) {
      const inp = state.session.input;
      nodes.push(
        h(
          "details",
          { className: "fold", open: !job },
          h("summary", null, "Input format"),
          h(
            "div",
            { className: "fold-body" },
            h("div", { className: "cols" }, inp.required_columns.map((c) => h("code", { text: c }))),
            h("dl", { className: "kv" }, [["Separator", inp.separators], ["Decimal", inp.decimal], ["Timestamps", inp.timestamps]].map(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })])),
          ),
        ),
      );
    }

    nodes.push(
      h(
        "div",
        { className: "side-foot" },
        h("span", null, icon("lock", 12), "127.0.0.1 · offline · no telemetry"),
        h("button", { type: "button", className: "btn btn-ghost btn-sm", onclick: clearSession, title: "Delete uploads, outputs and the demo model from this session" }, "Clear session"),
      ),
    );
    side.replaceChildren(...nodes);
  }

  // ---- overlay over the chart canvas

  function renderOverlay() {
    const ov = $("overlay");
    const job = state.job;
    const actions = () =>
      h(
        "div",
        { className: "overlay-actions" },
        h("button", { type: "button", className: "btn btn-yellow", onclick: startSynthetic }, "Run synthetic check"),
        h("button", { type: "button", className: "btn btn-ghost", onclick: () => $("file-input").click(), disabled: !(state.session && state.session.model.available) }, "Analyze CSV"),
      );
    let content = null;
    if (state.fatal) {
      content = [errbox(state.fatal)];
    } else if (!job) {
      content = [
        h("p", { className: "overlay-title", text: "Detect rooftop PV from smart-meter data" }),
        h("p", { className: "overlay-sub", text: "Find which customers installed solar, when, and how large, from 15-minute grid withdrawal. Start with the synthetic check, or analyze your own CSV with the trained model." }),
        actions(),
      ];
    } else if (isActive(job)) {
      content = [
        spinner(40),
        h(
          "ol",
          { className: "stages", "aria-label": "Job stages" },
          job.stages.map((st) =>
            h("li", { className: "stage", "data-s": st.status }, h("span", { className: "stage-mark" }, st.status === "done" ? mark("ok") : st.status === "failed" ? mark("bad") : st.status === "active" ? spinner(16) : h("i")), st.label),
          ),
        ),
        job.state === "receiving" && state.upload ? h("div", null, h("div", { className: "progress", role: "progressbar", "aria-label": "Upload", id: "upload-bar" }, h("i", { id: "upload-fill" })), h("p", { className: "overlay-sub num", id: "upload-text" })) : null,
        h("button", { type: "button", className: "btn btn-ghost btn-sm", onclick: cancelJob }, "Cancel"),
      ];
    } else if (job.state === "failed") {
      content = [errbox(job.error || { title: "The run failed", detail: "No outputs were made downloadable.", action: "" }), actions()];
    } else if (job.state === "cancelled") {
      content = [h("p", { className: "overlay-title", text: "Cancelled" }), h("p", { className: "overlay-sub", text: "The worker was stopped and its files deleted. Nothing from this run can be downloaded." }), actions()];
    } else if (!state.result) {
      content = [spinner(40)];
    }
    ov.hidden = !content;
    ov.replaceChildren(...(content || []).filter(Boolean));
    updateUpload();
  }

  function errbox(err) {
    return h("div", { className: "errbox", role: "alert" }, h("b", { text: err.title }), h("p", { text: err.detail }), err.action ? h("p", { text: err.action }) : null);
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

  // ---- customer table (nowcasting solar-site table style)

  function filtered() {
    const q = state.query.trim().toLowerCase();
    return state.result.customers.filter((c) => (state.filter === "pv" ? c.pv_installed : state.filter === "nopv" ? !c.pv_installed : true) && (!q || c.customer_id.toLowerCase().includes(q)));
  }

  function renderTable() {
    const host = $("table-scroll");
    const r = state.result;
    if (!r) {
      host.replaceChildren(h("div", { className: "panel-empty", text: isActive(state.job) ? "Customers appear when the run completes." : "No customers yet." }));
      $("count").textContent = "";
      return;
    }
    const synthetic = r.kind === "synthetic";
    const rows = filtered();
    const shown = rows.slice(0, 500);
    const maxCap = Math.max(1, ...r.customers.map((c) => c.capacity_kwp || 0));
    $("count").textContent = rows.length > shown.length ? `first ${shown.length} of ${fmtInt(rows.length)}` : `${fmtInt(rows.length)} customers`;
    const body = h("tbody", { onkeydown: onTableKey });
    if (!shown.length) body.appendChild(h("tr", { className: "empty-row" }, h("td", { colspan: synthetic ? 6 : 5, text: "No customers match." })));
    for (const c of shown) {
      const bar = h("span", { className: "cap-bar" });
      bar.style.width = `${c.capacity_kwp ? (c.capacity_kwp / maxCap) * 100 : 0}%`;
      body.appendChild(
        h(
          "tr",
          { "aria-selected": String(c.customer_id === state.selected), "data-cid": c.customer_id, onclick: () => selectCustomer(c.customer_id) },
          h("td", null, h("button", { type: "button", className: "id-btn", text: c.customer_id, "aria-label": `Inspect customer ${c.customer_id}` }), bar),
          h("td", null, h("span", { className: c.pv_installed ? "yes" : "no", text: c.pv_installed ? "Yes" : "No" })),
          h("td", { className: "num" }, c.installation_date || h("span", { className: "dash", text: "–" })),
          h("td", { className: "r num" }, c.capacity_kwp !== null ? [c.capacity_kwp.toFixed(2), h("span", { className: "unit", text: "kWp" })] : h("span", { className: "dash", text: "–" })),
          h("td", null, c.pv_installed ? REGIME[c.regime] : h("span", { className: "dash", text: "–" })),
          synthetic ? h("td", null, truthCell(c)) : null,
        ),
      );
    }
    host.replaceChildren(
      h(
        "table",
        { className: "sites" },
        h("caption", { className: "sr-only", text: "Predictions per customer" }),
        h("thead", null, h("tr", null, h("th", { scope: "col", text: "Customer" }), h("th", { scope: "col", text: "PV" }), h("th", { scope: "col", text: "Install date" }), h("th", { scope: "col", className: "r", text: "Capacity" }), h("th", { scope: "col", text: "Date regime" }), synthetic ? h("th", { scope: "col", text: "Known" }) : null)),
        body,
      ),
    );
  }

  function truthCell(c) {
    const t = c.truth;
    if (!t) return h("span", { className: "dash", text: "–" });
    const agree = t.pv_installed === c.pv_installed;
    return h("span", { className: agree ? "ok" : "bad", text: `${t.pv_installed ? "PV" : "No PV"} ${agree ? "✓" : "✗"}` });
  }

  function onTableKey(ev) {
    if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp") return;
    const tr = ev.target.closest("tr[data-cid]");
    const next = tr && (ev.key === "ArrowDown" ? tr.nextElementSibling : tr.previousElementSibling);
    if (!next || !next.dataset.cid) return;
    ev.preventDefault();
    selectCustomer(next.dataset.cid);
    next.querySelector(".id-btn").focus();
    next.scrollIntoView({ block: "nearest" });
  }

  // ---- selected customer

  async function selectCustomer(cid) {
    if (!state.result) return;
    state.selected = cid;
    for (const tr of document.querySelectorAll("tr[data-cid]")) tr.setAttribute("aria-selected", String(tr.dataset.cid === cid));
    const jobId = state.result.job_id;
    let diag = state.diagCache.get(cid);
    if (!diag) {
      try {
        diag = await api(`/api/jobs/${encodeURIComponent(jobId)}/customers/${encodeURIComponent(cid)}/diagnostics`);
        state.diagCache.set(cid, diag);
      } catch (e) {
        return handleApiError(e, "Could not load the customer view");
      }
    }
    if (!state.result || state.result.job_id !== jobId || state.selected !== cid) return;
    state.diag = diag;
    renderStatus();
    renderCustomer();
  }

  function renderCustomer() {
    const d = state.result && state.diag && state.diag.customer_id === state.selected ? state.diag : null;
    renderCustBar(d);
    drawMain(d);
    drawProfile(d);
  }

  function renderCustBar(d) {
    const bar = $("cust-bar");
    if (!d) {
      bar.replaceChildren(h("div", null, h("div", { className: "cust-id", text: state.result ? "Select a customer" : "Customer" }), h("div", { className: "cust-sub", text: state.result ? "Pick a row in the table below." : "Results appear here after a run." })));
      return;
    }
    const p = d.prediction;
    const t = d.truth;
    const cf = (cap, ...val) => h("div", { className: "cf" }, h("span", { className: "cf-val num" }, ...val), h("span", { className: "cf-cap", text: cap }));
    const knownCap = t && t.capacity_kwp !== null && t.capacity_kwp !== undefined ? Number(t.capacity_kwp) : null;
    const figs = d.usable
      ? [
          cf(t ? `Install date · known ${t.installation_date ? fmtDate(t.installation_date) : "none"}` : "Install date", h("span", { className: p.installation_date ? "c-yellow" : "dash", text: p.installation_date ? fmtDate(p.installation_date) : "–" })),
          cf(
            t ? "Capacity · predicted / known" : "Capacity",
            h("span", { className: p.capacity_kwp !== null ? "c-yellow" : "dash", text: p.capacity_kwp !== null ? p.capacity_kwp.toFixed(2) : "–" }),
            t ? h("span", { className: "slash", text: ` / ${knownCap !== null ? knownCap.toFixed(2) : "–"}` }) : null,
            h("small", { text: "kWp" }),
          ),
          cf("Days with data", fmtInt(d.coverage.days_with_data)),
          cf("Missing slots", fmtPct(d.coverage.missing_slot_share, 1)),
        ]
      : [h("span", { className: "unavailable" }, h("span", { text: "History" }), h("span", { text: "unavailable" }))];
    const sub = p.pv_installed ? `PV detected · ${d.regime.label}` : "No PV detected";
    bar.replaceChildren(
      h("div", { style: null }, h("div", { className: "cust-id num", text: d.customer_id, title: d.customer_id }), h("div", { className: "cust-sub" }, sub, t ? ` · known: ${t.pv_installed ? "PV" : "no PV"}` : "", p.pv_installed ? info(`${d.regime.label}: ${d.regime.explanation}`) : null)),
      h("div", { className: "cust-figs" }, figs),
    );
  }

  function drawMain(d) {
    const host = $("chart");
    if (!d || !d.usable) {
      lineChart(host, { xs: [], series: [], xKind: "date", yLabel: "Withdrawal (kWh/day)", markers: [] });
      return;
    }
    const weekly = state.resolution === "weekly" || d.series.resolution === "weekly";
    const src = weekly ? d.weekly : d.series;
    const ys = state.metric === "total" ? src.total_kwh : src.midday_kw;
    const unit = state.metric === "total" ? "kWh/day" : "kW";
    const xs = src.dates.map(dateMs);
    const cut = d.markers.predicted_date ? dateMs(d.markers.predicted_date) : Infinity;
    const before = ys.map((v, i) => (xs[i] <= cut || (i > 0 && xs[i - 1] <= cut) ? v : null));
    const after = ys.map((v, i) => (xs[i] >= cut || (i + 1 < xs.length && xs[i + 1] >= cut) ? v : null));
    const markers = [];
    if (d.markers.predicted_date) markers.push({ x: cut, cls: "ref", label: `Install ${fmtDate(d.markers.predicted_date)}` });
    if (d.markers.known_date) markers.push({ x: dateMs(d.markers.known_date), cls: "ref known", label: `Known ${fmtDate(d.markers.known_date)}` });
    lineChart(host, {
      xs,
      series: [
        { ys: before, color: C.blue, fill: 0.5, width: 1 },
        { ys: after, color: C.yellow, fill: 0.8, width: 1 },
      ],
      xKind: "date",
      yLabel: state.metric === "total" ? "Withdrawal (kWh/day)" : "Midday power 10–15 h (kW)",
      markers,
      ariaLabel: `${state.metric === "total" ? "Daily withdrawal" : "Midday power"} for customer ${d.customer_id}`,
      tip: (i) => ({
        title: `${weekly ? "Week of " : ""}${fmtDay(xs[i])}`,
        rows: [[state.metric === "total" ? "Withdrawal" : "Midday power", fmtVal(ys[i], unit), xs[i] >= cut ? C.yellow : C.blue]],
      }),
    });
  }

  function drawProfile(d) {
    const host = $("profile");
    const note = $("profile-note");
    if (!d || !d.usable || !d.profile.available) {
      host.replaceChildren(h("div", { className: "panel-empty", text: !d ? "Select a customer to compare a typical day before and after the estimated installation." : !d.usable ? "No usable history for this customer." : d.profile.reason || "Not enough coverage on both sides of the date." }));
      note.textContent = "";
      return;
    }
    const pr = d.profile;
    const xs = Array.from({ length: 96 }, (_, i) => i * 0.25 + 0.125);
    lineChart(host, {
      xs,
      series: [
        { ys: pr.before.kw, color: C.blue100, width: 3 },
        { ys: pr.after.kw, color: C.yellow, width: 3 },
      ],
      xKind: "hour",
      xDomain: [0, 24],
      yLabel: "kW",
      markers: [],
      compact: true,
      ariaLabel: `Typical day before and after the estimated installation for customer ${d.customer_id}`,
      tip: (i) => ({
        title: `${fmtHour(i * 0.25)}–${fmtHour(i * 0.25 + 0.25)}`,
        rows: [["Before", fmtVal(pr.before.kw[i], "kW"), C.blue100], ["After", fmtVal(pr.after.kw[i], "kW"), C.yellow]],
      }),
    });
    note.textContent = `Observed withdrawal, not recovered generation; windows differ in season. Before ${fmtDate(pr.before.start)} – ${fmtDate(pr.before.end)} (${pr.before.days} d) · after ${fmtDate(pr.after.start)} – ${fmtDate(pr.after.end)} (${pr.after.days} d).`;
  }

  // ------------------------------------------------------------------ charts (Quartz/Recharts look)

  function niceStep(span, count) {
    const raw = span / Math.max(1, count);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const n = raw / mag;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * mag;
  }

  function fmtTick(v, step) {
    const mag = Math.floor(Math.log10(step));
    const frac = (step / 10 ** mag) % 1 !== 0;
    return v.toFixed(Math.min(3, Math.max(0, -mag) + (frac ? 1 : 0)));
  }

  function dateTicks(x0, x1, maxTicks) {
    const a = new Date(x0);
    const b = new Date(x1);
    const months = (b.getUTCFullYear() - a.getUTCFullYear()) * 12 + b.getUTCMonth() - a.getUTCMonth() + 1;
    const step = [1, 2, 3, 6, 12].find((st) => months / st <= maxTicks) || 12;
    const ticks = [];
    let m = a.getUTCFullYear() * 12 + Math.ceil(a.getUTCMonth() / step) * step;
    for (let guard = 0; guard < 400; guard++, m += step) {
      const t = Date.UTC(Math.floor(m / 12), m % 12, 1);
      if (t > x1) break;
      if (t >= x0) ticks.push(t);
    }
    return ticks;
  }

  function lineChart(host, o) {
    const W = Math.max(280, Math.round(host.clientWidth || 600));
    const H = Math.max(160, Math.round(host.clientHeight || 300));
    const m = o.compact ? { l: 52, r: 18, t: 14, b: 30 } : { l: 70, r: 30, t: 52, b: 48 };
    const pw = W - m.l - m.r;
    const ph = H - m.t - m.b;
    const xs = o.xs;
    const finite = o.series.flatMap((se) => se.ys.filter((v) => v !== null && Number.isFinite(v)));
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.ariaLabel || "Empty chart" });

    const x0 = o.xDomain ? o.xDomain[0] : xs.length ? xs[0] : Date.UTC(2022, 0, 1);
    const x1 = o.xDomain ? o.xDomain[1] : xs.length ? Math.max(xs[xs.length - 1], x0 + 1) : Date.UTC(2025, 11, 31);
    let lo = finite.length ? Math.min(0, ...finite) : 0;
    let hi = finite.length ? Math.max(...finite) : 40;
    if (hi === lo) hi = lo + 1;
    const step = niceStep(hi - lo, Math.max(3, Math.floor(ph / 60)));
    lo = Math.floor(lo / step) * step;
    hi = Math.ceil((hi + (hi - lo) * 0.04) / step) * step;
    const X = (v) => m.l + ((v - x0) / (x1 - x0)) * pw;
    const Y = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;

    const xt = o.xKind === "hour" ? [0, 3, 6, 9, 12, 15, 18, 21, 24].filter((hh) => pw > 380 || hh % 6 === 0) : dateTicks(x0, x1, Math.max(3, Math.floor(pw / 70)));

    // Quartz CartesianGrid: alternating vertical fills (ocf-grey 900/800 at 0.5 opacity).
    const edges = [x0, ...xt.filter((t) => t > x0 && t < x1), x1];
    for (let i = 0; i < edges.length - 1; i++) {
      svg.appendChild(s("rect", { x: X(edges[i]), y: m.t, width: Math.max(0, X(edges[i + 1]) - X(edges[i])), height: ph, fill: i % 2 ? C.stripeB : C.stripeA, "fill-opacity": 0.5 }));
    }
    const ax = s("g", { class: "ax" });
    for (let v = lo; v <= hi + step * 1e-9; v += step) {
      const vv = Math.abs(v) < step * 1e-9 ? 0 : v;
      svg.appendChild(s("line", { class: "gridline", x1: m.l, x2: m.l + pw, y1: Y(vv), y2: Y(vv) }));
      ax.appendChild(s("text", { x: m.l - 8, y: Y(vv) + 4, "text-anchor": "end", text: fmtTick(vv, step) }));
    }
    let lastYear = null;
    for (const t of xt) {
      svg.appendChild(s("line", { class: "gridline", x1: X(t), x2: X(t), y1: m.t, y2: m.t + ph }));
      if (o.xKind === "hour") {
        ax.appendChild(s("text", { x: X(t), y: m.t + ph + 18, "text-anchor": "middle", text: fmtHour(t) }));
      } else {
        const d = new Date(t);
        ax.appendChild(s("text", { x: X(t), y: m.t + ph + 18, "text-anchor": "middle", text: MONTHS[d.getUTCMonth()] }));
        if (d.getUTCFullYear() !== lastYear) {
          lastYear = d.getUTCFullYear();
          ax.appendChild(s("text", { x: X(t), y: m.t + ph + 36, "text-anchor": "middle", text: String(lastYear) }));
        }
      }
    }
    svg.appendChild(s("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "none", stroke: "rgb(204 204 204 / 0.35)" }));
    if (lo < 0) svg.appendChild(s("line", { class: "zero", x1: m.l, x2: m.l + pw, y1: Y(0), y2: Y(0) }));
    svg.appendChild(s("text", { class: "ax-label", transform: `translate(${o.compact ? 14 : 20},${m.t + ph / 2}) rotate(270)`, "text-anchor": "middle", text: o.yLabel }));

    const base = Y(Math.max(lo, 0));
    for (const se of o.series) {
      let line = "";
      let area = "";
      let start = null;
      let prev = null;
      const close = () => {
        if (start !== null && se.fill) area += `L${prev},${base.toFixed(1)}L${start},${base.toFixed(1)}Z`;
        start = null;
      };
      for (let i = 0; i < xs.length; i++) {
        const v = se.ys[i];
        if (v === null || !Number.isFinite(v)) {
          close();
          continue;
        }
        const px = X(xs[i]).toFixed(1);
        const py = Y(v).toFixed(1);
        const cmd = start === null ? "M" : "L";
        line += `${cmd}${px},${py}`;
        if (se.fill) area += `${cmd}${px},${py}`;
        if (start === null) start = px;
        prev = px;
      }
      close();
      if (se.fill && area) svg.appendChild(s("path", { d: area, fill: se.color, "fill-opacity": se.fill, stroke: "none" }));
      if (line) svg.appendChild(s("path", { d: line, fill: "none", stroke: se.color, "stroke-width": se.width, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }
    svg.appendChild(ax);

    // Quartz reference line with a filled label box at the top of the plot.
    let lastX = -Infinity;
    let row = 0;
    for (const mk of o.markers) {
      const px = X(mk.x);
      if (px < m.l || px > m.l + pw) continue;
      row = Math.abs(px - lastX) < 160 ? row + 1 : 0;
      lastX = px;
      const y = m.t - 30 + row * 26;
      const label = mk.label.toUpperCase();
      const w = label.length * 7.2 + 16;
      const lx = Math.min(Math.max(px - w / 2, m.l), m.l + pw - w);
      svg.appendChild(s("g", { class: mk.cls }, s("line", { x1: px, x2: px, y1: y + 21, y2: m.t + ph }), s("rect", { x: lx, y, width: w, height: 21 }), s("text", { x: lx + w / 2, y: y + 15, "text-anchor": "middle", text: label })));
    }

    if (!xs.length) {
      host.replaceChildren(svg);
      return;
    }
    const hover = s("line", { class: "hover-line", y1: m.t, y2: m.t + ph, visibility: "hidden" });
    const dots = o.series.map((se) => s("circle", { class: "hover-dot", r: 4, fill: se.color, visibility: "hidden" }));
    const hit = s("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" });
    svg.append(hover, ...dots, hit);
    const tip = h("div", { className: "tip", hidden: true });
    hit.addEventListener("pointerleave", () => {
      tip.hidden = true;
      hover.setAttribute("visibility", "hidden");
      dots.forEach((dt) => dt.setAttribute("visibility", "hidden"));
    });
    hit.addEventListener("pointermove", (ev) => {
      const rect = svg.getBoundingClientRect();
      const xv = x0 + (((ev.clientX - rect.left) * (W / rect.width) - m.l) / pw) * (x1 - x0);
      let a = 0;
      let b = xs.length - 1;
      while (b - a > 1) {
        const mid = (a + b) >> 1;
        if (xs[mid] < xv) a = mid;
        else b = mid;
      }
      const i = Math.abs(xs[a] - xv) <= Math.abs(xs[b] - xv) ? a : b;
      const px = X(xs[i]);
      hover.setAttribute("x1", px);
      hover.setAttribute("x2", px);
      hover.setAttribute("visibility", "visible");
      o.series.forEach((se, k) => {
        const v = se.ys[i];
        const ok = v !== null && Number.isFinite(v);
        dots[k].setAttribute("visibility", ok ? "visible" : "hidden");
        if (ok) {
          dots[k].setAttribute("cx", px);
          dots[k].setAttribute("cy", Y(v));
        }
      });
      const t = o.tip(i);
      tip.replaceChildren(h("div", { className: "tip-head" }, h("span", { text: t.title }), h("hr")), ...t.rows.map(([k, v, col]) => {
        const row = h("div", { className: "tip-row" }, h("span", { text: k }), h("span", { className: "num", text: v }));
        row.style.color = col;
        return row;
      }));
      tip.hidden = false;
      const left = (px / W) * rect.width;
      tip.style.left = `${left + 270 > rect.width ? left - 268 : left + 12}px`;
      tip.style.top = `${(m.t / H) * rect.height + 8}px`;
    });
    host.replaceChildren(svg, tip);
  }

  // ------------------------------------------------------------------ misc UI

  function toast(text) {
    const el = h("div", { className: "toast", text });
    $("toasts").appendChild(el);
    setTimeout(() => el.remove(), 4000);
  }

  function openMenu() {
    if (!state.result) return;
    const menu = $("download-menu");
    const items = [["predictions.csv", state.result.outputs.predictions], ["pv_extensions_predictions.csv", state.result.outputs.extensions]].map(([name, o]) =>
      h("button", { type: "button", className: "menu-item", role: "menuitem", onclick: () => download(name) }, h("span", { text: name }), h("small", { text: `${fmtInt(o.rows)} rows` })),
    );
    menu.replaceChildren(h("div", { className: "menu-head", text: "Exact files written by the CLI writers" }), ...items);
    menu.hidden = false;
    $("btn-download").setAttribute("aria-expanded", "true");
    items[0].focus();
  }

  function closeMenu() {
    $("download-menu").hidden = true;
    $("btn-download").setAttribute("aria-expanded", "false");
  }

  function init() {
    $("btn-synthetic").addEventListener("click", startSynthetic);
    $("btn-upload").addEventListener("click", () => $("file-input").click());
    $("file-input").addEventListener("change", (ev) => startAnalysis(ev.target.files && ev.target.files[0]));
    $("btn-download").addEventListener("click", (ev) => {
      ev.stopPropagation();
      if ($("download-menu").hidden) openMenu();
      else closeMenu();
    });
    document.addEventListener("click", (ev) => {
      if (!ev.target.closest(".menu-wrap")) closeMenu();
    });
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && !$("download-menu").hidden) {
        closeMenu();
        $("btn-download").focus();
      }
    });
    $("search").addEventListener("input", (ev) => {
      state.query = ev.target.value;
      renderTable();
    });
    $("sel-filter").addEventListener("change", (ev) => {
      state.filter = ev.target.value;
      renderTable();
    });
    $("sel-metric").addEventListener("change", (ev) => {
      state.metric = ev.target.value;
      renderCustomer();
    });
    $("sel-res").addEventListener("change", (ev) => {
      state.resolution = ev.target.value;
      renderCustomer();
    });
    let raf = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(renderCustomer);
    });
    ro.observe($("chart"));
    ro.observe($("profile"));
    window.addEventListener("focus", () => {
      if (!isActive(state.job) && !state.fatal) api("/api/session").then((sess) => { state.session = sess; renderHeader(); }).catch(() => {});
    });
    render();
    refreshSession();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
