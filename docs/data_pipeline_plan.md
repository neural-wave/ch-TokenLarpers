# Preprocessing and synthetic-data implementation plan

Scope: two local, non-model pipelines. No training architecture, forecasting system, final predictor, external datasets, or weather/location/irradiance sources. Never open AIL measurements/labels with coding agents. Implement and verify using independently fabricated data only; authorized humans run the resulting ordinary Python locally on AIL files.

## Delivery and ownership

- `preprocessing` branch: ingestion, canonical schema, daily features, labels/extensions, customer-disjoint splits, packaging, preprocessing CLI, privacy ignores, tests and usage docs.
- `synthetic-data-pipeline` branch: constrained generation, validation, provenance, synthesis CLI, tests and usage docs. This is the valid Git spelling of the requested branch name containing a space.
- Each branch lives in a separate worktree. Do not merge or switch the user's main worktree. Deliver commits on each branch and exercise their integration in an isolated temporary workspace.
- Available agent controls do not expose Luna 5.6 selection. Use two general-purpose implementation agents; do not claim a specific underlying model.
- Share this schema before implementation. Synthesis depends on canonical files, not private internals of ingestion; both slices can implement concurrently.

## Privacy and team workflow

Keep the existing raw-data ignore. Add `/data_private/`, `/artifacts_private/`, `/.env`, `/predictions.csv`, `/pv_extensions_predictions.csv`, Python cache/build ignores as appropriate. Gitignore is not an AI-access boundary. Exclude private paths from agents or keep them outside the code workspace. No real-data-derived samples, logs, hashes or reports enter tracked tests/docs or external agent context.

Each teammate obtains organizer files through the authorized channel and supplies a private manifest; Git shares code, schemas, fixed config and fully artificial fixtures only. Raw files remain unchanged. Local Parquet, reports, folds, generation provenance and fitted artifacts remain private.

Minimal runtime: Python 3.11+, numpy, pandas, pyarrow; preprocessing may use scikit-learn for stratified folds. Packaging uses `src/neuralwave_data`, installed with `pip install -e .`. Do not add large synthesis frameworks.

## Shared persisted contract v1

Every run output has `schema_version: 1` in `run_manifest.json`; consumers reject unsupported versions. Parquet artifacts are ordinary files, not framework-specific serialization. All joins preserve exact string customer IDs. Missing label does not imply non-PV.

### `measurements.parquet`

Columns, in order:

1. `customer_id`: string, non-null.
2. `timestamp_epoch`: int64 Unix seconds, non-null.
3. `timestamp_local`: timezone-aware Europe/Zurich timestamp, derived from epoch.
4. `local_date`: date.
5. `local_slot`: uint8, local hour*4 + minute//15; repeats during autumn DST.
6. `dst_fold`: uint8 0/1.
7. `utc_offset_minutes`: int16.
8. `energy_kwh`: nullable float64, original interval energy, never implicitly normalized.
9. `value_status`: string enum `observed`, `absent_interval`, `missing_input`.

Unique key `(customer_id, timestamp_epoch)`, sorted by that key. Reindex first-to-last epoch per customer on a 900-second grid, not to artificial common start/end dates. Null energy is unavailable, not zero. No mandatory smoothing, clipping, or imputation. Preserve finite negative values unless an explicitly declared meter mode rejects them. Exact duplicates collapse; conflicting duplicates fail. Empty retained inputs produce a typed empty table/report, not a crash or invented customer.

For memory-bounded ingestion, use chunked CSV plus a local temporary SQLite staging table (standard library), or equivalent disk-backed partitioning. Do not accumulate all raw chunks in a list and concatenate. Final shared file may be written per customer using ParquetWriter. Report counts without exposing rows in exceptions.

### `labels.parquet`

`customer_id` string; `pv_installed` bool; `pv_installation_date` nullable date; `pv_capacity_kwp` nullable float64. One row per labelled customer. Non-PV date/capacity null; positive capacity finite and positive when supplied. Parse booleans explicitly. Keep installation dates outside observation coverage; they may be legitimate. Labels are optional for preparation and never required for measurements-only use.

### `extensions.parquet`

`customer_id` string; `extension_date` date; `pv_capacity_after_extension_kwp` positive float64. Multiple events/customer allowed; validate duplicate events, positive capacities and chronology when initial labels are available. Do not combine initial and extended capacity.

### `splits.parquet`

`customer_id` string; `fold` int64. One row/customer in labelled development set. Deterministic stratified customer folds when each class supports >=2 folds; reduce requested count to smallest class. If impossible, fail split creation clearly rather than manufacture reliable validation. Support preparing measurements without requesting folds. A held-out fold is excluded by synthesis, including all descendants.

### `daily_features.parquet`

One row/customer/local_date. Include expected_intervals (92/96/100 from adjacent local midnights), observed_intervals, coverage_fraction, is_boundary_day, observed energy sum (null for all-missing), weekday/month, zero/negative fractions, and observed counts/mean/median for half-open local bands night [00,05), morning [06,09), midday [10,15), evening [18,22). Include midday/night ratio with null zero denominator. Require configurable 0.9 daily/band coverage for affected statistical features, preserve counts/observed energy even on partial days. Never extrapolate partial-day sums. Boundary days are flagged, not dropped. Gaps stay unavailable.

## Pipeline A: preprocessing

### Input manifest and config

Private JSON manifest has `sources`: a list of objects `{role, path, delimiter, decimal, encoding}`. Roles: `development_measurements`, `unlabelled_measurements`, `development_labels`, `extension_labels`, `inference_measurements`. Paths resolve relative to the manifest directory; `${AIL_DATA_ROOT}` expansion is supported if implemented and documented. Explicit defaults: UTF-8, semicolon delimiter, decimal point. Epoch unit seconds. Reject ambiguous or unsupported settings instead of guessing silently. Input columns match the challenge schema; extra columns may be ignored explicitly.

**Organizer correction: for `unlabelled_measurements`, keep ONLY rows with a non-null ID whose literal value starts with case-sensitive `POD`. Apply this before validating timestamps/energy, duplicate checks, feature computation, or any analysis.** Equivalent SQL: `customer_id LIKE 'POD%'` under case-sensitive comparison. Do not trim/lowercase IDs to make excluded records qualify. Count excluded null/non-POD rows separately in the private report. Junk excluded rows may have invalid numeric/time fields and must not abort preprocessing. Do not silently impose this filter on labelled/inference roles: preserve their provided IDs and reject invalid null/empty IDs there.

### Implementation order

1. `schema.py`: typed empty tables, version and validation helpers.
2. `io.py`/`canonical.py`: strict streaming CSV ingestion; early unlabelled filter; explicit ID/numeric/time parsing; disk-backed grouping/dedup; canonicalization.
3. Validate epoch `% 900 == 0`; derive Zurich local instant. Compare supplied local wall-clock fields against epoch-derived fields, including explicit offsets if provided; avoid independently localizing ambiguous naive strings. Invalid retained data fails with sanitized diagnostics. Numeric blank is missing_input, malformed/nonfinite numeric fails.
4. `features.py`: coverage-aware daily aggregates.
5. `splits.py`: deterministic customer-level folds from labels.
6. `prepare.py`: CLI orchestrates measurements, optional labels/extensions/folds, private QA counts and run manifest. Never overwrite a populated output directory silently. Output files use the shared names. Multiple measurement sources with overlapping IDs must be rejected or safely reconciled explicitly; document chosen policy.
7. `pyproject.toml`, default config and README usage. Avoid training or evaluation-model code.

CLI: `python -m neuralwave_data.prepare --manifest PATH --out DIR [--config PATH] [--folds N] [--seed N]`.

`run_manifest.json` records schema version, configured source roles, source hashes, settings, row/filter/duplicate/gap counts, artifact names and software versions. Data-bearing reports remain under the output directory. Manifest absent/invalid paths fail without printing file contents.

### Acceptance and smoke

Permanent regression tests: early POD filter with null/non-POD malformed records; literal case-sensitive ID matching; labelled IDs preserved; 92/96/100-day handling and repeated wall clocks; shuffled input; exact/conflicting duplicates; gaps versus zeros; all-null days; partial bands; finite negative values; leading-zero IDs; explicit locale; unsupported schemas/settings; labels optional; fold disjointness/determinism.

Smoke: invoke the actual CLI on artificial CSV/JSON in a temp directory, reload Parquet, inspect exact energy, masks/status, IDs, QA exclusions and DST counts. No AIL input. Update branch README/docs with exercised commands.

## Pipeline B: constrained synthetic data

### CLI and inputs

`python -m neuralwave_data.synthesize --canonical-dir DIR --splits PATH --held-out-fold N --out DIR --meter-mode {import_only,signed_net} [--seed N] [--config PATH]`.

Input canonical directory must contain versioned manifest, measurements and labels. Read splits using the shared columns. Require every selected background has a unique split assignment; exclude held-out fold before computing any scales/statistics. Select only explicitly confirmed non-PV labelled training customers. Unlabelled customers are not negative backgrounds. Reject unknown meter mode; no fabrication of semantics. Negative baseline readings are incompatible with treating a baseline as nonnegative native load: exclude/reject with explicit reason, do not silently clip.

Implement own synthesis modules only; no dependency on preprocessing internals beyond the on-disk contract. Namespace package files are owned by preprocessing. Synthesis can be tested standalone with `PYTHONPATH=src` and no package initializer; integration adds packaging from preprocessing.

### Construction

For each eligible background, choose an internal local-date onset with configurable minimum pre/post coverage. Default minimum 7 observed local days on each side; require adequate coverage and nonzero daytime scale. Raise a clear error/report for no eligible backgrounds. Preserve source timestamps and missingness. Use a smooth compact daytime production basis in fixed UTC+1 standard-clock coordinates, with variable width/asymmetry, smooth annual modulation, bounded correlated daily/intraday attenuation and an optional activation ramp. Use timestamps only, not external solar geometry or weather. No PV before onset, nonnegative G, zero outside declared support.

Strength is an observable energy-deficit fraction of a robust training-background daytime scale, not kWp. Default range 0.05-1.5; expose meaningful validated bounds in config. No unnecessary data-fitted generator or deep generation model.

Generate four paired scenarios per source/seed: original negative, PV positive, demand-change hard negative, PV plus independent demand change positive. Demand changes include whole-day decline, business-hours decline and temporary holiday decline across seeds/variants; keep nonnegative latent load. Pairwise PV checks compare equal demand backgrounds. Apply Y=L-G for signed_net; Y=max(L-G,0) for import_only. Restore source missingness after arithmetic. Do not add nuisance gaps/noise only to one class. Use distinct generated IDs and provenance; never predictive source IDs.

### Outputs

- `measurements.parquet`: exact canonical columns above, with generated IDs and unchanged temporal metadata/status.
- `synthetic_labels.parquet`: generated customer_id, pv_installed bool, pv_installation_date nullable date, pv_capacity_kwp always null, date_target_valid bool, capacity_target_valid always false. Keep separate from real labels because null positive capacity is deliberate here. Non-PV onset null. Covered positive onset is the known injected event, not a claim about real commissioning observability.
- `provenance.parquet`: generated customer_id, source_customer_id, source_fold, seed, generator_version, scenario, meter_mode, onset date, ramp days, scale and parameters/config hash.
- `latent.parquet`: generated customer_id, timestamp_epoch, baseline_kwh, demand_kwh, pv_kwh for independent equation checking; private output only.
- `quality_report.json`: hard-gate results, excluded-source reasons/counts, diagnostic distribution summaries.
- `run_manifest.json`: schema version, generator version, seed/config, source manifest identity, output names and counts.

Do not silently overwrite populated output directories. Seed determinism must survive input row ordering. Process customers/variants incrementally, not all four variants for the entire dataset as one dense allocation.

### Validation and acceptance

Hard checks independently recompute equations from latent components; reject nonfinite observed values, negative PV, invalid import sign, pre-onset/outside-support generation, changed source missingness, inconsistent positive/date flags, fabricated capacity, held-out ancestry, duplicate generated keys, missing provenance and inconsistent schema version. Persist valid source_fold lineage. Diagnostic checks report energy quantiles, zero fractions, ramp statistics and daily-profile differences without claiming realism is certified.

Permanent tests cover both meter modes/strong clipping, onset gate/ramp, standard-time DST continuity, hard-negative labels, demand-change boundaries, missingness/null inputs, zero-scale/short sources, positive real labels excluded, held-out customers excluded before scale computation, missing split assignments, seed determinism/input reorder, and validator rejection of corrupted equations/labels.

Smoke: build a private artificial canonical directory directly from this contract, invoke the actual CLI, reload outputs, validate source disjointness and each equation/label. No AIL inputs and no tests that merely assert forwarding or file existence.

## Integration and done condition

Each agent owns its branch, skips mid-flight checks, then runs its focused suite and actual CLI once implementation is complete, fixes genuine failures, updates usage docs and commits only its own files. No agent may inspect raw AIL directories.

Parent integration owner combines the two branches in an isolated temporary worktree, runs all permanent tests and an actual artificial CSV -> preprocessing -> folds -> generation scenario. Both positive and negative labels are supplied so folds are meaningful; held-out ancestors must not appear in synthesis. Keep original branches intact; integration fixes go back to the owning branch. Deliver branch names, commit IDs, commands exercised and any limits. Neither pipeline claims improved real-data generalization; real-only versus augmented customer-held-out evaluation remains the consumer team's responsibility.
