# Numerical time-series embedding experiment — implementation plan

## Scope and branch

Create `time-series-embeddings` FROM `synthetic-data-pipeline`, then merge current `main` to inherit canonical preprocessing. Preserve local README/plan edits. Work inside this repository, no new Desktop worktrees. Keep both source branches unchanged. Commit implementation and push only the new experiment branch.

This is a runnable offline representation-learning experiment, not a claim of real AIL accuracy. No coding agent reads, lists, samples or processes AIL measurements/labels/extracts. Smoke runs and tests use wholly fabricated data. Authorized humans run the finished CLI locally on AIL data. No external pretrained checkpoints, language models, weather, location or irradiation. Available tools cannot choose Luna 5.6; two available general implementation agents will be used after this plan is written.

## Decision and experiment arms

Use a small PyTorch numerical patch encoder, randomly initialized and masked-reconstruction pretrained only on permitted training inputs. No Hugging Face checkpoint, no large model, no CNN requirement. Compare identical outer customer folds with fixed settings:

1. `baseline`: interpretable raw-energy/calendar/customer-summary features.
2. `frozen`: self-supervised encoder embeddings plus the same raw magnitude features; freeze encoder; fit small downstream heads.
3. `finetuned`: start from the same pretrained encoder, limited supervised fine-tuning on active-PV windows, then refit the same downstream heads.

Presence is the primary representation-learning hypothesis. Date and initial capacity are secondary outputs evaluated with the actual challenge score, not promised capabilities. Dates must come from time-indexed candidates, not one whole-history embedding. Capacity models use real capacity labels only and retain energy scale. Synthetic training is opt-in and lineage-checked, never an evaluation set or a capacity ground truth source. Arms must work when synthesis is absent.

## Canonical input and privacy

Reuse schema_version1 measurements.parquet, labels.parquet, splits.parquet, optional extensions.parquet, and run_manifest.json. Existing preprocessing keeps only literal non-null POD-prefixed IDs for unlabelled measurement sources; do not add a second conflicting loader/filter. Canonical input may combine labelled and unlabelled customers; only explicit real negative labels designate negatives. Verify version and required columns.

For outer fold k: exclude its labelled customers before ALL representation fitting, normalizer fitting, threshold selection, synthetic sampling and supervised heads. Explicitly permitted unlabelled IDs may supplement self-supervision only if they do not overlap any held-out labelled customer. Never include source/customer IDs as numeric features. Restrict synthetic variants by provenance source_customer_id and source_fold; reject missing, inconsistent or held-out ancestry. Synthetic kWp must remain absent and excluded from capacity fitting. Default no synthetic input.

Output directory must be empty/new and private; refuse overwrite. All embeddings, checkpoints, per-customer predictions and diagnostics derived from AIL stay ignored/private. No external network access at training/evaluation runtime. Source code/config and artificial fixtures only in Git.

## Frozen interface between parallel implementation slices

Package: `src/neuralwave_experiment/` (new, does not alter neuralwave_data APIs).

Encoder/data owner files: `__init__.py`, `windows.py`, `encoder.py`, `pretraining.py`, `tests/test_representation.py`, `docs/representation_encoder.md`.

Experiment owner files: `features.py`, `heads.py`, `scoring.py`, `experiment.py`, `tests/test_experiment.py`, `docs/embedding_experiment.md`. Parent owns pyproject extra, default config, README integration, artificial integration smoke and final review. No shared-file edits by agents.

Shared Python interfaces (names/shape fixed; message peer immediately for clarification):

- `WindowConfig` dataclass in windows.py: context_length=672, stride=96, min_observed_fraction=0.75. Numeric epoch length not civil-day reshape.
- `WindowBatch` dataclass: `values` np.float32 [N,L,1], `observed` np.bool_ [N,L,1], `calendar` np.float32 [N,L,4] (sin/cos local time-of-day and weekday), `magnitude` np.float32 [N,4] (center, robust scale, observed mean kWh, observed std kWh), `customer_ids` list[str], `center_dates` list[datetime.date], `start_epochs` int64[N].
- `build_windows(measurements: pd.DataFrame, config: WindowConfig) -> WindowBatch`: per-customer sorted unique epochs, enforce regular 900s canonical grid, observed mask true only for finite value_status observed; one left-padded window for short histories, one final tail window if distinct; no all-missing eligible windows; no gap-to-zero observations. Per-window median/MAD floor0.001 normalization, masked numeric fill0, preserve magnitude. Derive calendar from epoch to Zurich, avoid ambiguous timestamps. Each customer with zero eligible windows must produce an explicit validation error in evaluator, not disappear silently.
- `EncoderConfig` dataclass in encoder.py: context_length=672, patch_length=16, d_model=64, n_heads=4, n_layers=2, dropout=0.1. Nonoverlapping patches only; require divisibility.
- `NumericPatchEncoder(config)` nn.Module: `forward(values, observed, calendar)` returns embeddings [B,d_model]. Encode patch values/mask/calendar, positional information, zero-weight wholly unavailable patches at attention/pooling; no NaNs for sparse input. Use independent layer initialization. Retain amplitude outside shape encoder.
- `encode_windows(model, batch, batch_size=32, device='cpu') -> np.float32[N,d_model]` deterministic eval inference; no gradients.
- `pretrain_encoder(model, batch, *, epochs=5, batch_size=32, learning_rate=0.001, mask_ratio=0.3, seed=0, device='cpu') -> list[float]`: updates provided model in place, reconstruct observed values of hidden nonoverlapping patches only; hide values at all target positions AND prevent neighboring/covariate features exposing them. Original missing positions never enter reconstruction target. Do not copy target into model inputs. A learned mask token/decoder is owned by this module; return finite epoch loss history. Handle zero valid reconstruction samples clearly.

Torch is an optional project extra `experiment`; CPU is default for deterministic smoke. Use existing numpy/pandas/pyarrow/sklearn dependencies. Model configs and state_dict saved locally; no pickled arbitrary estimator downloads. Parent installs torch into an isolated environment if absent; agent skips installation/setup repetition.

## Downstream protocol (experiment owner)

CLI: `python -m neuralwave_experiment.experiment --canonical-dir DIR --out DIR [--config PATH] [--synthetic-dir DIR] [--seed N] [--device cpu]`.

Default config has context672/stride96, patch16, d64/layers2/heads4, pretrain_epochs5, finetune_epochs3, batch32, seed0. Smoke overrides dimensions/epochs for speed but runs all three real implementations. No hidden choice of pretrained weights.

Read labels/splits and use existing fold assignments. Read measurements by selected customer/Parquet filters rather than materialize 1148 long histories at once; construct only training windows and bounded batches. Loading windows for the small labelled experiment is acceptable with a documented max-window cap per customer; deterministic uniform-time selection, not only the start of history. Unlabelled pretraining optional with explicit capped per-customer sampling; do not allocate all overlapping histories densely.

Feature heads:

- Preserve per-window temporal embedding and magnitude alongside clock-band raw features.
- Presence: aggregate each customer's features using mean/std and first-versus-last change; train regularized logistic regression on real customer labels. Fixed threshold0.5 for initial comparison (or train-only inner-fold tuning, never outer labels).
- Date: build candidate local dates from covered time-indexed windows. Train a small regularized candidate scorer using only training-event information: local pre/post differences in midday-versus-night raw demand and (for embedding arms) embedding changes, with event-near candidates labelled positive and distant candidates negative. Event mask/candidate tolerance documented (e.g. +/-3 days), balance per customer, no candidate-label leakage. For detected PV choose highest-scoring candidate, tie-break earliest. Known pre-existing dates are not training candidates clamped to recording start; mark unobservable training events excluded and report separately. A train-fold lacking sufficient event candidates needs a declared raw-feature change-point fallback, not a silently fabricated learned model.
- Capacity: Ridge on log(real initial kWp) using amplitude and customer pre/post representation features from the initial installation segment only. Use optional extension labels to exclude extended segments. No synthetic capacity supervision. Require adequate positive real training capacities; otherwise fail the experiment with a clear prerequisite rather than invent capacity. Exponentiate finite positive predictions. Avoid training target leakage through installation-date-based feature selection at validation: use predicted date or label-free features there; training protocol must match.

Fine-tuning: learn active-PV window classification on training labels/dates using a temporary small torch head, updating at most the final encoder block/projection with low LR. Pre-installation windows of positive customers are negative; unknown/out-of-coverage event windows masked as appropriate. Do not assign all windows the customer-ever-PV label. Re-extract embeddings and refit identical downstream heads. Capacity labels not used in encoder fine-tuning initially. Drop the temporary active-PV head after representation fitting.

Synthetic optional augmentation: use only fold-safe variants with explicit date labels for active-window fine-tuning or pretraining; default keep downstream customer heads trained on real customers to preserve real prevalence. Never tune/evaluate against synthetic outputs. Fail closed on provenance mismatch. Do not reuse a fold0-generated directory across all outer folds without checking its ancestry.

`--synthetic-dir` denotes a root containing `fold_0/`, `fold_1/`, etc., each generated with its corresponding held-out fold. A single flat fold-specific output is not reused across outer folds. Require the matching directory and validate its source IDs/folds before augmentation.

## Scoring and artifacts

`score_predictions(labels, predictions)` reports precision, recall, F1, true-positive date D=mean(max(0,1-abs_day_error/30)), capacity P=mean(max(0,1-relative_error/0.5)), and main_score=F1*(70+15D+15P). Missing/invalid conditional targets score0; no correctly detected PV gives0. Pool outer out-of-fold predictions before calculating headline metrics; also report fold metrics. Exactly one prediction/customer/arm, preserve strings; output booleans and nullable date/capacity with non-PV fields empty. No test-label access during training/prediction, only scoring after outputs produced.

Private outputs: per-arm out-of-fold predictions CSV (semicolon), fold/pooled metrics.json, experiment config/seed/library versions/input identities, per-fold encoder checkpoints and pretraining/fine-tune loss histories, feature definitions and training-source lineage audit. Include clear actual-vs-artificial data designation supplied by smoke runner; never claim real benchmark evidence from fabricated fixtures. Model reload should reproduce frozen embeddings and predictions on a deterministic artificial example. No final hackathon inference.py required for this research CLI.

## Verification and delegation sequence

1. Parent freezes this plan before spawning agents, creates/merges branch, adds optional dependency and installs torch once.
2. Spawn two agents together with this contract. Encoder slice and evaluator slice proceed independently, communicate interface details. They skip builds/tests/formatters mid-flight; parent integrates and checks once slices complete.
3. Permanent tests catch boundary/invariant bugs: DST/gaps/padding and per-customer boundaries, reconstruction target masking, magnitude retention, deterministic eval reload, exact score penalties, conditional date/capacity masks, held-out ancestry and training-ID exclusion, synthetic provenance rejection, predictions for all evaluation IDs. No source-text/wiring-only assertions.
4. Actual CLI smoke uses independently fabricated canonical data with >=4 positives and >=4 negatives across two folds, known dates/capacities, enough days for transitions. Exercise baseline, real masked pretraining, frozen heads, real gradient fine-tuning, date/capacity predictions, pooled score, output reload. Use small encoder/one epoch in smoke; score is only a smoke number, never model-quality evidence.
5. Re-run existing preprocessing/synthesis behavioural suite in the combined branch after integration. Fix genuine failures, update usage docs, commit exact changed files and push `time-series-embeddings`. Do not push changes to main/source branches.

## Risks and interpretation

40 labelled customers limit reliable model comparison. Window count is not independent customer count. Self-supervised reconstruction may learn consumption routines rather than PV. Date estimation can be censored; initial kWp is not uniquely observable. Synthetic production is assumption-dependent. A working experiment cannot establish superiority to a CNN, handcrafted baseline or real-world generalization until an authorized human runs the fixed protocol locally on real AIL data. Do not access those data through agents to obtain that evidence.

References: PatchTST paper https://arxiv.org/abs/2211.14730; PyTorch numerical TransformerEncoder https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html. Context7 tools are not exposed; use official docs directly. No external pretrained checkpoint or source dataset is incorporated.

## Implemented verification

Implemented on `time-series-embeddings`, branched from synthesis and merged with main's preprocessing before implementation. No source branch was changed.

- Optional PyTorch dependency: `pip install -e '.[experiment]'`. Verification runtime used PyTorch 2.14.1 in an isolated temporary environment.
- Combined command: `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests`: **22 passed**.
- Actual CLI smoke prepared canonical files from eight wholly artificial labelled customers, two customer-disjoint folds, known onsets/capacities, and a separate artificial unlabelled customer.
- All three arms ran real masked reconstruction and limited gradient fine-tuning where applicable, wrote eight unique out-of-fold predictions per arm, and produced conditional date/capacity scores, histories, checkpoints and head artifacts.
- A second end-to-end run generated separate `fold_0` and `fold_1` synthetic directories using the existing synthesis CLI, enabled unlabelled pretraining, and completed all experiment arms with synthetic lineage checks and finite pretraining/fine-tuning histories.
- Reconstruction tests cover observed-hidden targets versus original missing positions; sparse deterministic encoder reload and strict grid/stride/tail behavior are exercised.
- PyTorch reports that nested-tensor optimization is not enabled for pre-norm layers. This is a performance warning, not a failed check.
- No AIL input or label was opened or processed. Smoke scores are deliberately not presented as evidence of predictive quality or superiority to a baseline.
