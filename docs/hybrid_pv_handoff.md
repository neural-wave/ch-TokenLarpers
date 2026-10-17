# Hybrid PV transformer — historical integration snapshot and main-promotion guide

## 1. Read this first

This document records the historical integration-branch snapshot and the guide for promoting that work to `main`. It is not a claim that the transformer is superior, and it must not be read as a statement of the current checked-out branch.

The user wanted to learn representations from the challenge's large unlabelled consumption dataset, adapt them using the small labelled development set, and determine whether they improve PV detection. The eventual implementation combined a frozen PatchTST encoder with Jeffre's domain-feature pipeline rather than replacing that pipeline.

### Historical integration snapshot

The following is immutable historical context for the integration work; it is not a current-branch assertion:

- Historical working branch: `integration/hybrid-pv-v1`.
- Historical latest completed merge: `a29735c`.
- Historical upstream merge source: `origin/Jeffremm10/new-stack` at `ffe1f69`.
- Local learning-curve fixes were preserved first in `1339f8b`.
- Nothing was pushed during that historical merge. Source branches were not overwritten.
- This document was prepared after that merge; it was a documentation deliverable, not part of the previously completed merge commit.
- No training or evaluation was launched while preparing that handoff.

The main-promotion owner should verify the intended commits, merge/push state, approved repository-contained encoder files, and fresh remote-main behavior independently. Do not infer any of those from this historical snapshot.

Subsequent user instruction explicitly approved committing both pretrained encoder pairs directly under `pretrained/small` and `pretrained/medium` for team/judge reuse. This is a narrow publication exception to the historical model-artifact rule below. `configs/pretrained_encoders.json` and the hygiene checks pin the exact four approved files; private training state and AIL metrics remain excluded. See `docs/hybrid_team_setup.md` for the current clone-contained workflow.

### Binding privacy boundary

Read `CLAUDE.md`, `IMPLEMENTATION_HANDOFF.md`, and README's organizer clarifications before doing more work.

The incoming organizer clarification explicitly prohibits sharing AIL-derived per-customer values **or aggregate statistics** with coding agents. Accordingly:

- Agents develop and verify on fabricated data from `pvdetect/synthetic.py`.
- Real-data preprocessing/training/evaluation commands are run by the data holder.
- Do not open raw datasets, private checkpoints, real-data logs, prediction files, QA reports, or private metric reports in an agent session.
- Do not repeat the earlier agent-side real-data inspection/execution workflow merely because it appears in the archived conversation.
- Do not put real-data results or model artifacts in tracked documentation, commits, public reports, screenshots, or prompts.
- A separate, private historical-results appendix was supplied to the user from numbers already present in the conversation. It is for the data holder, not for forwarding into another coding-agent session.

The challenge brief describes 40 labelled development customers, 1,148 unlabelled customers, and 20 hidden evaluation customers. Those are organizer-supplied population descriptions, not independently verified current data contents.

## 2. Goal and challenge tasks

The challenge asks for:

1. Whether PV is installed.
2. Its installation/commissioning date.
3. Initial nominal capacity in kWp.
4. An optional extension output; extensions are still header-only in this implementation.

Inputs are quarter-hourly electricity-consumption/grid-withdrawal histories and timestamp-derived information. They are not direct solar-generation measurements. No external weather, irradiation, inferred location, or LLM inference is permitted. The final predictor must work offline.

The research hypothesis was:

> Self-supervised representations learned from permitted, disjoint unlabelled customers may contain PV-relevant information beyond hand-designed consumption features, reducing the amount of labelled supervision needed.

This was a hypothesis to test, not an established benefit of using a transformer.

## 3. How the approach evolved

### 3.1 Earlier transformer work and credibility audit

Before the integrated hybrid, the repository contained custom numerical encoders, an adapted official PatchTST backend, whole-customer embedding experiments, and a supervised window-level PV classifier.

The earlier window classifier assigned supervised labels to windows using known installation dates and aggregated window probabilities into a customer decision. Its historical reported scores are **not** the scores of the final hybrid.

The audit identified reasons not to treat early numbers as conclusive evidence:

- Several experiments used different preprocessing, windows, aggregation, budgets, or split procedures.
- A rolling aggregation based on consecutive sampled windows did not necessarily mean consecutive calendar days.
- In-sample threshold calibration was weaker than proper inner-CV calibration.
- Fixed-customer-split seed repetitions do not independently establish generalization.
- Synthetic injection and physical-profile inspections were not independent real-world validation.
- An exploratory hourly-profile logistic model was a serious comparator; a transformer was not automatically the strongest approach.

A corrected standalone evaluator was implemented in `src/neuralwave_experiment/evaluation.py` with customer-level nested calibration, persisted split lineage, and calendar-time temporal diagnostics. It remains a separate research path from the hybrid described below. Its matched arms are not interchangeable with the full three-task hybrid comparison.

### 3.2 Clarifying the two pipelines

There was confusion in the conversation about Jeffre using an already-pretrained model. The implemented `pvdetect` pipeline we inspected used fitted logistic/CatBoost and ridge/CatBoost models, not a pretrained transformer as its core.

The integration decision was:

- Keep the working three-task `pvdetect` pipeline.
- Pretrain a numerical transformer from scratch on permitted unlabelled histories.
- Freeze the encoder.
- Add a small, fold-fitted embedding feature block to the presence classifier.
- Leave date and capacity estimators on their established physical-feature paths for the first comparison.

LoRA and end-to-end encoder fine-tuning were discussed but were **not implemented or evaluated** in this hybrid experiment. The small number of labelled customer groups, not GPU memory, motivated keeping the encoder frozen initially.

### 3.3 Controlled implementation and merging

The integration was planned in `goals/hybrid-pv-integration/plan.md`; the user approved the plan in chat, recorded in `approval.json`.

Work began in a separate integration worktree, combining:

- Corrected transformer evaluation/pretraining code.
- The shared customer-split and out-of-fold output contract.
- Jeffre's complete presence/date/capacity pipeline.

Conflicts were resolved explicitly; the older calibration/persistence behavior was not restored merely to make Git merge cleanly. The integration branch was subsequently moved to the user's ordinary checkout so they could work there.

Later, Jeffre's newer `new-stack` implementation was fetched and merged at `a29735c`. Its expanded option pool is described in section 10. The completed historical learning curve predates that last merge.

## 4. What “physical baseline” means

It does **not** mean a physics simulator or a collection of fixed rules with no learning.

The baseline extracts domain-informed consumption features, including daytime/nighttime contrasts, seasonal differences, near-zero daytime import, persistent changes, and consumption magnitude. These are indirect PV clues and can also be produced by non-PV behavior.

It then fits:

- Logistic regression or CatBoost for presence.
- Season-aware change-point/date-regime logic with fold-trained corrections for the installation date.
- Ridge regression or CatBoost with magnitude/deficit features for capacity, plus selected corrections.

The feature pipeline uses a fixed UTC+1 day matrix. Transformer calendar features use Europe/Zurich local time; those conventions are distinct and should not be silently conflated.

The hybrid adds evidence to the baseline; it does not replace all three task estimators with a neural network.

## 5. Self-supervised pretraining: exact implemented design

### Inputs and population separation

A manifest explicitly separates unlabelled training and unlabelled validation sources. Labelled evaluation customers must be excluded from both. Training and validation must be customer-disjoint; windows from one customer must not be randomly split across these populations.

Unknown PV status in the unlabelled pool is not interpreted as “non-PV.” No actual PV labels, commissioning dates, or capacities are used as reconstruction targets.

### Window and model configuration

Both checked-in configurations use:

| Setting | Value |
|---|---|
| Backend | Adapted, vendored `official_patchtst` |
| Input channel | One numerical energy channel |
| Context length | 672 quarter-hourly values: seven days |
| Patch length | 16 values: four hours |
| Tokens per full window | 42 non-overlapping patches |
| Candidate-window stride | 96 values: one day |
| Minimum observed fraction | 0.75 |
| Maximum indexed windows per customer | 128 |
| Sampled windows per epoch | At most 4,096 |
| Epoch budget | Five |
| Batch size | 64 |
| Mask ratio | 0.30 of eligible patches |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Weight decay | 0.0 |
| Gradient clipping norm | 5.0 |
| Dropout | 0.2 |
| Seed | 17 |
| Final GPU AMP setting | bfloat16 |
| Early-stopping argument used in the historical runs | Patience three |

Architecture variants:

| Variant | Width | Layers | Attention heads | Historically measured instantiated pretraining-model parameters |
|---|---:|---:|---:|---:|
| Small | 64 | 2 | 4 | 110,034 |
| Medium | 128 | 3 | 4 | 614,931 |

These measured counts concern the instantiated adapted model, including its reconstruction machinery, not just attention matrices. Earlier discussions used slightly different counts for classifier-bearing or custom models; do not mix them with these two variants.

Files: `configs/pretraining_unlabelled.json`, `configs/pretraining_unlabelled_medium.json`.

### Training objective and what gets trained

For each sampled window:

1. Retain genuinely observed points and select eligible hidden patches.
2. Reconstruct physical values from the window representation.
3. Compute normalization from visible observations only, so hidden targets cannot leak through the normalizer.
4. Hide target-patch energy values from the encoder and update the observation mask accordingly.
5. Feed values, observation masks, and timestamp-derived calendar features to the encoder.
6. Map encoded patch vectors back to patch values using a small linear reconstruction head.
7. Minimize mean squared error only at genuinely observed positions in selected hidden patches.

The encoder and reconstruction head both receive gradient updates. This is not a language-model decoder and not training only a head on a permanently random frozen encoder. The reconstruction head is not used to predict PV presence/date/capacity downstream.

Current visible-only training normalization:

- Median centering.
- MAD when it is sufficiently nonzero; standard-deviation fallback otherwise.
- A positive scale floor of 0.05.
- Normalized values clipped to [-50, 50].

Missing measurements are not known zero-valued reconstruction targets.

### Sampling, validation, and stopping

The sampler randomizes customer order per epoch, draws round-robin across customers, and rotates among each customer's available seasonal buckets. It improves customer coverage and limits long-history dominance.

An epoch is a **sampled-window budget**, not an exhaustive pass over every quarter-hour or eligible overlapping window. Visiting every eligible customer does not establish exhaustive historical or seasonal coverage.

Validation runs on distinct unlabelled customers under `model.eval()` and `torch.no_grad()`, with a fixed masking RNG seed. Validation loss guides checkpoint selection and optional early stopping; it is not a PV-label accuracy measure.

Private resumable checkpoints include weights, optimizer, AMP scaler where applicable, sampler position, RNG state, training histories, progress, and validation progress. Best-validation weights are exported as the compact `encoder.pt`/`encoder.json` pair.

## 6. Data preparation and GPU execution history

The data-preparation work converted raw CSVs into canonical Parquet with timestamp validation, explicit missingness, duplicate handling, daily features, labelled customer folds, and customer-disjoint unlabelled pretraining partitions.

Important issues encountered:

- A source used a different CSV delimiter; the pipeline correctly failed rather than silently dropping it.
- One extension annotation lacked required event fields; it was preserved separately instead of inventing date/capacity values or treating it as proof of no extension.
- The original bounded SQLite staging design was safe for unordered inputs but expensive in time and disk.
- CSV parsing and daily-feature computation were vectorized while preserving their contracts.
- The prepared unlabelled population did not exactly match the brief's stated count. The session did not establish the cause; do not attribute it to a specific customer without authorized data-holder investigation.

The historical GPU environment used a Vast RTX 5090 with Python 3.12 and a CUDA 12.8 PyTorch build. An initial synthetic setup pilot verified CUDA execution for both architecture sizes, small-model interrupted/resumed state equality on that host, and GPU-trained encoder reload on CPU. GPU/CPU embedding differences were reported below approximately 4e-7 on that synthetic probe.

A stopped/reassigned instance was replaced; originals and prepared inputs remained on the user's Mac. The prepared upload was compressed with zstd after slow direct transfer. The real pretraining runs used the replacement environment. Hostnames, SSH details, credentials, and current server state are intentionally not carried in this document.

**Current GPU state/billing is unknown.** Check the console before claiming a server is running, stopped, or deleted. No server was stopped or destroyed during the final merge/handoff work.

### Numerical and operational fixes

- FP16 pretraining encountered non-finite losses on the historical workload.
- Source investigation found zero/near-zero MAD could produce extreme normalized inputs.
- Robust scale fallback, a larger floor, clipping, and bfloat16 AMP were added. Subsequent completed runs are described in the user's private results appendix.
- An outdated installed package was loaded instead of the edited checkout during earlier remote attempts. Editable installation/explicit source paths were needed; an old running process does not pick up code edits automatically.
- Stream scanning initially converted large columns to Python objects and wrote excessive tiny shards. PyArrow operations and per-customer writers replaced that path.
- Existing window shards were later reused without rebuilding them.
- Deterministic cache directories avoided repeated expensive indexing between runs.
- Customer coverage and best-validation checkpoint selection were added before completed training.

Technical success here means the machinery executed, resumed, validated, and exported. It does not by itself establish useful solar representations.

## 7. Downstream frozen-feature integration

### Exact feature path

For each labelled customer:

1. Build chronological eligible windows using the exported encoder's saved window configuration.
2. Run the frozen encoder to produce one vector per window.
3. Aggregate vectors as coordinate-wise mean, standard deviation, and last-minus-first difference.
4. Concatenate those into a vector with `3 * d_model` values: 192 for small, 384 for medium.
5. Fit imputation, standardization, and a four-component PCA projection using only the current training customers.
6. Append the four resulting `embedding_pc1`–`embedding_pc4` features to the presence feature matrix.
7. Fit/select the small supervised task models and threshold under inner CV.

PCA is variance-preserving, not PV-supervised feature selection. The raw mean/std/drift summary does not retain a complete time-indexed installation trace.

### What stays unchanged

In the first hybrid experiment, date detection and capacity regression continued to use their existing physical-feature matrices and estimators. Their predictions and the full score can still change indirectly if presence decisions or selected true positives change.

No encoder weights were updated using the labelled customers in this hybrid. No LoRA, end-to-end encoder adaptation, or embedding-based installation-date model was evaluated as part of the completed learning curve.

### Inner selection and persistence

`--feature-block auto` offers baseline versus embedding-assisted prediction using identical inner customer folds. The hybrid requires at least a two-point improvement in full inner score and paired-bootstrap stability at least 0.80. This bootstrap is an inner selection heuristic, not an independent significance test.

`--feature-block embedding` forces the embedding path for diagnostics; it is not evidence that automatic selection prefers it.

Raw embeddings may be cached across folds because the encoder is fixed and fitted on an excluded unlabelled population. PCA, imputation, scaling, label-fitted climatology, date-regime parameters, and supervised models must remain fit-local.

A selected baseline saves no encoder dependency. A selected hybrid saves:

- `model.json`, schema v3.
- Referenced CatBoost siblings when selected.
- `encoder/encoder.pt` and `encoder/encoder.json`, with hashes.
- The fitted projection and exact feature names.

The in-memory fitted hybrid must retain `_embedding_encoder`; otherwise `predict()` cannot construct embedding columns before serialization/reload. This omission was found and fixed.

## 8. The completed labelled-sample learning curve

The completed historical study compared three complete procedures at each labelled training size:

- Physical-feature baseline.
- Baseline with the small pretrained encoder available as an optional feature block.
- Baseline with the medium pretrained encoder available as an optional feature block.

The encoders were pretrained once and frozen; they were not retrained for each labelled fold or each labelled sample size.

### Customer-level protocol

- Three predefined outer customer partitions, from split-generation seeds 17, 42, and 100.
- Five folds per partition.
- Each outer fold holds out eight of the 40 development customers.
- From the remaining 32, select a class-stratified training subset of size 8, 16, or 32.
- The subset-selection seed is `17 + repeat_index * 100 + fold_index`.
- All three competing arms use the same outer assignments and the same subset for a given repeat/fold/training size.
- The unused outer-training customers at smaller training sizes are not extra supervised training examples.
- Model/threshold/feature choices use only that subset's inner CV.
- Pool the held-out predictions across five folds before scoring each repeat.

The design has nine configurations and 135 outer fits. Every configuration should produce one prediction per development customer per repeat: complete coverage implies 1,080 held-out prediction rows (`40 * 3 * 9`). Historical artifact-verification details belong in the private appendix.

**Important subtlety:** subsets are stratified and deterministic, but are selected separately for each requested size. The implementation does not enforce that the eight-customer subset is literally contained in the sixteen-customer subset. Do not call them strictly nested training subsets.

### Score interpretation

The main challenge score is:

`F1 * (70 + 15 * D + 15 * P)`

- F1: PV-presence F1, not percent accuracy.
- D: installation-date reward, averaged over true positives; linear to zero at 30 days absolute error. Missing dates score zero.
- P: capacity reward, averaged over true positives; linear to zero at 50% absolute percentage error. Missing/nonpositive invalid capacity predictions score zero.

Do not use the earlier incorrect additive description `70*F1 + 15*D + 15*P`.

Displayed mean ± standard deviation is across three split repeats, not a confidence interval. Repeats reuse the same development population and are not independent datasets. No blind-test result was established.

### Where to obtain the historical numbers

The user received a separate private appendix with the exact previously reported nine-row result table, feature-adoption counts, and pretraining losses. Those figures are not reproduced in the agent-safe repository handoff. Only the data holder should inspect the private result files listed below.

## 9. Why evaluation took so long, and what changed

The earlier execution was repeatedly restarted with inadequate 120/180-second tool deadlines, sparse progress logging, and end-of-run-only saving. Early runtime promises were not measured and were wrong.

Contributing software issues/fixes included:

1. Stale installed-package imports versus edited checkout code.
2. Missing active encoder retention in fitted in-memory hybrid models.
3. Repeated embedding extraction; an initial cache fix tried to add an attribute to a frozen `FeatureTable`, so it did not persist. The final cache uses its existing mutable `cache` dictionary, with encoder hashes/configuration/frame identity in the key.
4. Scalar calendar conversion and per-window calendar assembly. The final implementation computes a vectorized calendar per customer timeline and slices it into windows. DST and short/padded windows have regression coverage.
5. Repeated official-score DataFrame normalization inside the inner paired bootstrap. Per-row scoring terms were precomputed. Missing-date handling was corrected to match official scoring.
6. Timing assumptions used 128 windows/customer while the historical exported encoders requested uncapped downstream window extraction.
7. No fold-level recovery in the original learning-curve script.

Current `scripts/run_learning_curve.py`:

- Prints loading, physical-feature construction, encoder extraction, and fold progress.
- Defaults Torch CPU execution to one thread; configurable with `--torch-threads`.
- Uses saved encoder window settings without silently dropping windows.
- Atomically saves each completed fold to `<out>.progress.json`.
- Saves aggregate results after every completed configuration.
- Supports `--resume` only when input/encoder/code/configuration/runtime fingerprints match.
- Refuses an existing output path without explicit resume.
- Does not impose its own runtime deadline.

A fabricated-data interruption exercise preserved seven completed folds, resumed to 18 completed folds across the three arms, matched a fresh run's results exactly, and rejected changed input bytes. A separate synthetic uncapped profiling scenario reduced extraction from 3.98 to 1.56 seconds on the same workload; that is a development performance check, not a PV-accuracy result.

## 10. The latest Jeffre merge changes the current baseline

Upstream `ffe1f69` was merged into this branch at `a29735c` after the historical learning curve completed.

Incoming behavior retained:

- E1 persistence feature candidate.
- E2 logistic/CatBoost probability averaging candidate.
- Optional schedule-consistency feature block.
- Optional unlabelled-derived climatology and evidence scaling, with sensitivity gates.
- Pseudo-labelling from a bounded unlabelled pool.
- `train.py --split-dir` and related split controls.
- Shared CLI/GUI inference, diagnosis/comparison tooling, and optional local GUI.
- Loader and threshold tie-break fixes.

Integration resolutions:

- `FeatureTable` retains its epoch frame, unlabelled prior, and pseudo pool.
- `fit` accepts upstream `options` alongside `encoder` and `feature_block`.
- Averaged classifiers retain both required model components for saved inference.
- Frozen embeddings still augment only the presence feature names; projections remain fold-local.
- Pseudo and hybrid are separate candidates because the compact pseudo pool has no raw epoch frame for encoder windows.
- If pseudo wins physical selection, the hybrid must compare against that actual winning candidate, not the weaker pre-pseudo baseline. `choose_pseudo` now returns the matching selected OOF frame as a third tuple item.
- A selected hybrid records pseudo as off with a reason rather than attempting a broken combined pseudo-plus-embedding refit.
- GUI assets and licenses were added to wheel package data.

**Do not treat historical pre-merge scores as measurements of this expanded pipeline. Do not resume an old fingerprinted study against changed source code or overwrite its result artifacts.** A new real-data comparison, if needed, belongs to the data holder and needs a new output path.

### Verification actually exercised for the merge

- Full test run: 243 passed and two fixture failures.
- The GUI corruption fixture omitted a referenced capacity-model sibling; it was fixed to preserve all siblings before corrupting the intended classifier artifact.
- The privacy/release fixture bypassed OOF generation with `None`, which no longer worked once selected outputs were rescored; that obsolete bypass was removed.
- Focused rerun: both corrected checks passed.
- No second full-suite run after those fixture-only edits was claimed.
- Actual synthetic CLI smoke passed: baseline fitting; forced-hybrid customer-CV with schedule, unlabelled source and pseudo-pool preparation; 16 correctly assigned OOF rows; saved encoder reload; baseline/hybrid offline inference on six synthetic customers.
- The small synthetic unlabelled prior was gated off for insufficient population; the smoke does not prove an active full-sized prior was selected.
- Wheel build and checks for pipeline, encoder, GUI assets, and bundled licenses passed.
- Hygiene checks passed before committing.

Earlier pre-merge runtime verification passed 168 tests, plus the standalone resume smoke. Do not collapse these different code states into one test-count claim.

## 11. Claims to correct and unresolved technical questions

### Claims that were overstated earlier

- Falling reconstruction loss does not prove learning PV signatures, solar physics, or transferable task features.
- Better reconstruction by a larger model does not establish better downstream PV detection.
- Validation loss declining over five checks is not proof of an absence of overfitting.
- Freezing the encoder reduces supervised degrees of freedom; it does not mathematically guarantee no overfitting.
- Attention does not make a model inherently better than directly relevant domain features.
- A presence F1 is not a fraction of all customers correctly classified.
- Trying several models/splits repeatedly on the same small development population does not create a fresh independent test set.

### Source-level discrepancy to investigate before more GPU spending

Observed in current source, not established as the cause of any private result:

- Pretraining `_visible_inputs` uses visible-only median normalization, MAD/std fallback, minimum scale 0.05, and clipping to [-50, 50].
- Downstream `build_windows` uses median/MAD with `max(MAD, 0.001)` and no matching clipping; `raw_customer_features` feeds those values directly to the frozen encoder.

That is a real preprocessing difference, beyond the intentional difference that downstream inference has no artificial target mask. It may produce out-of-distribution magnitudes for flat/low-MAD windows [INFERENCE]. Investigate on fabricated ordinary, flat, sparse, and spike-heavy histories; do not silently change historical checkpoints or benchmarks. No normalization-alignment experiment was performed during this handoff.

### Other plausible limitations, not proven diagnoses

- Reconstruction can learn regular consumption routines instead of PV-specific evidence [INFERENCE].
- Mean/std/last-first aggregation may erase installation-time structure [INFERENCE].
- Four unsupervised PCA directions may prioritize non-PV variation [INFERENCE].
- Domain features may already capture much of the available presence signal [INFERENCE].
- A small labelled population makes both fitting and comparison unstable [INFERENCE].

## 12. What has not been tested

Do not claim completion of:

- Matched random-initialized frozen encoder versus pretrained frozen encoder controls.
- A complete matched scratch-versus-pretrained end-to-end supervised transformer study on the large-unlabelled pretraining path.
- Improved downstream date/capacity models using time-indexed transformer features.
- LoRA, supervised encoder fine-tuning, or a large-model parameter sweep.
- A controlled test of the normalization discrepancy above.
- A post-`a29735c` real-data comparison of the expanded upstream pipeline.
- Independent hidden-customer performance.
- A formal significance claim from the historical split-repeat means.

The last suggested research control was a matched randomly initialized frozen encoder under the same feature extraction, fold-fitted projection, selection, and evaluation protocol. The user asked to merge Jeffre's updates before doing that; **that control has not started**.

## 13. Source map and private artifact locations

### Source to read in a new coding session

| Area | Files |
|---|---|
| Rules and plan | `CLAUDE.md`, `IMPLEMENTATION_HANDOFF.md`, `goals/hybrid-pv-integration/plan.md` |
| Canonical preprocessing | `src/neuralwave_data/prepare.py`, `io.py`, `features.py`, `labels.py`, `pretraining_split.py` |
| Windowing | `src/neuralwave_experiment/windows.py` |
| Streaming/index/sampler | `src/neuralwave_experiment/streaming.py` |
| Adapted backbone | `src/neuralwave_experiment/official_patchtst.py`, `encoder_factory.py`, `vendor/patchtst/` |
| Pretraining loss and loop | `src/neuralwave_experiment/pretraining.py` |
| CLI/checkpoints | `src/neuralwave_experiment/pretrain.py`, `checkpoint.py` |
| Frozen summaries and PCA | `pvdetect/embedding.py` |
| Feature construction and inference | `pvdetect/pipeline.py` |
| Nested option selection | `pvdetect/procedure.py`, `pseudolabel.py`, `selection.py` |
| Upstream unlabelled processing | `pvdetect/unlabelled.py`, `unlabelled_audit.py` |
| Model persistence and entrypoint | `pvdetect/model.py`, `train.py`, `inference.py` |
| Customer folds and scoring | `pvdetect/evaluation.py`, `pvdetect/scoring.py`, `src/neuralwave_experiment/oof_contract.py` |
| Learning curve | `scripts/run_learning_curve.py` |
| Standalone transformer evaluation | `src/neuralwave_experiment/evaluation.py`, `evaluation_models.py`, `temporal.py` |
| Synthetic/regression fixtures | `pvdetect/synthetic.py`, `tests/` |
| Export and hygiene | `scripts/export_submission.sh`, `scripts/check_release.py`, `scripts/check_hygiene.py` |

### Private paths: for data-holder navigation, not agent inspection

- Original challenge inputs: `AIL_data_for_neural_wave_v1.1/challenge_set/`.
- Completed prepared dataset: `data_private/ail-prepared-20261017-v3/`.
- Labelled inputs/splits: its `development/` subdirectory.
- Canonical unlabelled measurements: its `unlabelled/` subdirectory.
- Pretraining `train.parquet`, `validation.parquet`, and manifest: its `pretraining/` subdirectory.
- Historical small model backup: `artifacts_private/real_pretrain_small_v1/`.
- Historical medium model backup: `artifacts_private/real_pretrain_medium_v1/`.
- Initial synthetic GPU evidence: `artifacts_private/cuda-smoke-only-20261017-FrZTkyug/`.
- Completed historical learning curve: `artifacts_private/learning_curve_results_verified.json`.
- Fold-level recovery/predictions: `artifacts_private/learning_curve_results_verified.json.progress.json`.

Do not assume failed earlier `v1`/`v2` preprocessing directories or old comparison artifacts are the final prepared dataset. Do not delete them without authorization.

## 14. Next-session starting instructions

1. Read this document, then the repository rules and source files above. Do not open private artifacts to reconstruct metrics.
2. Inspect current Git state before editing; do not assume no intervening user work or make broad cleanup/staging commands. The prior session left unrelated generated `catboost_info/` untracked; it was not deleted or committed.
3. Do not launch another real-data study, rent a GPU, upload data, or stop/destroy a server automatically.
4. Keep Jeffre's complete pipeline as a comparison arm and preserve exact fold/OOF/scorer contracts.
5. Investigate preprocessing equivalence on synthetic edge cases before concluding that insufficient model size caused a lack of task benefit.
6. If the user chooses a new control, specify one question at a time: random versus pretrained frozen features, or scratch versus pretrained supervised transformer. These are different experiments.
7. Keep outer evaluation customers out of all fitting and automatic choices. Log outer-informed redesign/admission decisions per `DECISIONS.md`; do not adopt a model by selecting the best outer score.
8. Use a fresh output path for changed code/settings. Preserve completed studies. Add per-fold progress/recovery to any new expensive command before running it; do not promise unmeasured runtime.
9. Report only exercised checks. Synthetic execution success is not real-world accuracy, and pretraining reconstruction is not downstream success.

The engineering pipeline is implemented and merged. Its research value still needs the specific controlled evidence above; architecture size and a successful GPU run are not substitutes for that evidence.
