# Hybrid PV integration — execution plan

Status: APPROVED by the user in chat: "i accepted the plannotator plans". Approval provenance is recorded in approval.json; no tool-result payload is fabricated. Two available agents completed scoped designs; Luna5.6 cannot be selected through the available agent tool.

## Goal and scope

Combine the corrected PatchTST implementation with the friend's complete pvdetect pipeline. First hybrid adds exactly four fold-fitted projected frozen-embedding features to presence classification. Existing physical features, commissioning estimator, capacity estimators and output contract stay unchanged. Optional baseline versus hybrid is selected inside existing customer-level inner CV using the complete score. Add a bounded, resumable unlabelled-only CPU/CUDA pretrainer and complete offline artifact packaging. Agents use artificial data only; no real training, cloud rental, upload, model publication, extensions, supervised encoder fine-tuning or giant-model search.

## Fixed inputs and known conflicts

- Friend: `10f44a26d92e1143b8a2d585ca4a9c2bc14d2197`.
- Corrected transformer: `497874548b4e2edf61923e73eac772a301e8e310`.
- Transformer shared contract: `5199b6d80d2f451960243b6d495478120e034758`.
- Main: `2562246a3a1d16cdfbf9d1b1f5bccd9b77b06114`; do not update it in this goal.
- `git merge-tree --write-tree --name-only` found conflicts before any checkout changes:
  - Transformer pair: `benchmarks.md`, `src/neuralwave_experiment/task1_detector.py`.
  - Corrected transformer versus friend: `.gitignore`, `README.md`, `tests/test_evaluation.py`.
- Original workspace contains untracked private/build/source directories. Do not clean, stash, move, stage or copy them into the implementation worktree.

## 1. Parent-only Git integration (serial prerequisite)

1. Create a uniquely named new worktree outside the original workspace, branch `integration/hybrid-pv-v1`, starting at corrected commit4978745. If the branch/path already exists, inspect it; do not reset it.
2. Record source hashes and branch refs in the goal metadata. No reset/force push, branch deletion, hook bypass or source-branch movement.
3. Merge contract5199b6d, then friend10f44a2, with parent-owned explicit resolutions. Stop at unexpected conflict paths and inspect both inputs; never blanket-accept ours/theirs.
4. Keep corrected timestamp persistence and nested threshold calibration. Port split-file acceptance and shared OOF writing into the corrected evaluator, not the retired `train_fold` path. Remove tests/scripts made obsolete by this clean cutover; replace only with consumer-visible behavior regressions.
5. Preserve aggregate-only corrected benchmark status, friend's evidence/privacy rules and all licenses. README documents both the baseline and opt-in hybrid. Rename the independent test modules to unique names so one add/add resolution does not silently discard a suite.
6. Union ignore protections for raw inputs, both artifact conventions, private data, Python caches, build outputs and model weights. Never ignore the source package itself. Stage explicit named source/config/test/doc files only.
7. Freeze compatible dependency locks using the already exercised NumPy1.26/pandas2.2 compatibility line and a supported pinned Torch/CatBoost environment. Keep CPU baseline imports Torch-free; provide a separate documented CUDA installation path. Do not copy the incompatible NumPy2/pandas3 lock over transformer constraints.

Check: source branch refs unchanged; integration refs contain both histories; no unresolved merge entries; original private paths not staged. Run one integrated baseline suite after serial resolution, not per agent edit. Create a named integration checkpoint commit only after tests/hygiene pass. Do not push or merge into main without an explicit release decision.

## 2. Freeze interfaces before edit agents (parent)

Reuse `neuralwave_experiment` encoder, vendored backbone and window contracts; do not fork these into parallel `pvdetect/patchtst_*` implementations.

- `neuralwave_experiment/streaming.py`: bounded canonical source/index, deterministic customer/season sampling, window batches, sampler state. Explicit source roles and labelled/evaluation exclusion IDs. Train/validation unlabelled customer sets disjoint.
- `neuralwave_experiment/checkpoint.py`: schema-versioned inference artifact and private training-state save/load. Inference artifact is `encoder.pt` tensor state plus `encoder.json` metadata. Include configuration and hashes; no customer IDs, labels, predictions or training-state arrays in released metadata. Private full training checkpoints may contain runtime sampler/provenance state and remain private.
- `neuralwave_experiment/pretrain.py`: actual CLI with `--manifest`, `--exclude-labels`, `--out`, `--config`, `--device cpu|cuda`, `--resume`. Exclusion label file is read only for IDs, never supervision. Resume writes a new checkpoint generation rather than overwriting completed evidence silently.
- `pvdetect/embedding.py`: `FrozenEncoder.load(reference, device='cpu')`, `raw_customer_features(table, ids)`, `fit_projection(raw, train_ids, n_components=4)`, `transform_projection(raw, ids, projection)`. Exactly four embedding columns; no additional magnitude block confounding the first representation ablation. Existing magnitude features remain intact.
- `pvdetect.procedure.fit(..., encoder=None, feature_block='auto')`: absent checkpoint means baseline; a supplied invalid checkpoint is an error, not a silent baseline fallback. Shared inner splits evaluate baseline and embedding. Projection failure makes the optional candidate explicitly unavailable for that fit, with a recorded reason.
- Saved model schema bumps cleanly for all final fits; no legacy-load compatibility shim. Previously generated model artifacts require refitting. Baseline predictive invariance is checked against pinned baseline fixtures/code, not by preserving the old artifact schema.

## 3. Implementation ownership and parallel waves

One owner per file. Parent owns Git, dependency files, integration tests, shared contracts, docs and final verification. Agents never run Git mutation, install, build, lint, tests or formatters mid-flight.

### Agent A — streaming pretraining and checkpoints

Own only `neuralwave_experiment/streaming.py`, `checkpoint.py`, `pretrain.py`, necessary scoped changes in `pretraining.py`, `configs/pretraining_unlabelled.json`, and uniquely named streaming/checkpoint tests.

- Read canonical epoch Parquet in bounded record batches; build a private per-customer shard/index cache in one bounded pass, then read selected customer arrays without rescanning the full source for every window. Cache records input identity and preprocessing version. No dense all-history calendar/window tensors.
- Sample customers approximately evenly, rotate through available seasons/time spans, and use a frozen windows/update budget. Training sampling caps never apply to inference/temporal evaluation.
- Draw fresh observed-patch masks each step; leave visible observations. Restore physical values before visible-only median/MAD normalization; hidden and originally missing values cannot supply visible statistics or reconstruction targets.
- Default encoder: width64, layers2, heads4, patch16, context672, dropout0.2; default model roughly110k. Config can express fixed width128/layers3/heads8 candidate later, but no selection sweep now. No new cosine/warmup schedule is necessary: keep Adam lr0.001, batch64, mask0.3, clip5, five-epoch readiness configuration; larger real-training budgets require a separate frozen config. Validation loss is computed on separate unlabelled customers with fixed validation masks.
- CPU path runs without AMP. CUDA supports explicit fp32/fp16/bf16 only when supported; unscale before clipping for fp16. No unsupported-device fallback that pretends CUDA ran.
- Save model, optimizer, optional scaler, Python/NumPy/Torch CPU/CUDA RNG, sampling/mask RNG, epoch/step cursor and config/input identities atomically. Exact CPU resume reproduces the next batch/mask/update. Do not claim cross-hardware CUDA bitwise equality.
- Emit a compact inference artifact separately from the private training checkpoint.

### Agent B — frozen feature option and packaged inference

Own only `pvdetect/embedding.py`, `features.py`, `pipeline.py`, `procedure.py`, `model.py`, `train.py`, `inference.py`, `scripts/export_submission.sh`, `scripts/check_release.py`, and uniquely named hybrid/release tests.

- Derive raw epoch arrays from existing parsed input; keep fixed UTC+1 physical features and Zurich calendar encoder features explicit. Preserve string IDs, negative readings, missing masks and absolute units.
- Aggregate frozen window embeddings in chronological order (mean/std/first-last), then fit imputation, scaling and deterministic four-component projection on the current training customers only. Projection/candidate state is never stored in a global label-free cache.
- Cache raw embeddings only by validated frozen checkpoint+input-contract hash. The encoder must have unlabelled-only provenance with no overlap; metadata assertions alone are not proof, so training exclusion audits remain required.
- Evaluate exactly baseline and baseline+embedding on the same inner splits. Reuse existing2-point margin and0.8 inner bootstrap-stability conventions; compute candidate full scores from each candidate's actual date/capacity/presence predictions. Do not compare post-hoc outer scores to adopt a feature block.
- Refit the selected projection/classifier on outer training only. Keep baseline date/capacity code and feature matrices unchanged; presence gating can still change which customers receive outputs.
- Persist selected option, projection arrays, checkpoint config and exact artifact hashes. A baseline-selected model must not require Torch or copy an unused encoder into its submission.
- Extend `referenced_files` with exact encoder files when selected. Reject unsafe/traversal/absolute references and hash mismatches. Export only those exact files; no broad .pt allowlist, no optimizer-state release. Inference never fits a transform or accesses a network.

### Parent — comparison contracts and integration

Own `neuralwave_experiment/evaluation.py`, `oof_contract.py`, detector cutover, shared evaluation tests, dependency configs, README/benchmark status and goal docs. Fix strict `(repeat, customer_id)->fold` matching, exact customer coverage, supplied split-file handling, and standardized OOF export. Keep classifier-only transformer D/P limitations separate from complete hybrid scoring. Wire owner APIs only after their implementations are published.

## 4. Verification gates

Run checks once after each complete integration wave; no holdout benchmark suite and no real AIL commands.

1. Baseline invariance: pinned artificial input gives the same baseline feature values, probabilities, regimes, dates and capacities without a checkpoint.
2. Exclusion: overlapping pretraining/validation/labelled IDs rejected; shuffled held-out labels cannot change that fold's predictions. Projection trained only on inner/outer training IDs. All candidates use the same splits.
3. Stream/mask: bounded fake-source batches, DST/epoch alignment, missingness, customer/season sampling, changing masks, hidden-target normalizer non-interference.
4. Resume: interrupted versus uninterrupted CPU run has identical next sample/mask and final parameters. Config/input/corrupt checkpoint mismatches fail clearly.
5. Shared scoring: mismatched folds/repeat coverage are rejected. Known scoring boundaries preserved; no missing conditional target is dropped from the mean.
6. Artifacts: selected hybrid reloads within floating tolerance; unused encoder excluded from baseline release; exact-reference/hash/traversal checks; offline inference blocks socket creation in the artificial smoke.
7. Actual CLI smoke: independently fabricated disjoint unlabelled and labelled customer sets; pretrain, interrupt/resume, run shared-fold baseline and auto-hybrid training, export the saved selected procedure, invoke `inference.py` and inspect both output CSVs. If auto selects baseline, separately exercise a real frozen-encoder diagnostic path on artificial data without claiming its score validates the hybrid.
8. Commands in isolated worktree: `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 <verified-python> -m pytest -q tests`; `python scripts/check_hygiene.py --all`; actual CLI smoke and `scripts/check_release.py` with exact references. Pin the interpreter after dependency verification. Report only executed checks.
9. GPU gate: CPU environment currently has no CUDA. Supply a user-run artificial CUDA forward/backward/AMP/resume/VRAM-throughput smoke; label it unverified until run. No claim that CPU tests establish CUDA performance or real accuracy.

## 5. Delivery and stop conditions

Deliver an integration branch with reviewed commits, frozen configs, architecture/feature/checkpoint metadata, usage and experiment commands, and a complete synthetic verification record. No source branches are overwritten and no main push occurs automatically. Data holder then obtains hosting approval, performs CUDA smoke, and runs pretraining/real evaluation. Large-model experiments are a separate decision based on downstream transfer, not reconstruction loss alone.

Approval is recorded from the user's statement. Implementation proceeds in /tmp/ch-tokenlarpers-hybrid-pv-v1 on integration/hybrid-pv-v1. The parent resolved the previewed conflicts, plus additional hygiene conflicts introduced by combining all three histories. Slice design artifacts at local://hybrid-pretrain-slice.md and local://hybrid-model-slice.md are inputs; this parent plan overrides their duplicate-module, extra-magnitude, legacy-shim and differing-default proposals.
