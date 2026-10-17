# Frozen evaluation protocol

Use `python -m neuralwave_experiment.evaluation`, not the retired Task 1 detector/stress-test scripts. Architecture tuning is paused. `configs/evaluation.json` freezes one comparison, not a search space. The historical embedding/backend experiment remains available for research, but its results are not this protocol's results.

## Run

```bash
python -m pip install -e '.[experiment]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m neuralwave_experiment.evaluation \
  --canonical-dir data_private/patchtst_comparison_canonical \
  --out artifacts_private/evaluation_v1 \
  --config configs/evaluation.json \
  --data-kind development
```

Run only where local data use is authorized. Data and derived artifacts must remain private. Output must be new/empty; populated directories are rejected. Canonical inputs require `measurements.parquet` and `labels.parquet`. Existing `splits.parquet` is intentionally not reused: every outer split is regenerated and persisted. Customer IDs are strings, never numeric features. A customer without eligible windows is an explicit error, not silently omitted.

`--data-kind artificial` explicitly marks smoke results. Development CV cannot be designated blind evaluation. The manifest states that no blind evaluation was performed. The organizer's 20 hidden cases/labels are not supplied to this evaluator; the 1148 unlabelled histories cannot establish accuracy.

## Matched arms and budgets

1. `hourly_logistic`: 24 hourly means and 24 hourly standard deviations of normalized observed values; training-only feature standardization, logistic regression with fixed C=1 and inverse class prevalence weights.
2. `patchtst_scratch`: official-backbone adapter initialized randomly.
3. `patchtst_pretrained`: the same model initialization, preceded by masked reconstruction on that fit's training customers only.

All arms use seven-elapsed-day windows (672 quarter-hours), daily stride, uncapped temporal coverage, local median/MAD normalization, the same physical window labels, customer folds, persistence rule, and threshold grid. Non-PV windows and windows centered more than seven days before installation are negative; windows centered more than seven days after installation are positive; transition windows are excluded from supervised loss. Missing installation dates on positive customers retain the existing pre-existing-PV training convention; date diagnostics are censored.

PatchTST remains d_model=64, two layers, four heads, patch length 16, dropout 0.2. Both transformer arms have six supervised AdamW epochs, lr 0.001, weight decay 0.01, batch 64, positive weighting, and gradient clipping 2. Initialization, supervised shuffle RNG, and dropout RNG are matched. The pretrained arm receives five additional reconstruction epochs with mask ratio 0.3. This extra compute is disclosed: the ablation tests pretraining under an equal supervised budget, not equal total compute. There is no architecture/optimizer search for any arm. Both physical training classes are required.

Masked reconstruction restores physical values using the saved per-window affine statistics, then computes its normalization from visible observations only. Hidden targets therefore do not supply the normalizer. Missing values never enter the reconstruction loss; at least one observed patch must remain visible. Pretraining normalization is intentionally different from the ordinary inference-window normalization.

## Time-based persistence

Each probability is weekly context evidence at its epoch midpoint, representing at most one following elapsed day. A candidate block spans 14*86400 seconds and requires at least 90% observed-weighted support and no inter-center gap over two days. Observation fractions come from each window's observed mask. Overlapping daily support is unioned, not counted twice; duplicate UTC-day buckets are collapsed. No cap/subsampling may turn sparse selected windows into continuous evidence.

The block score is its median probability. The customer score is the maximum eligible block score. Records without eligible support score 0 with `coverage_eligible=false`; there is no short-record maximum fallback. The earliest block reaching the selected threshold supplies a retrospective onset estimate at its first center and a separate confirmation date at the represented support end. These are offline estimates using centered weekly windows, not a causal real-time alarm. Median persistence is not a promise to reject every vacation or arbitrary load drop.

## Nested customer CV

- Three outer repetitions: split seeds 17, 42, 100; five stratified customer folds each. Split membership must genuinely differ, even ignoring fold-number permutations. Model seed stays fixed so repetitions isolate customer assignments.
- Within each outer training set: four stratified customer folds. Fit each arm on inner training customers; generate one inner-OOF customer score for each outer-training customer.
- Select the threshold maximizing pooled inner-OOF customer F1 over 0.20–0.80 in 0.01 steps. Lowest threshold breaks ties. Each arm has the same candidates and opportunities.
- Refit the arm on all outer-training customers. Apply its threshold unchanged to outer validation. No outer scores/labels select thresholds, epochs, aggregation, or models.
- Pool each repetition separately, exactly one outer prediction/customer/arm. Across-repetition means/std are descriptive: repeated customers are not independent observations. Do not select the luckiest seed as a headline.

Repeated CV does not erase past development-set tuning. A final generalization claim needs fresh blinded customer evaluation with the frozen pipeline.

## Temporal diagnostics

Every outer customer row retains score, threshold, presence decision, retrospective onset, confirmation, and coverage eligibility. Label-dependent diagnostics run only after predictions:

- Pre-installation false alarms: rerun persistence only on windows ending more than seven days before true installation; insufficient support is null, not a successful rejection.
- Signed/absolute onset-day error and date reward max(0,1-abs(error)/30). Missing predictions on a known-date PV customer score 0. Missing truth is censored and counted separately.
- Known commissioning dates outside recorded support are flagged, not silently discarded.
- Late installations: true commissioning within the final 60 elapsed days of raw window support; report count and detected count.
- Non-PV false alarms and insufficient-persistence-coverage counts.

Date-error medians/hit fractions are conditional on an available date prediction; missing prediction counts and all-PV date reward accompany them. `date_reward_detected_pv` uses the official Task 2 conditioning. The all-PV reward is a component diagnostic. Initial-capacity regression and the overall challenge score are deliberately not evaluated by this presence/temporal protocol. Do not attach old capacity rewards to the new detector's F1.

## Retained evidence

- `config.json`, `manifest.json`: complete settings, config/input hashes, package versions, budget and data designation, explicit no-blind-claim status.
- `splits.csv`: every outer and inner training/validation membership.
- `predictions.csv`: one customer prediction per arm/repetition plus temporal diagnostics.
- `metrics.json`: pooled metrics per repetition and descriptive summaries, not a fabricated blind score.
- `repeat_R/fold_F/ARM/inner_I/` and `outer/`: all window traces, model/scaler artifacts, histories, configuration, and training IDs/lineage.
- `inner_oof.csv`, `calibration.json`: inner customer scores, every candidate's metrics, and the selected threshold.

Old four-entry persistence, in-sample threshold calibration, hardcoded cached-seed evaluation, training-overlapping synthetic evaluation, and hardcoded-threshold unlabelled scripts were removed. Historical result files were not overwritten. Physical load-profile inspection remains exploratory, not independent confirmation of tariffs or hardware.

## Verification

Implementation verification used wholly artificial data. The full suite passed 47 tests, including hidden-target normalization leakage regressions. An actual CLI run covered 12 artificial customers over 120 days, two genuinely different outer partitions, two inner folds, and all three arms. It retained 72 outer customer rows and 36 fit lineages/histories. All inner thresholds were independently recomputed; training/evaluation ancestry was disjoint. Twelve outer model artifacts were reloaded and reproduced saved probabilities within 1e-6. A populated output directory was rejected. Smoke scores are not evidence of AIL accuracy or superiority.
