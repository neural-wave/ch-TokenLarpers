# Evaluation protocol v2: nested CV + sealed lock-box + full refit

Status: implemented (synthetic-verified; real-data stages are run by the data holder). Commands: README, "Lock-box protocol (data holder)". Builds on `docs/evaluation_protocol.md` (main) and the `pvdetect` nested CV; adds one sealed hold-out and a final refit. Standard references: nested CV for unbiased selection (Varma & Simon 2006), selection bias from repeated model comparison (Cawley & Talbot 2010), untouched hold-out = private leaderboard / lock-box.

## Stages (one state file per private run directory; each stage refuses to run out of order)

| Stage | Data | What happens | Output (private) |
|---|---|---|---|
| 0. Pre-register | none | Commit `configs/lockbox.json`: lock-box seed, fraction (10 of 40), strata, outer split seeds (17, 42, 100 as on main), decision budget (≤ 3 Tier B comparisons), success rules. Commit before any data is touched. | — |
| 1. Seal | labels + energy | Stratified lock-box split (PV/non-PV; within PV by install half-year and present-at-start where possible) → `dev.csv` (30), `lockbox.csv` (10). Write `seal.json`: sha256 of the sorted lock-box id list and of `lockbox.csv`, config hash, timestamp. Shared outer split files for the 30 (3 seeds) are generated once and reused by every run (pvdetect and hybrid arms). | `dev.csv`, `lockbox.csv`, `seal.json`, `splits_s{17,42,100}.csv` |
| 2. Develop | dev (30) | Every candidate (option set, hybrid feature block) runs `train.py --cv --folds <the 3 split files> --oof-out`. Comparisons only via `pvdetect.compare` on identical splits; each comparison appends one DECISIONS line and decrements the budget. Diagnosis/GUI on the 30 only. | `runs/<name>/oof.csv`, `diagnosis.json`, compare reports |
| 3. Freeze | — | Tag the commit (`freeze-vN`); record procedure config and model-options hash in the state file. No code change after this stage. | `freeze.json` |
| 4. Open lock-box (once) | lockbox (10) | Fit the frozen procedure on all 30 dev customers, predict the 10, score with the official scorer + diagnosis, paired bootstrap CI (customers resampled), compare with the nested estimate on the 30. Opening twice refuses unless `--force`, which is recorded. | `lockbox_result.json` (aggregates), predictions private |
| 5. Final refit | all 40 | Same frozen procedure, inner selection on 40; export with `scripts/export_submission.sh`; offline smoke. | submission model |
| 6. Report | — | Nested estimate on the 30 (selection-biased, k decisions), lock-box score with CI (n = 10, single look), overfitting check (lock-box inside nested CI?), label-free stability on the unlabelled set (predicted PV share, flips vs previous model). | report table |

## Code changes (implemented)

1. `pvdetect/lockbox.py` + `scripts/lockbox.sh` with subcommands `seal | develop NAME [-- TRAIN_FLAGS] | compare A B | freeze NAME | open | final | status`, state machine in `PRIVATE_DIR/state.json`, refuses out-of-order stages, a tampered lock-box and a second opening (`--force` is recorded).
2. Stratified split: `split_by_customer(..., labels=..., n_test=...)` allocates the test customers per `pv_installed` class (largest remainder, ≥1 per class on each side when possible). Only `pv_installed` is a stratum (`configs/lockbox.json`); install half-year and present-at-start are not stratified.
3. Fold files for the dev customers generated once at seal time (`evaluation.stratified_fold_map`, the same construction as `neuralwave_data.labels.make_splits`), so pvdetect options and hybrid arms are compared on identical customer assignments.
4. `ci90` pools all repeats: each draw resamples the customers once and averages every repeat's score.
5. `pvdetect.stability MODEL_A MODEL_B UNLABELLED`: predicted PV share and customer flips on the unlabelled pool (label-free robustness check).
6. GUI: lock-box result card read from `lockbox_result.json` (aggregates only), marked "single look".

Implementation notes: the tool never writes to the repository, so `compare` prints the `DECISIONS.md` line to add and `freeze` prints the `git tag freeze-vN` command; `open` predicts with the frozen run's `model.json`, which `train.py` already fits on all dev customers after its nested CV; `seal.json` also hashes `lockbox_labels.csv`; the GUI result and `state.json` record forced overrides and a code change after freeze.

## Success / stop rules (pre-registered in stage 0)

- Adopt a candidate in stage 2 only if `pvdetect.compare` verdict = `improvement` on all three split seeds pooled; else keep baseline.
- After freeze, the lock-box result changes nothing; if it falls below the nested 90 % interval, report "possible overfitting" and still submit the frozen model (no re-tuning on the lock-box).
- Budget exhausted → freeze.

## Round 2: improvement loop on the same seal (pre-registered before any round-2 run)

`python -m pvdetect.lockbox round --dir <private>/round2 --from <private>/lockbox` starts a new development round on the round-1 seal: the same 30 dev and 10 lock-box customers and the same three split files (hashes verified, files hardlinked). Config `configs/lockbox_round2.json` changes only `decision_budget` (10); `round` refuses any change to the split keys.

Loop (each iteration):
1. **Baseline** — `develop` the current `main` (defaults) on the dev set; `pvdetect.diagnose` names the component that loses the most points.
2. **Error review (dev out-of-fold only)** — the data holder inspects the dev customers that lose those points (per-customer errors, plots) and writes mechanism descriptions without customer identifiers or values.
3. **Candidates** — parallel workers implement one mechanism each as a Tier A option (inner CV decides per fit), on synthetic dev scenarios that reproduce the mechanism, with tests; each on its own branch.
4. **Comparison** — each candidate branch is `develop`ed on the same split files and `compare`d with the iteration's baseline (one decision each, one `DECISIONS.md` line each). A candidate is merged into `main` only on verdict `improvement`; `within_noise` and `worse` are not merged. A candidate built for one component is compared with `--target <component>`: `improvement` needs that component's points-lost reduction interval above 0 and a non-inferior total score (point estimate ≥ 0, lower bound ≥ −3 points); this rule was added after the first two round-2 comparisons and is logged as such. The merged state is the next iteration's baseline.
5. Stop when the budget is spent or at the freeze time.

Then `freeze` the last baseline, `open` once (look 2: no round-2 decision used look 1, which changed nothing), `final` on all 40, export, offline smoke. The lock-box customers are never inspected individually in any round. Every round-2 comparison is outer-informed, so the nested estimate is reported as selection-biased with its decision count; the lock-box look is the unbiased check of the final procedure.
