# Evaluation protocol v2: nested CV + sealed lock-box + full refit

Status: plan. Builds on `docs/evaluation_protocol.md` (main) and the `pvdetect` nested CV; adds one sealed hold-out and a final refit. Standard references: nested CV for unbiased selection (Varma & Simon 2006), selection bias from repeated model comparison (Cawley & Talbot 2010), untouched hold-out = private leaderboard / lock-box.

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

## Code changes needed

1. `pvdetect/lockbox.py` + `scripts/lockbox.sh` with subcommands `seal | develop NAME [TRAIN_FLAGS] | freeze | open | final`, state machine in `PRIVATE_DIR/state.json`, refuses out-of-order stages and second opening.
2. Stratified split (current `split_by_customer` is unstratified: a random 10 of 40 can contain 3 or 8 PV customers). New `split_by_customer(..., labels=..., strata=...)`.
3. Fold files for the 30 dev customers generated once at seal time (`--folds` already supported by `train.py` and main's evaluator), so pvdetect options and hybrid arms are compared on identical customer assignments.
4. `ci90` over all repeats (currently repeat 0 only).
5. `pvdetect.stability MODEL_A MODEL_B UNLABELLED`: predicted PV share and customer flips on the unlabelled set (label-free robustness check).
6. GUI: lock-box result panel read from `lockbox_result.json` (aggregates only), marked "single look".

## Success / stop rules (pre-registered in stage 0)

- Adopt a candidate in stage 2 only if `pvdetect.compare` verdict = `improvement` on all three split seeds pooled; else keep baseline.
- After freeze, the lock-box result changes nothing; if it falls below the nested 90 % interval, report "possible overfitting" and still submit the frozen model (no re-tuning on the lock-box).
- Budget exhausted → freeze.
