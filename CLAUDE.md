# Claude Code operating instructions — AIL PV Detection

Read `IMPLEMENTATION_HANDOFF.md` first. It is the authoritative implementation plan for this codebase.

## Data handling

- Agent work, tests and fixtures use synthetic data from `pvdetect/synthetic.py` only. Real-data commands in `IMPLEMENTATION_HANDOFF.md` (P3, P7) are run by the user.
- No LLM may be in the inference/training pipeline; no external weather, irradiation or geographical data. All inference must run offline, without network calls. Use timestamp-derived calendar features only.
- Synthetic PV augmentation in competitive training only after organizer approval (Step 19 gate).
- The hygiene hook must be installed (`sh scripts/install_hooks.sh`) and `python scripts/check_hygiene.py --all` must pass before every commit: no data or model dumps, nothing under artifacts directories, no real customer identifiers (`POD` followed by digits) in any file, message or report.
- Explicit user-approved model publication exception: the small/medium `pretrained/*/encoder.pt` and `encoder.json` pairs may be committed and used by agents on fabricated inputs. Only the exact four SHA-256-pinned files in `scripts/check_hygiene.py` qualify; this does not allow datasets, private metrics, optimizer states, other model dumps, or AIL inputs in agent sessions.

## Selection and evidence

### Selection policy (binding)

- **Tier A, automatic choices:** classifier kind, threshold, capacity kind, κ, date offsets, regime parameters, climatology source, extra feature blocks, date source.
  - All of these happen **only** inside `Procedure.fit`, chosen by inner CV on that fit's training customers.
  - The outer nested-CV score is an honest estimate of the *whole procedure*, including its option pool.
  - The final model is the same procedure fit on all 40.
- **Tier B, design decisions informed by outer results:** P3 error review, feature redesign, gate decisions, changing an option pool, any comparison of two nested runs.
  - These are allowed, but each one appends one line to `DECISIONS.md`: `- YYYY-MM-DD | decision | outer results consulted`.
  - `train.py --cv` counts these lines and prints `outer_informed_decisions=k`. When `k > 0` it labels the estimate `selection-biased`.
  - No candidate is ever *adopted* by comparing outer scores. Unlocking a candidate (Tier B) only adds it to the inner option pool; inner CV then decides per fit.
- **Bootstrap `P(diff > 0)`:** used only inside inner selection, as a stability heuristic alongside the margin. It is never computed on outer predictions and never cited as validation.
- **Outer CV is computed once per code state for reporting.** Rerunning it to compare variants is Tier B.
- **Feature cache rule:** `FeatureTable` and any other cross-fold cache hold **label-free** quantities only (raw signals, `s`, `mid`, `night`, `eve`, `zero`, `early_contrast`, label-free tsfresh kinds, TS2Vec embeddings, CNN windows). Anything that depends on a fitted artifact (climatology, `ds`, `yield_shape`, regime cut, PCA, feature selection, scalers) is computed inside `fit`/`predict` with that fit's artifact. Enforced by the label-flip invariant test (P2 verification): changing a held-out customer's label must not change any prediction in its outer fold.
- **Common inner splits:** every Tier A option, from logreg to the CNN, is scored on the same `inner_splits` customer folds of a given fit. No option may use its own fold count, and no split may separate one customer's windows or augmented descendants.

### Evidence tiers (binding)

|Tier|What|Where|Allowed use|
|---|---|---|---|
|U, unit|Deterministic contract and algorithm checks on hand-built arrays or tiny fixtures|`tests/`|Exact asserts; always run|
|R, synthetic regression|Fixed **dev** seeds (< 1000), generator variant A|`tests/`|Catch regressions; never evidence of accuracy; tolerances fixed in this plan|
|G, synthetic generalization|Generator variant B (independent PV, cloud and load families, plus confounder kinds absent from A), seed block from `benchmarks/HOLDOUT_SEED_BLOCK` (starts at 10000)|`python -m pvdetect.benchmark --suite holdout --log benchmarks/LOG.md`|Run only at phase gates (end of P2, P4, P7). Results are appended with the git SHA. A code change made in response burns the block (+10000) and adds a `DECISIONS.md` line|
|E, real|Nested CV on the 40 dev customers|P3 and P7 (user)|Primary evidence|

Never run the holdout suite outside a phase gate.

## What exists

P1–P2 are implemented per `IMPLEMENTATION_HANDOFF.md` (tags `submit-v0`, `submit-v1`); nothing is competition-validated.

## Your immediate task

B4, X2 and D-pub wording are done (v3.3). Next work in `IMPLEMENTATION_HANDOFF.md`: C3 belongs to the `oof_contract` owner; the team decides the D-pub items marked in `report/REPORT.md`; E only for the component the private real-data results name, one challenger at a time, under the admission criteria; F only if time remains after G is safe. Never place a real-data model under this repository; real models reach the NW_26 repo only via `scripts/export_submission.sh`. Never put real-customer plots or per-customer values into the report or video. Commit each step separately, keep `pytest -q` and `python scripts/check_hygiene.py --all` green, then STOP and report changed files, tests and risks.

## Exact submission contract

`python inference.py path/to/test.csv` -> working-directory `predictions.csv` with `customer_id;pv_installed;pv_installation_date;pv_capacity_kwp`, and `pv_extensions_predictions.csv` with `customer_id;extension_date;pv_capacity_after_extension_kwp`. UTF-8, semicolon, TRUE/FALSE, YYYY-MM-DD, missing date/capacity when FALSE, customer IDs as strings preserving leading zeros. Header-only extension output is valid for MVP. File examples in the PDF use commas despite its explicit semicolon instruction; follow section 4.1 pending clarification.

## Engineering bar

Customer-level splits only; no validation leakage via variants, fit transforms, tuning or TS2Vec; no unlabeled-as-negative assumption; synthetic datasets are tests, not proof of competition accuracy. Develop minimal, deterministic, tested changes; no real data in prompts, logs, patches, public repositories, screenshots or reports.
