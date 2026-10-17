# AIL PV detection — project overview

*Status as of 2026-10-17. Authoritative plan: `IMPLEMENTATION_HANDOFF.md` (v3). Binding rules: `CLAUDE.md`.*

## Goal and constraints

Detect PV installations, their commissioning date and nominal kWp from 15-minute grid-withdrawal series alone. Scoring is F1 times a weighted sum of detection, date accuracy and capacity accuracy, with a small bonus for extensions.

Data: 40 labelled customers for development, 20 hidden customers for the final score, and 1148 unlabelled series with unknown PV status. No weather, location or irradiation data; no LLM in the pipeline; inference runs offline. AIL data never reaches a coding agent: all agent work runs on synthetic data, and real-data commands are run by a team member with data access.

## Approach

Knowledge lives in hand-built physics features:

- a season-aware, deseasonalized daytime signal;
- exact-zero clipping statistics (import-only meters clip PV surplus to zero);
- a change-point detector with a year-over-year recurrence check;
- capacity from the midday pre-vs-post dip and a physics kWp estimate from the pre-install midday level and the width of the zero-reading window around solar noon.

The learned part is small: a logistic or CatBoost classifier on 21 features, a decision threshold, a ridge or CatBoost capacity model and one scale factor. All of these are chosen by inner CV inside a single fit procedure (`Procedure.fit`). Nested CV on the 40 labelled customers is the only real evidence, and every human decision informed by outer results gets a line in `DECISIONS.md`.

## Status

| Phase | Content | State |
|---|---|---|
| P1 | Ingestion, DST, scoring, nested-CV harness | committed |
| P2 | Season-aware detection, date regimes, confounder features, benchmark suites | committed, gate logged |
| P3 | Real-data nested CV and error review | **not run yet** (team member with data access) |
| P4 | Extensions and bottleneck fixes | after P3 |
| P5 | Unlabelled tooling, gated augmentation | after P3, needs organizer approval |
| P6 | tsfresh, TS2Vec, CNN options | gated |

The P2 synthetic gate benchmark scored between 77 and 88 across eight stress scenarios (`benchmarks/LOG.md`). These are synthetic numbers and say nothing about real accuracy.

Environment: `.venv` with the pinned requirements; `python -m pytest -q` passes (43 tests). A full nested-CV run on a 40-customer, four-year dataset takes about one minute on an Apple M3 Pro CPU with under 1 GB of memory. No GPU is needed for P1–P5.

## Decisions on synthetic and unlabelled data

Synthetic data has three uses, treated differently:

1. **Tests, regression checks, workload sizing.** Always fine; this is what `pvdetect/synthetic.py` exists for.
2. **Augmentation inside each fit** (plan Step 19): real training-fold load with physics PV injected, plus synthetic hard negatives, grouped with their source customer so no inner split sees both. Adopted only if P3 points to it and the organizers allow it.
3. **A standalone synthetic training set replacing the 40.** Rejected. It would calibrate the detector to our simulator rather than to AIL's meters and labelling practice. The overnight "synthetic data factory" brief was set aside for this reason; its label-leak checks and paired-counterfactual idea are kept as small generator improvements.

The 1148 unlabelled series are used only for label-free purposes (climatology options, spread, pretraining). They are never treated as PV-negative and never used as donors for labelled synthetic customers.

## Next steps

1. Run P3 (`train.py --cv` on the real 40) and keep only the aggregate printed lines; the out-of-fold table stays private.
2. Ask the organizers whether semi-synthetic augmentation of the provided series is allowed in the competitive model.
3. Read three numbers from P3: the inner-vs-outer gap, the weakest of F1 / date / capacity, and date error per regime. Each conclusion becomes a `DECISIONS.md` line.
4. Depending on the answers, implement Step 19 augmentation and the generator improvements.
5. The holdout benchmark suite stays untouched until the P4 gate.

## Known risks

- The hidden set is 20 customers, so the final score moves about 5 points per flipped positive and will be noisy regardless of the model.
- Labelled commissioning dates can lag the physical onset by months; only the real labels can teach this.
- Capacity is the hardest term because clipping hides most production.
- P3 may reveal a gap between the synthetic benchmark and real behaviour. That gap, not more synthetic data, decides the next design step.
