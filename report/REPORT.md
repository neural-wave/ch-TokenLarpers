# Detecting PV installations from grid-withdrawal profiles

AIL NeuralWave challenge. Team: Alessandro Tognoni, Jeffre Daniel Martinez Marroquin, Tarun Deepak.

Outline, at most 3 pages.

## Approach

A rooftop PV system lowers what a household draws from the grid on sunny daytime hours. We look for that signature in 15-minute withdrawal series. We use no weather, irradiation or location data, only the timestamps.

Most of the knowledge sits in hand-built features:

- a deseasonalized daytime signal, so that summer does not look like a PV start;
- exact-zero statistics, because import-only meters clip PV surplus to zero;
- a changepoint detector with a year-over-year recurrence check;
- confounder features for night loads, batteries and commercial profiles;
- capacity from the midday pre-vs-post dip and a physics kWp estimate from the pre-install midday level and the width of the zero-reading window around solar noon.

The learned part is small: a logistic or CatBoost classifier, a decision threshold, a ridge or CatBoost capacity model, a scale factor κ and a date rule. Dates come from one of three regimes: a visible event, PV already present at the start of the series, or too little data.

All of these choices are made by inner cross-validation inside one fit procedure. Nested cross-validation on the 40 labelled customers scores the whole procedure, option pool included. The final model is the same procedure fit on all 40.

## Results

Filled after the real-data run (P3) from aggregate lines only: `nested:`, `dates:`, `selection:` and the shared-fold comparison against the other branches. No per-customer values.

`outer_informed_decisions=1`, so the estimate is labelled selection-biased.

How to read the numbers. Every number we report rests on 40 labelled customers. Cutting their histories into thousands of weekly windows does not create thousands of independent cases. The spread across CV repeats shows how much the result depends on the particular split of those 40 customers; it is not new data. The bootstrap interval resamples fixed out-of-fold predictions and leaves out the uncertainty of model selection. Capacity cannot be read off directly from grid withdrawal: load, orientation, shading, inverter limits and batteries can produce the same readings, so kWp is an estimate.

<!-- figures: synthetic and labelled "synthetic customer" only; real data only as aggregate lines, and only if the organizers allow (Step 2.5-D-pub). No real-customer plots, traces, dates or per-customer predictions. -->

Synthetic holdout benchmark (P2 gate, 40 customers, 2 repeats, code `3d2e5cd`, before the Phase 2.5 fixes). These numbers are synthetic and are not evidence of competition accuracy:

| scenario | score |
|---|---|
| base | 83.6 |
| confounders x2 | 81.3 |
| east/west heavy | 84.2 |
| battery heavy | 87.9 |
| short coverage | 77.5 |
| 5% missing | 84.4 |
| commercial heavy | 81.5 |
| PV before start | 81.2 |

## Challenges

- The raw changepoint fired on seasonal drops: spring looks like a new PV system. On synthetic data, deseasonalizing and a year-over-year recurrence check removed most of these false detections.
- The labelled commissioning date may lag the onset that is visible in the data. We do not assume a fixed lag; an offset is learned from the training customers inside each fit.
- Night loads such as heat pumps and storage heaters shift consumption in ways that mimic PV. Dedicated confounder features separate them on synthetic data; they do not solve it in general.
- With 40 labelled customers, any choice made by looking at results overfits. We moved all tuning inside the fit and log every outer-informed decision.
- Different branches scored on different splits, so numbers were not comparable. We fixed shared customer folds and a common out-of-fold format.
<!-- team: keep or drop the next item (Step 2.5-D-pub); if dropped, keep only the controls. -->
- Data hygiene: a benchmark report on one branch exposed customer identifiers and per-customer values. We scrubbed it at the branch tip, added a commit hook that blocks data files and identifiers, keep the repository private until the old commits are removed, and hand in only a clean export without history.
- What did not work: (to fill after P3).

## AI agents and LLM usage

<!-- team: confirm this section is accurate before submission -->

Coding assistants and agents wrote much of the code. They worked from a plan the team wrote and reviewed, and every change was reviewed. Agents worked on synthetic data only; real-data runs were done by a team member. No LLM is part of training or inference. Inference runs offline.
