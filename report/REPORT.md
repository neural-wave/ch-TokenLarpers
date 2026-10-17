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
- a shoulder-hour deficit that maps to capacity in kWp.

The learned part is small. A logistic or CatBoost classifier, a decision threshold, a ridge or CatBoost capacity model, a scale factor κ and a date rule. Dates come from one of three regimes: a visible event, PV already present at the start of the series, or too little data.

All of these choices are made by inner cross-validation inside one fit procedure. Nested cross-validation on the 40 labelled customers scores the whole procedure, option pool included. The final model is the same procedure fit on all 40.

## Results

Filled after the real-data run (P3) from aggregate lines only: `nested:`, `dates:`, `selection:` and the shared-fold comparison against the other branches. No per-customer values.

`outer_informed_decisions=1`, so the estimate is labelled selection-biased.

Synthetic holdout benchmark (P2 gate, 40 customers, 2 repeats). These numbers are synthetic and are not evidence of competition accuracy:

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

- The raw changepoint fired on seasonal drops. Spring looks like a new PV system. Deseasonalizing and a year-over-year check fixed most of it.
- The labelled date lags the onset we can see. We learn an offset instead of trusting the visible break.
- Night loads such as heat pumps and storage heaters shift consumption in ways that mimic PV. Dedicated confounder features help; they do not solve it.
- With 40 labelled customers, any choice made by looking at results overfits. We moved all tuning inside the fit and log every outer-informed decision.
- Different branches scored on different splits, so numbers were not comparable. We fixed shared customer folds and a common out-of-fold format.
- Data hygiene: an identifier slipped into the repository once. We scrubbed the history and added a commit hook that blocks data files and identifiers.
- What did not work: (to fill after P3).

## AI agents and LLM usage

<!-- team: confirm this section is accurate before submission -->

Coding assistants and agents wrote much of the code. They worked from a plan the team wrote and reviewed, and every change was reviewed. Agents worked on synthetic data only; real-data runs were done by a team member. No LLM is part of training or inference. Inference runs offline.
