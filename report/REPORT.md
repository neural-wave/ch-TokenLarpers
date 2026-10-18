# Voltera — three-page hackathon report

Canonical manuscript: [`report.qmd`](report.qmd). Finished rendering: [`voltera-report.pdf`](voltera-report.pdf). The earlier LaTeX draft has been superseded; there is one maintained manuscript.

## Page structure

1. **Time-series problem and data:** task, meter interpretation, daily/weekly/seasonal structure, missingness and daylight saving. One conceptual before/after PV-withdrawal schematic, explicitly not customer measurements.
2. **Time-series methodology:** preprocessing, features, temporal baselines, changepoints, classification, commissioning date and nominal capacity. A LaTeX/TikZ vector flowchart shows the separate evidence paths. Extensions remain a separate research experiment.
3. **Results and discussion:** customer-disjoint evaluation, synthetic extension findings in prose, date/capacity errors and failure cases. One diagram follows our Transformer research and decision to change direction. There is no benchmark table.

## Build

Rendering the manuscript requires Quarto >=1.7 with bundled Typst. The rendered revision used Quarto 1.10.19 / Typst 0.15.1. The compiled diagram is included, so a normal report render does not need LaTeX. Editing the diagram requires `latexmk`, `pdflatex`, TikZ and the `standalone` document class.

From the repository root:

```bash
quarto render report/report.qmd --to arxiv-typst
pdfinfo report/voltera-report.pdf
```

Each page has one LaTeX/TikZ vector visual. Sources are `figures/pv-signature.tex`, `figures/methodology.tex` and `figures/research-path.tex`; each is embedded as its corresponding PDF. None contains customer data. To rebuild them from the repository root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/voltera-tikz-build report/figures/*.tex
cp /tmp/voltera-tikz-build/{pv-signature,methodology,research-path}.pdf report/figures/
quarto render report/report.qmd --to arxiv-typst
```

The previous Python plots and their generator have been removed. Recheck the **three-page limit** after substantive edits.

## Template and methodological provenance

- Vendored [quarto-arxiv-typst](https://github.com/stephenturner/quarto-arxiv-typst) under `_extensions/arxiv`, upstream revision `6a038e279822eb911fb91c2dc0af064a99623415`. Upstream MIT license and bundled font licenses are retained. A4/margins are configured in the manuscript.
- [aeon-benchmark](https://github.com/aeon-toolkit/aeon-benchmark), reviewed at `8d21e6e4327857bd2758a1627ce2b00243f2ac29`, informs reproducibility conventions: fixed splits, seeds, matched comparisons and task-specific metrics. No aeon benchmark code or result is claimed as Voltera's.
- Latest reviewed `origin/main`: `d886652` (same model implementation as `883a3ad`; newer commit updates the decision log).
- Preserved local Transformer hybrid: `8b81795`. The team confirmed PatchTST is **not in the chosen Voltera pipeline**. Retained code/checkpoints do not imply selected inference use.
- Isolated extension experiment: `14e0049`, branch `experiment/pv-extensions`, not merged into main.
- Results transcribe the recorded fabricated benchmark's **end-to-end** arm: nine customer scenarios and seven true events, using predicted initial dates/capacities. This is not the conditional run supplied with known initial targets.
- The 44-test statement refers to the recorded focused regression run, not a full-suite rerun for this manuscript.

## Evidence and publication boundary

This is a team-review draft. Dataset sizes come from the public specification, not private inspection. No claim is made that every supplied unlabelled customer entered training. No aggregate AIL metrics, real-customer traces, identifiers or predictions are included. Repository/organizer rules prohibit real-customer figures even when anonymized. Real main-task precision/recall/F1 and date/capacity performance remain data-holder material and must not be inferred from the optional synthetic benchmark.

Development/training results are not independent validation. The decision log records an audit involving lock-box customers, so the report does not describe subsequent evidence as a pristine lock-box result. The team must review the final narrative and AI-use disclosure before publication. No new encoder training, pipeline change or model adoption is part of this report revision.
