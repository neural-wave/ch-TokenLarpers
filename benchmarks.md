# Benchmark status: historical results, evaluation reset

Architecture tuning is paused. The current protocol is [frozen matched evaluation](docs/evaluation_protocol.md), configured in `configs/evaluation.json`. No new AIL development result or blind-test result has yet been produced with that protocol. Artificial smoke metrics must not be presented as predictive-quality evidence.

## Historical development results

These are earlier results on the same 40 development customers (22 PV, 18 non-PV), not independent model-selection or hidden-test evidence.

| Historical method | Presence F1 | True positives | False positives | False negatives |
|---|---:|---:|---:|---:|
| Whole-customer raw-feature baseline | 0.6667 | 15 | 8 | 7 |
| Custom frozen embedding pilot | 0.6829 | 14 | 5 | 8 |
| Official PatchTST frozen/fine-tuned embedding pilot | 0.6486 | 12 | 3 | 10 |
| Supervised PatchTST window classifier, positional persistence | 0.9167 | 22 | 4 | 0 |
| Exploratory hourly-profile logistic audit, matched old window formulation | 0.9565 | 22 | 2 | 0 |

The supervised classifier was retrained and reproduced all saved customer scores exactly. The exploratory logistic comparison was an audit run, not the new nested-CV benchmark. Its result establishes a necessary baseline, not definitive model-family superiority. Full older three-task results remain documented in [the original comparison](docs/official_patchtst_comparison.md).

## Corrections to the previous validation report

- The embedding pilot's F1=0.6829 was from the custom encoder, not the official adapter. The later supervised classifier was trained from scratch, not a demonstration of masked-pretraining benefit.
- The supervised classifier had 110,099 parameters; approximately 30,500 described the smaller embedding pilot.
- Folds were stratified customer folds. The old multi-seed script reused one persisted partition; its first two entries were hardcoded cached metrics. It did not establish stability across shuffled customer assignments.
- The old threshold search used fitted training-customer scores, not inner out-of-fold predictions. Training F1=1 is not an anti-overfitting guarantee.
- The 4,746 windows were not independent customers. Capping selected windows spread their centers mostly 12 days apart; a four-entry block typically spanned 36 days between centers, about 43 days of weekly context. The prior 12–14-day persistence claim was incorrect.
- The synthetic perturbation test evaluated a fold 0 checkpoint using backgrounds excluding fold 1. Twelve of 15 eligible backgrounds overlapped that checkpoint's training customers. Sixty paired generated variants were not 60 independent held-out households. This test did not establish hard-negative immunity.
- Night tariffs/heating/charging were hypotheses from post-hoc load profiles, not independently verified physical causes.
- Applying old persistence to pre-installation-only windows triggered 10 of 21 eligible PV customers. A good customer-ever-PV score did not validate commissioning dates.

The old stress scripts and positional/in-sample evaluator were removed. Their private result files were preserved for historical inspection, not relabelled as new evidence.

## Current acceptance evidence

The new evaluator compares the hourly baseline, fixed PatchTST scratch, and identical PatchTST with masked pretraining. It uses daily uncapped windows, elapsed-time/coverage persistence, inner customer-OOF threshold calibration, and genuinely different repeated outer partitions. It retains every prediction/configuration and reports temporal failures separately.

Verification: 47 regression tests passed. A real CLI smoke used 12 wholly artificial customers, two outer partitions, two inner folds, and three arms; it retained 72 outer customer predictions and 36 training lineages/histories. Reloading 12 outer model artifacts reproduced saved probabilities within 1e-6. No smoke score is claimed as AIL accuracy.

Fresh blind evaluation remains unperformed. The organizer's 20 hidden labels are unavailable to this evaluator. Repeated development CV and unlabelled predictions cannot substitute for blind validation. Initial capacity and the overall challenge score are not claimed by this presence/temporal evaluation reset.
