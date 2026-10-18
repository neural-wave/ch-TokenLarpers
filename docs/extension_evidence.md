# PV extension detector: evidence and audit (synthetic only)

All numbers below come from fabricated data (`pvdetect.synthetic.generate_extension_dataset`). They show that the code behaves as designed and how the method trades recall against false alarms. They are **not** an estimate of real accuracy. The real-data gate is pre-registered in `DECISIONS.md` (2026-10-18).

## What ships
`pvdetect/extensions.py` runs after the main model and returns the extension rows. Configuration:
- Nowcaster trained on labelled customers only; no `--ext-unlabelled`.
- Level test plus year-over-year box test.
- Global cap of 3 events, at most 2 per customer.

`--no-extensions` falls back to the header-only file. Method: README → "PV extensions (optional bonus)".

## Safety evidence: unit and contract tests (`tests/test_extensions.py`, 24 tests)
| Property | Test |
|---|---|
| `predictions.csv` identical with extensions on and off | `test_extensions_never_change_the_main_predictions_and_respect_the_output_contract` |
| ≤ 3 rows, ≤ 2 per customer; only TRUE customers; date after predicted install; total above predicted initial kWp | same |
| Any detector exception → header-only file, main predictions unchanged | `test_a_failing_detector_degrades_to_header_only_without_touching_predictions` |
| Save/load round trip: `catboost_ext.cbm` referenced, exported and reproduced exactly | `test_saved_model_reproduces_the_extension_rows` |
| Leakage: extension labels of non-training customers cannot change the fitted nowcaster | `test_held_out_extension_labels_cannot_change_the_fit` |
| A 5 → 10 kWp extension is dated within ±15 d and sized within ±25 % | `test_a_doubling_is_dated_within_15_days_and_sized_within_25_percent` |
| No event for no-extension customers (3 seeds × {none, vacation, AC, home office}) | `test_no_extension_customers_give_no_event` (12 cases) |
| A predicted install date 4.5 months too early does not turn the installation into an extension | `test_an_early_predicted_install_date_…` |
| Missing or zero capacity labels are skipped, not fatal | `test_unusable_capacity_labels_are_skipped_not_fatal` |
| Change tests on hand-built series: one step found at its window and size; flat noise, outage dip and seasonal bias rejected; two steps both found; box located at its start | 5 unit tests |

Full suite: 426 passed. The 4 failures (`test_augmentation` ×2, `test_preprocessing` ×1, `test_preprocessing_scale_io` ×1) fail identically without this change. They come from the borrowed pandas 3 test environment, not the project pins.

## Independent review
A separate reviewer agent audited the diff for safety, leakage, artifacts, indexing and contract. It found 8 issues, all fixed and re-reviewed:
- **Blocker:** two existing artifact tests broke. They now fit with `extensions=False`.
- **Crash paths in training:**
  - A NaN or zero capacity label crashed the CatBoost fit. Such labels are now skipped.
  - An embedding block combined with a pool crashed training. The pool prediction is now caught.
  - Separately, `procedure.fit` wraps the extension fit, so the main model is always returned.
- **False-alarm mode:** an installation dated too early by the main model was reported as an extension. A guard now catches it (regression test above).
- **Nits fixed:**
  - `cum` is now updated only after the guards pass.
  - Two events within 30 days are merged into one.
  - `stability.py` now predicts with extensions disabled.
  - The `ext:` CV line notes that the top-3 cap applies per outer fold.

Two remaining nits are accepted, not fixed. Fixing them would have changed the benchmarked code after evaluation.
- A NaT extension-label date drops that customer's earlier windows. This loses training data but never mislabels.
- The per-customer cap of 2 is applied before the installation guard, so a rejected installation step can use up a slot.

## Benchmark (`python -m pvdetect.ext_benchmark`)
Each repeat runs end to end:
- **Main model:** `procedure.fit` trains on a dev-sized labelled set (24 customers, 23 with PV).
- **Test population:** 90 customers covering all 15 cases, 6 of each.
- **Inputs to the extension step:** presence, dates and initial kWp come from the fitted main model, not from labels.

The suite is deliberately hard: 60 events per repeat, including small (+15–30 %), late, winter and clipped extensions, plus look-alikes such as outages, battery retrofits and load changes.

The **hidden-test Monte Carlo** mimics the PDF's test geometry. Each of 3000 draws holds 1 one-event customer, 1 two-event customer and 8 non-extension PV customers, with the shipped top-3 rule applied per draw.

### Development seeds (configuration chosen here)
Train 500+r, pool 600+r, test 700+r, 5 repeats per variant. Arm names follow the original run: there, "shipped" included the pool.

| Variant | Arm | TP | FP | FN | Bonus (population) | 90 % CI | MC mean bonus |
|---|---|---|---|---|---|---|---|
| A | with pool | 4 | 7 | 296 | 0.22 | [0.06, 0.41] | 0.06 |
| A | **no pool** | 35 | 42 | 265 | **1.64** | [1.23, 2.04] | **1.09** |
| B | with pool | 5 | 18 | 295 | 0.28 | [0.06, 0.49] | 0.08 |
| B | **no pool** | 23 | 41 | 277 | **1.12** | [0.78, 1.49] | **0.59** |

**Decision (synthetic, Tier R): drop the pool.** The research prototype's pseudo-labels were idealized: true end-of-series kWp with random noise. The fitted main model's own labels made the nowcaster nearly flat over time and cut recall 2–9× (35 → 4, 23 → 5, 26 → 11 and 29 → 4 TPs). `--ext-unlabelled` remains available but documented as not recommended.

### Confirmation seeds (fresh, after the decision)
Train 800+r, pool 900+r, test 1000+r. These are outside the holdout block, which starts at 10000.

| Variant | Arm | TP | FP | FN | F1_ext | D_ext | P_ext | Bonus (population) | 90 % CI | MC mean bonus | P(bonus > 0) | MC TP | MC FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | **shipped** | 26 | 53 | 274 | 0.137 | 0.771 | 0.542 | **1.23** | [0.87, 1.57] | **0.82** | 0.21 | 0.22 | 0.88 |
| A | with pool | 11 | 21 | 289 | 0.066 | 0.861 | 0.425 | 0.59 | [0.33, 0.87] | 0.19 | 0.05 | 0.05 | 0.69 |
| A | no box | 33 | 73 | 267 | 0.163 | 0.759 | 0.539 | 1.45 | [1.11, 1.81] | 1.34 | 0.34 | 0.39 | 1.19 |
| A | no top-k | 26 | 53 | 274 | 0.137 | 0.771 | 0.542 | 1.23 | [0.89, 1.58] | 0.81 | 0.21 | 0.22 | 0.88 |
| B | **shipped** | 29 | 38 | 271 | 0.158 | 0.557 | 0.512 | **1.36** | [1.01, 1.74] | **0.84** | 0.22 | 0.24 | 0.76 |
| B | with pool | 4 | 17 | 296 | 0.025 | 0.808 | 0.535 | 0.22 | [0.06, 0.42] | 0.06 | 0.02 | 0.02 | 0.69 |
| B | no box | 32 | 89 | 268 | 0.152 | 0.542 | 0.492 | 1.30 | [0.97, 1.65] | 0.87 | 0.27 | 0.28 | 1.72 |
| B | no top-k | 29 | 38 | 271 | 0.158 | 0.557 | 0.512 | 1.36 | [1.01, 1.72] | 0.84 | 0.22 | 0.24 | 0.78 |

Per case, shipped arm (TP/FP/FN, summed over 5 repeats; 30 customers per case):

| Variant | single | double | early | late | winter | small | clipped | ew_added | with_battery | plain | outage | battery_retrofit | load_change | degradation | no_pv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 3/11/27 | 4/3/56 | 4/7/26 | 0/4/30 | 4/3/26 | 1/0/29 | 1/2/29 | 7/7/23 | 2/4/28 | 0/1/0 | 0/2/0 | 0/5/0 | 0/4/0 | 0/0/0 | 0/0/0 |
| B | 5/3/25 | 4/6/56 | 8/8/22 | 2/2/28 | 1/1/29 | 1/0/29 | 1/2/29 | 4/1/26 | 3/6/27 | 0/1/0 | 0/2/0 | 0/3/0 | 0/2/0 | 0/1/0 | 0/0/0 |

## Findings
1. **Expected value is positive but small.** On the hidden-test geometry the shipped detector averages 0.82–0.84 bonus points out of 10. About 1 draw in 5 scores anything. Because the bonus never goes below 0, shipping cannot lower the final score.
2. **False alarms are rare on non-extension PV customers.** Across 150 such customers per variant (plain, outage, battery retrofit, load change, degradation), A has 12 and B has 9, about 0.06–0.08 per customer. Most remaining FPs sit on extension customers: an event dated more than 30 days off counts as both FP and FN.
3. **Date accuracy, not detection, is the largest loss.** On `single` (A), 3 events are matched and 11 are mis-dated. A better date refinement is the clearest next lever.
4. **The year-over-year box is not a consistent win.** It roughly halves FPs (B: 89 → 38; A: 73 → 53) but loses some TPs. Net bonus: A is better without it (1.45 vs 1.23), B is about even (1.30 vs 1.36). It stays because constants were fixed before the confirmation run, and fewer false alarms is the safer choice for the real gate.
5. **The global top-3 cap rarely binds**, because candidates are few. It is a safety bound, not a score driver.
6. **Weak spots:** small (+15–30 %) and late extensions are essentially undetected, and the second event of a two-extension customer is usually missed.

## Real-data gate (data holder)
Pre-registered in `DECISIONS.md`. Ship extensions only if the nested-CV `ext:` line shows `fp ≤ 1.00` (mean over outer repeats; 40 dev customers out-of-fold). Otherwise make the final fit with `--no-extensions`. The dev extension's `tp` is reported, not gated. No extension constant changes after the run.

## Second audit: four independent agents (fresh seeds 1100+ / 1300+, synthetic)
Each agent ran its own scripts read-only (under `/tmp/audit_*`) and confirmed it reproduced the shipped arm exactly.

**1. The harness measures correctly.**
- Truth fed back as predictions scores 10.0.
- Date shifts of ±20 d, +30 d and +31 d and a ×1.25 kWp error each give the expected D, P and TP.
- Independent brute-force re-scoring of all arms matches exactly (A 26/53/274, B 29/38/271).
- No customers are dropped or duplicated. Dates have no systematic offset: median signed error −1 d (A) and −4 d (B).
- Low recall is real, not a measurement bug.

**2. Where events are lost** (variant A, 120 events, main-model inputs):

| Stage | Share |
|---|---|
| Customer not called PV | 0 % |
| Main model's install date on/after the extension, or within 60 d of it (predicted 440–1389 d late) | 22 % |
| Nowcast shows a step below ×1.15 | 28 % |
| Level test rejects (MIN_RATIO against the max earlier plateau) | 17 % |
| No PELT boundary | 7 % |
| Dated > 30 d off | 6 % |
| Found (TP) | 10 % |

With a perfect nowcast, the ceiling under the main model's inputs and guards would be about 61 % (73 of 120).

**3. Root cause: the nowcaster.**
- Inside one constant-capacity regime it swings with the season by 0.26–0.43 in log space (p90–p10). That is three times the ×1.15 threshold.
- It registers only about half of a true step: median observed/true ratio 0.47–0.53, event-level slope 0.23. Clipped extensions are almost invisible (ratio 0.13).
- The swing is customer-specific. A cross-fitted per-week bias curve spans only 0.05–0.14, so subtracting it leaves the swing unchanged.

**4. Fixes tested on fresh seeds; none adopted.** Population TP/FP; top-3 = TP/FP after the shipped global top-3 cap.

| Change | Result vs. shipped | Verdict |
|---|---|---|
| Oracle presence/date/kWp | +4 TP of 120 | Main-model inputs are not the main loss |
| Prototype HistGradientBoosting nowcaster | ≈ equal at n_train 24; +5 TP only at 90 | Not a fix at dev size |
| Training customers 24 → 90 | +4–5 TP | Small data-hunger effect |
| Reference = latest plateau | fresh 8/16 → 14/47; top-3 4/2 → 1/5 | Reject |
| Seasonal de-bias (raw or customer-centred) | fresh main 8/16 → 8/16 or 8/13; top-3 equal or worse | Reject |
| De-bias + latest plateau | top-3 4/2 → 1/5 | Reject |
| Install-step guard removed | +1–2 TP, +1–5 FP; inconsistent across repeats | Reject |
| MIN_RATIO 1.08 | +3 TP, +2 FP (within repeat noise of ±4 TP) | Not adopted |
| Six date-refinement alternatives | ±2 TP | Keep `refine_day` |

**Conclusion.** The detector is implemented and measured correctly. Its ceiling is set by a nowcast whose customer-specific seasonal swing is larger than typical extension steps. A real fix needs a redesign: a per-customer seasonal model, or change detection on the year-over-year difference of the nowcast. That is unmeasured and out of scope before the deadline. The shipped constants stay as pre-registered.

## Third round: per-customer seasonal change-point (rejected by pre-registered rule)
**Candidate.** Replace the level test and year-over-year box with a per-customer harmonic change-point model: the customer's own two annual harmonics, fitted jointly with up to two steps (BFAST-style). It targets the root cause above.

**Unit check.** On a hand-built series (customer-specific swing 0.35, true step 0.25):
- The shipped test finds nothing.
- The harmonic model finds the step at the exact window (δ 0.246, z 14.8).
- It stays silent on the flat and outage series.

**Adoption rule** (pre-registered in `DECISIONS.md`): adopt only if, in both variants, the hidden-test MC mean bonus beats shipped **and** MC FP is at most shipped + 0.10. FP matters because the real-data gate is `fp ≤ 1.00`.

| Run | Shipped MC bonus / FP | Harmonic MC bonus / FP | Verdict |
|---|---|---|---|
| A dev r0–1 (Z chosen = 3) | 0.64 / 1.00 | 1.01 / 1.70 | (dev) |
| A fresh r2–5, Z = 3 | 1.18 / 0.83 | 1.87 / 0.97 | fail: FP +0.14 |
| B dev r0–1 (Z chosen = 3) | 1.06 / 0.98 | 1.92 / 0.95 | (dev) |
| B fresh r2–5, Z = 3 | 1.24 / 0.86 | 2.21 / 0.77 | pass |
| B new block r6–9, Z = 4 (Tier B, pre-registered) | 1.19 / 0.68 | 1.56 / 0.95 | fail: FP +0.27 |

**Reading.** The harmonic model raised the MC bonus in every run (+0.37 to +0.97), but false alarms grew in three of four held-out comparisons, mostly on extension customers (double, clipped, winter). By the pre-registered rule the shipped detector stays.

**Next steps after the deadline.** The bonus gain is consistent. Revisit with an FP control learned on training customers, for example a per-customer z calibration, and confirm on a new seed block.
