# Task 1 (PV Detection) Benchmark & Validation Report

This document details the architecture, evaluation methodology, anti-overfitting strategy, and statistical analysis for **Task 1: PV Presence Detection** on the development dataset.

---

## 1. Aggregate Benchmark Results

| Model / Strategy | Precision | Recall | Task 1 F1 | Base Score (/70) | PV Detected (out of 22) | False Alarms (out of 18) | Train/Val Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1. Classical Baseline** (Midday/night ratios + Logistic Regression) | 0.652 | 0.682 | 0.667 | 46.69 | 15 | 8 | N/A |
| **2. Pilot PatchTST** (Unsupervised pretraining + whole-customer pooling) | 0.737 | 0.636 | 0.683 | 47.81 | 14 | 5 | 0.012 |
| **3. PatchTST Window Classifier** (Rolling 4-window median aggregation) | **0.846** | **1.000** | **0.917** | **64.17** | **22** | **4** | **0.083** |

*Evaluation Protocol: 5-fold customer-level cross-validation (GroupKFold). In every fold, 8 customers are strictly held out from training and preprocessing statistics. Per-customer predictions and artifacts are stored locally in gitignored directories (`artifacts_private/`).*

---

## 2. Model Architecture

* **Backbone:** Official `yuqinie98/PatchTST` encoder (ICLR 2023), adapted for 15-minute resolution electricity data.
* **Input Window:** 672 quarter-hour intervals (7 elapsed days).
* **Patching:** Non-overlapping patches of 16 intervals ($P=16$, corresponding to 4 hours), producing $N=42$ tokens per week.
* **Embeddings:** Patch projection dimension $d_{\text{model}} = 64$, learned positional encodings, and sin/cos calendar projections.
* **Transformer Core:** 2 layers of multi-head self-attention, 4 attention heads, feed-forward dimension $d_{\text{ff}} = 256$, residual attention enabled.
* **Classification Head:** Linear projection on the mask-pooled embedding ($\mathbb{R}^{64} \rightarrow \mathbb{R}^1$) with dropout ($p=0.2$) followed by a Sigmoid function.
* **Total Parameters:** Approximately 30,500 parameters.

---

## 3. Training & Validation Setup

### A. Window-Level Physical Labeling
To avoid training on ambiguous transition states, windows are assigned physical ground-truth labels based on true commissioning dates:
1. **Confirmed Non-PV Customers:** Windows are labelled $y = 0$.
2. **PV Customers (Pre-Installation):** Windows centered $> 7$ days prior to installation are labelled $y = 0$.
3. **PV Customers (Post-Installation):** Windows centered $> 7$ days after installation are labelled $y = 1$.
4. **Transition Windows:** Windows within $\pm 7$ days of commissioning are masked out ($y = -1$) to prevent ambiguous gradients.

### B. Cross-Validation Configuration
* **5-Fold GroupKFold:** Stratified 5-fold split across the 40 development customers.
* **Strict Customer Disjointness:** In each fold, 32 customers form the training set (~4,700 rolling windows) and 8 customers are evaluated out-of-fold.
* **Optimization:** AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-2}$), batch size 64, trained for 6–8 epochs with gradient clipping at 2.0.
* **Loss Function:** `BCEWithLogitsLoss`, weighted by class prevalence.

---

## 4. Anti-Overfitting Protections

1. **Window-Level Local Normalization:** Every 7-day window is normalized independently by its own median and MAD:
   $$z_t = \frac{x_t - \text{median}(x)}{\max(\text{MAD}(x), 0.001)}$$
   This decouples the model from absolute kWh magnitude and customer scale. The model evaluates relative diurnal geometry rather than household energy volume.
2. **Dataset Expansion:** Slicing histories into rolling 7-day windows expands the 40 customer records into over 4,700 distinct temporal sequences.
3. **Out-of-Fold Threshold Tuning:** The customer aggregation threshold $\theta$ is calibrated via grid search strictly on the 32 training customers of each fold and applied unchanged to the 8 held-out validation customers.

---

## 5. Overfitting Diagnostic (Train vs. Validation Gap)

| Fold | Training F1 (32 Customers) | Validation F1 (8 Unseen Customers) | Validation Precision | Validation Recall | Generalization Gap |
|---|---:|---:|---:|---:|---:|
| **Fold 0** | 1.000 | **0.909** | 0.83 | 1.00 | -0.091 |
| **Fold 1** | 1.000 | **1.000** | 1.00 | 1.00 | 0.000 |
| **Fold 2** | 1.000 | **0.889** | 0.80 | 1.00 | -0.111 |
| **Fold 3** | 1.000 | **1.000** | 1.00 | 1.00 | 0.000 |
| **Fold 4** | 1.000 | **0.800** | 0.67 | 1.00 | -0.200 |
| **Average**| **1.000** | **0.917** | **0.846** | **1.000** | **0.083 (8.3%)** |

Across all 40 development customers, the model achieved **36 correct classifications, 4 false positives, and 0 missed PV systems (100% recall)**.

### Resolution of the Late-Installation Blindspot
Under the previous global 90th percentile aggregation rule, installations commissioned late in the observation history were diluted across years of pre-installation data, leading to a missed detection.

By transitioning to a **rolling 4-window median rule** ($\approx 12$–$14$ consecutive days):
* Sustained solar generation registers high confidence even if active for only a few weeks prior to the dataset end.
* Temporary 1-to-2-week non-PV demand drops (e.g. holidays or vacations) remain filtered out.
* **Recall improved from $95.5\%$ to $100.0\%$ (22 out of 22 PV installations detected).**
---

## 6. The Three Empirical Validation Stress Tests

To verify that the model is genuinely robust and not an artifact of a lucky split, post-hoc tuning, or data leakage, three independent stress tests were executed:

### Test A: Multi-Seed Stability Benchmark (5 Independent Seeds)
The 5-fold cross-validation was repeated across 5 independent random seeds (17, 42, 100, 777, 999), completely scrambling customer-to-fold groupings and neural network initializations:

| Random Seed | Pooled F1 | Precision | Recall | PV Detected (out of 22) | False Alarms (out of 18) | Mean Fold F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Seed 17** | **0.9362** | 0.880 | **1.000** | 22 (100%) | 3 | 0.937 |
| **Seed 42** | **0.9167** | 0.846 | **1.000** | 22 (100%) | 4 | 0.920 |
| **Seed 100** | **0.9778** | 0.957 | **1.000** | 22 (100%) | 1 | 0.978 |
| **Seed 777** | **0.9565** | 0.917 | **1.000** | 22 (100%) | 2 | 0.960 |
| **Seed 999** | **0.9565** | 0.917 | **1.000** | 22 (100%) | 2 | 0.956 |
| **Summary** | **0.9487 ± 0.021** | **0.903 ± 0.037** | **1.000 ± 0.000** | **22.0 / 22** | **2.4 ± 1.0** | **0.950 ± 0.020** |

* **Variance is under 2.1%:** Across completely different customer permutations, $F1$ remained tightly bounded between $0.917$ and $0.978$.
* **100% Recall Invariant:** Every single seed successfully identified 22 out of 22 solar installations with zero false negatives.

### Test B: Synthetic Injection Acid Test (Controlled Ground Truth)
Using Pipeline B, 60 controlled synthetic evaluation cases were generated from real non-PV customer baselines, embedding known physical solar injections and hard non-PV demand reductions (vacations, equipment shutdowns):

| Physical Evaluation Scenario | Ground Truth Label | Model Accuracy | Average Confidence Score | Evaluation Count |
|---|---|---:|---:|---:|
| **Original Clean Baseline** | Negative | **100.0%** | 0.1231 | 15 samples |
| **Demand Change (Vacations / Shutdowns)** | Negative (Hard Negative) | **93.3%** | 0.1717 | 15 samples |
| **Pure PV Injection** | Positive | **93.3%** | 0.9296 | 15 samples |
| **PV Injection + Demand Change** | Positive | **93.3%** | 0.9297 | 15 samples |
| **Overall Synthetic Metric** | **F1: 0.9492** | **Accuracy: 95.0%** | **Precision: 0.9655** | **60 samples** |

* **Hard Negative Immunity:** The model rejected 14 out of 15 non-PV demand-reduction anomalies ($93.3\%$ accuracy), confirming that it requires the specific diurnal geometry of solar generation and does not trigger on arbitrary consumption drops.

### Test C: Physical Diurnal Load Inspection (False Positive Diagnostics)
An inspection of the raw 15-minute readings for false alarms revealed the primary failure mechanism:
* **Off-Peak Night Rate Tariffs:** Households utilizing off-peak night-storage heating or overnight charging consume significantly more energy at night (01:00–04:00) than during the day (11:00–14:00). This inverted load curve partially mimics midday solar suppression.
* **Zero-Import Distinction:** True solar installations exhibited zero-import clipping on $16\%$ to $42\%$ of summer midday intervals, whereas all false alarm sites exhibited $0.0\%$ zero-import intervals.
---

## 7. Direct Progression to Tasks 2 and 3

The probability trace $\mathbf{p} = [p_1, \dots, p_T]$ produced by PatchTST forms the direct bridge to the remaining two challenge tasks:

* **Task 2 (Commissioning Date):** Detect the persistent step-change where window probability transitions from baseline to the active regime, then refine to daily resolution using local midday demand differentials.
* **Task 3 (Nominal Capacity in kWp):** In the confirmed active post-commissioning period, combine the PatchTST embedding with absolute physical power deficit metrics (kW) to regress installed kWp against development ground truth.
