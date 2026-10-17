# Task 1 (PV Detection) Benchmark & Validation Report

This document details the architecture, evaluation methodology, anti-overfitting strategy, and statistical analysis for **Task 1: PV Presence Detection** on the development dataset.

---

## 1. Aggregate Benchmark Results

| Model / Strategy | Precision | Recall | Task 1 F1 | Base Score (/70) | PV Detected (out of 22) | False Alarms (out of 18) | Train/Val Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1. Classical Baseline** (Midday/night ratios + Logistic Regression) | 0.652 | 0.682 | 0.667 | 46.69 | 15 | 8 | N/A |
| **2. Pilot PatchTST** (Unsupervised pretraining + whole-customer pooling) | 0.737 | 0.636 | 0.683 | 47.81 | 14 | 5 | 0.012 |
| **3. PatchTST Window Classifier** (Direct 7-day window supervision + $P_{90}$ aggregation) | **0.913** | **0.955** | **0.933** | **65.33** | **21** | **2** | **0.043** |

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
| **Fold 0** | 0.970 | **1.000** | 1.00 | 1.00 | +0.030 |
| **Fold 1** | 1.000 | **0.889** | 1.00 | 0.80 | -0.111 |
| **Fold 2** | 0.971 | **0.889** | 0.80 | 1.00 | -0.082 |
| **Fold 3** | 0.971 | **0.889** | 0.80 | 1.00 | -0.082 |
| **Fold 4** | 0.971 | **1.000** | 1.00 | 1.00 | +0.029 |
| **Average**| **0.976** | **0.933** | **0.913** | **0.955** | **0.043 (4.3%)** |

Across all 40 development customers, the model achieved **37 correct classifications, 2 false positives, and 1 false negative**.

---

## 6. Critical Technical Limitations & Validation Audit

The headline $F1 = 0.933$ represents out-of-fold performance under a specific hyperparameter configuration. An objective engineering audit reveals several key limitations:

### 1. Small Sample Uncertainty ($N=40$)
With only 40 development customers, each misclassification shifts the accuracy by $2.5\%$ and $F1$ by $\approx 2.2\%$. A score of $93.3\%$ on 40 samples corresponds to a **95% confidence interval of roughly $[81\%, 97\%]$**. Expected performance on the 20 hidden test customers should be estimated in the **$82\%$ to $90\%$ range**.

### 2. Selection Bias in Configuration
Hyperparameters (window length, stride, decision threshold rule, loss weighting, and epoch count) were evaluated on the same 40 out-of-fold records. Reporting the highest performing configuration introduces an optimistic selection bias. Multi-seed cross-validation across diverse splits is necessary to establish true parameter variance.

### 3. Structural Blind Spot for Late Installations
The customer decision rule relies on the 90th percentile of window probabilities ($P_{90}$) across solar months. If a customer installs a PV system late in the observation period (e.g., late autumn of the final year), the active solar period represents only a tiny fraction of total historical windows. In such cases, $P_{90}$ remains suppressed by the dominant pre-installation history, guaranteeing a false negative regardless of model confidence. A time-localized rolling sustained run rule is required to eliminate this blind spot.

### 4. Base Rate Disparity on Unlabelled Data
When evaluated on a random sample of 25 unlabelled series from `energy_consumption_unlabelled_dev.csv`, 14 series were classified as positive (56%). Typical residential/commercial solar penetration on a Swiss utility grid is significantly lower (~15–25%). This disparity suggests that the decision threshold calibrated on development data may be biased toward recall at the expense of false positives when deployed in the wild.

### 5. Label Lag vs. Physical Onset
Ground-truth commissioning dates often reflect utility administrative or metering registration rather than physical panel operation. While this label noise penalizes window-level training, it confirms that Task 2 (Commissioning Date) must be driven by physical change-point detection on the probability curve rather than fitting directly to administrative labels.

---

## 7. Direct Progression to Tasks 2 and 3

The probability trace $\mathbf{p} = [p_1, \dots, p_T]$ produced by PatchTST forms the direct bridge to the remaining two challenge tasks:

* **Task 2 (Commissioning Date):** Detect the persistent step-change where window probability transitions from baseline to the active regime, then refine to daily resolution using local midday demand differentials.
* **Task 3 (Nominal Capacity in kWp):** In the confirmed active post-commissioning period, combine the PatchTST embedding with absolute physical power deficit metrics (kW) to regress installed kWp against development ground truth.
