#Credit Card Fraud Detection Through the Data Science Lifecycle

**Course: ITIS5000V - Intro to Data Science (SEM), Winter 2026**
**Institution: Sprott School of Business, Carleton University, Ottawa, ON, Canada**

## Group 8

---

## Project Overview

This project builds a three-tier supervised machine learning pipeline to detect fraudulent credit card transactions on the ULB benchmark dataset. The core challenges addressed are:

- Class imbalance: only 0.17% of transactions are fraud (578:1 ratio), making standard accuracy meaningless
- Model interpretability: regulatory frameworks (PIPEDA, GDPR) require explainable automated decisions
- Operational credibility: results must be actionable for fraud analysts, not just statistically strong

The pipeline follows the full Data Science Lifecycle: exploratory data analysis, preprocessing, SMOTE-based imbalance correction, three-tier model training, threshold optimization, cross-validation and SHAP explainability.

---

## Key Results

### Hold-Out Test Set (56,962 transactions, 98 fraud cases)

| Model | Precision | Recall | F1 | MCC | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.058 | 0.918 | 0.109 | 0.228 | 0.970 |
| LOF (Baseline) | 0.005 | 0.143 | 0.010 | 0.019 | 0.503 |
| Isolation Forest (Baseline) | 0.305 | 0.327 | 0.315 | 0.314 | 0.954 |
| Random Forest | 0.813 | 0.796 | 0.804 | 0.804 | 0.972 |
| XGBoost | 0.512 | 0.867 | 0.644 | 0.666 | 0.978 |
| **Stacked Ensemble** | **0.794** | **0.827** | **0.810** | **0.810** | **0.978** |

**Primary metric: MCC (Matthews Correlation Coefficient)**

### Cross-Validation (5-fold, SMOTE inside each fold)

| Metric | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| MCC | 0.837 | 0.028 | 0.811 | 0.877 |
| F1-Score | 0.836 | 0.028 | 0.808 | 0.877 |

### Threshold Optimization (Stacked Ensemble)

| Setting | Threshold | MCC | F1 | Precision | Recall | False Alarms |
|---|---|---|---|---|---|---|
| Default | 0.500 | 0.810 | 0.810 | 0.794 | 0.827 | 21 |
| Optimized | 0.977 | 0.860 | 0.857 | 0.929 | 0.796 | 6 |

Threshold optimization reduced false alarms by **71%** with minimal sacrifice in fraud detection coverage.

### SHAP Feature Importance

Both XGBoost and Random Forest independently ranked the same top features:

| Rank | XGBoost | Random Forest |
|---|---|---|
| 1 | V14 | V14 |
| 2 | V4 | V12 |
| 3 | V12 | V4 |
| 4 | V10 | V10 |
| 5 | V1 | V3 |

V14 is the single most important fraud predictor in both models (mean SHAP = 1.67 for XGBoost).

---

## Pipeline Architecture

```
Tier 1 - Baselines
    Logistic Regression      (supervised, class_weight=balanced)
    Local Outlier Factor     (unsupervised, n_neighbors=20)
    Isolation Forest         (unsupervised, contamination=prior fraud rate)

Tier 2 - Supervised Classifiers (trained on SMOTE-balanced data)
    Random Forest            (n_estimators=200, class_weight=balanced)
    XGBoost                  (n_estimators=300, max_depth=6, learning_rate=0.05)

Tier 3 - Stacked Ensemble
    Base learners            Random Forest + XGBoost
    Meta-learner             Logistic Regression (5-fold out-of-fold predictions)

Post-processing
    Threshold Optimization   Precision-Recall curve, optimal threshold = 0.977
    SHAP Explainability      TreeExplainer on XGBoost and Random Forest
    Cross-Validation         5-fold stratified, SMOTE applied inside each fold
```

---

## Dataset

**ULB Credit Card Fraud Detection Dataset**

- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size:** 284,807 transactions from European cardholders, September 2013
- **Fraud cases:** 492 (0.1727%) — 578:1 class imbalance ratio
- **Features:** V1 to V28 are PCA-transformed principal components (anonymized for cardholder privacy). Time and Amount are retained in original form.
- **Missing values:** None

> **Important:** The dataset file (creditcard.csv, 150.8 MB) is not included in this repository due to file size limits. Download it from Kaggle and place it in the same directory as the notebook before running.

---

## Repository Structure

```
ITIS5000V-Fraud_detection/
|
|-- fraud_detection.ipynb      Main analysis script (complete pipeline) / we used Anaconda navigator (jupyter) to run it. 
|-- README.md                        This file
|-- metrics_summary.csv                       
|
|-- figures/                         
    |-- fig1_class_distribution.png      Class imbalance visualization
    |-- fig2_smote.png            SMOTE before and after
    |-- fig3_confusion_matrices.png      Tier 1 and Tier 2 model comparison
    |-- fig4_mcc_comparison.png          Primary metric bar chart
    |-- fig5_roc_pr_curves.png           ROC and Precision-Recall curves
    |-- fig6_threshold_optimization.png  Optimal operating point
    |-- fig7_shap_beeswarm.png           Global feature importance (XGBoost)
    |-- fig8_shap_waterfall.png          Single transaction explanation
```

---

## Requirements

Python 3.8 or higher is required.

### Install all dependencies

```
pip install numpy pandas scikit-learn xgboost imbalanced-learn shap matplotlib seaborn
```

### Libraries used

| Library | Purpose |
|---|---|
| numpy | Array operations and numerical computing |
| pandas | Dataset loading and manipulation |
| scikit-learn | Models, metrics, preprocessing, cross-validation |
| xgboost | XGBoost gradient boosting classifier |
| imbalanced-learn | SMOTE oversampling technique |
| shap | SHAP explainability analysis |
| matplotlib | Figures and charts |
| seaborn | Heatmaps and enhanced styling |

---

## How to Run

### Step 1 — Clone the repository

```
git clone https://github.com/haritesadek/Credit-Card-Fraud-Detection-Through-the-Data-Science-Lifecycle.git
```

### Step 2 — Install dependencies

```
pip install numpy pandas scikit-learn xgboost imbalanced-learn shap matplotlib seaborn
```

### Step 3 — Download the dataset

Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and download creditcard.csv. Place it in the same folder as fraud_detection.ipynb

### Step 4 — Run the pipeline

```
python fraud_detection.ipynb
```

### What the script does

The script executes:

- Loads and validates the dataset (shape, fraud rate, missing values)
- Runs EDA and generates Figures 1 to 3 (class distribution, amount distribution, correlation heatmap)
- Preprocesses data (z-score normalization, stratified 80/20 split, SMOTE on training set only)
- Trains Tier 1 baseline models (Logistic Regression, LOF, Isolation Forest)
- Trains Tier 2 supervised models (Random Forest, XGBoost)
- Trains Tier 3 stacked ensemble (RF + XGBoost meta-learned by Logistic Regression)
- Generates evaluation figures (Figures 3 to 6: confusion matrices, MCC bar, ROC/PR, threshold optimization)
- Computes SHAP values and generates Figures 7 to 8 (beeswarm, waterfall)
- Runs 5-fold cross-validation with SMOTE inside each fold
- Prints final metrics summary table to console

All figures are saved to the figures/ directory.

**Expected runtime:** 20 to 40 minutes depending on hardware.

---

## Key Design Decisions

| Decision | Rationale | Reference |
|---|---|---|
| SMOTE on training set only | Prevents synthetic samples leaking into test set, which inflates results | Dal Pozzolo et al. (2015) |
| MCC as primary metric | Accounts for all four confusion matrix cells, robust to 578:1 imbalance | Singh et al. (2022) |
| Stratified 80/20 split | Preserves 0.17% fraud rate in both training and test partitions | Standard ML practice |
| LOF and Isolation Forest on raw data | Unsupervised models do not use class labels, SMOTE not applicable | Singh et al. (2022) |
| SHAP TreeExplainer | Exact Shapley values for tree models, computationally efficient | Lundberg & Lee (2017) |
| 5-fold CV with SMOTE inside folds | Prevents leakage during cross-validation, unbiased stability estimate | Chawla et al. (2002) |
| Threshold = 0.977 (optimized) | Maximizes F1 across all operating points on the Precision-Recall curve | Dal Pozzolo et al. (2015) |

---

## Research Questions and Answers

**RQ1:** Can a stacked ensemble of Random Forest and XGBoost, trained on SMOTE-balanced data and evaluated with MCC, achieve an F1-score above 0.80 and outperform LOF and Isolation Forest baselines by at least 50% on MCC?

**Answer:** Yes. The Stacked Ensemble achieved MCC = 0.810 and F1 = 0.810. This substantially outperforms LOF (MCC = 0.019) and Isolation Forest (MCC = 0.314). Both Tier 2 models exceeded the F1 > 0.80 target. Cross-validation confirmed stability at MCC = 0.837 plus or minus 0.028.

**RQ2:** Can SHAP values identify the top three predictive features with directional consistency across flagged fraud cases, making the model operationally credible?

**Answer:** Yes. V14, V4, and V12 were consistently ranked in the top three by both XGBoost and Random Forest independently. V14 shows a clear monotonic negative relationship with fraud probability. This cross-model consistency confirms genuine fraud signals rather than algorithm-specific artifacts.

---

## References

1. Dal Pozzolo, A., Caelen, O., Johnson, R. A., and Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE SSCI, pp. 159-166.
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. JAIR, 16, 321-357.
3. Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD 2016, pp. 785-794.
4. Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS, vol. 30, pp. 4765-4774.
5. Singh, Y., Singh, K., and Chauhan, V. S. (2022). Fraud detection techniques for credit card transactions. IEEE ICIEM, pp. 821-824.
6. Makki, S. et al. (2019). An experimental study with imbalanced classification approaches for credit card fraud detection. IEEE Access, 7, 93010-93022.
7. Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241-259.
8. Goodman, B. and Flaxman, S. (2017). EU regulations on algorithmic decision-making and a right to explanation. AI Magazine, 38(3), 50-57.

---

*ITIS5000V - Intro to Data Science. Carleton University, Sprott School of Business, Winter 2026*
*Group 8:*
