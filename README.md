# Capstone Two: Fraud Detection in Financial Transactions

 <!-- Replace with an actual banner image, e.g., a visualization from the project or a fraud-themed graphic -->

[](https://www.python.org/)
[](https://opensource.org/licenses/MIT)
[](https://github.com/mesfin-k/capstone-two)
[](https://github.com/mesfin-k/capstone-two/commits/main)
[](https://www.kaggle.com/datasets/ealaxi/paysim1)

## Table of Contents
- [ Overview](#-overview)
- [ Problem Statement](#-problem-statement)
- [ Dataset](#-dataset)
- [ Methodology](#-methodology)
- [ Results](#-results)
- [ Key Visualizations](#-key-visualizations)
- [ Challenges and Learnings](#-challenges-and-learnings)
- [ Future Work](#-future-work)
- [ Repository Structure](#-repository-structure)
- [ Deliverables](#-deliverables)
- [ References](#-references)
- [ Acknowledgements](#-acknowledgements)

##  Overview
This project is a core component of the **Springboard Data Science Career Track Capstone Two**. It focuses on developing a sophisticated **fraud detection model** using the PaySim synthetic dataset, which simulates mobile money transactions in a financial ecosystem. The dataset has been meticulously filtered to include only `TRANSFER` and `CASH_OUT` transaction types—these are the primary vectors for fraudulent activities in the original data.

The primary objective is to enable **real-time fraud detection** to mitigate financial losses, enhance customer trust, and strengthen fraud prevention mechanisms. By leveraging machine learning, we aim to uncover subtle patterns in transaction data that signal malicious intent, all while handling the inherent challenges of imbalanced datasets and evolving fraud tactics.

This repository contains a complete, reproducible workflow—from raw data ingestion to model deployment readiness. Built on August 14, 2025, as part of the career track requirements.

##  Problem Statement
In the digital finance era, fraudulent transactions pose a severe threat, leading to billions in annual losses for banks, fintech companies, and consumers. Fraud is **rare (often <0.1% of transactions)**, **high-stakes**, and **adaptive**, making traditional rule-based systems insufficient. Key challenges include:
- **Imbalance**: Legitimate transactions vastly outnumber frauds, risking models that ignore minorities.
- **Evolving Patterns**: Fraudsters adapt, requiring models that generalize well.
- **Trade-offs**: High recall (catching frauds) vs. high precision (avoiding false positives that frustrate users).

The core question: Can we build a predictive model using transaction metadata to accurately classify `isFraud` while prioritizing recall to minimize undetected fraud? Success is measured by metrics like F1-score and ROC-AUC, ensuring a balance between detection efficacy and operational feasibility.

##  Dataset
- **Source**: [PaySim1 Synthetic Financial Dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) – A synthetic dataset generated from aggregated real-world private mobile money transaction data, simulating normal operations with injected malicious behaviors.
- **Original Size**: Approximately 6,362,620 rows and 11 columns, representing ~1 month of simulated transactions (each `step` = 1 hour).
- **Filtered Size**: ~460,394 rows (focused on `TRANSFER` and `CASH_OUT` types, as fraud occurs exclusively in these; further sampled to 1,000 for computational efficiency in this capstone).
- **Fraud Statistics**: Overall fraud rate ~0.129% (8,213 fraud cases in full dataset); in filtered data, ~1.78% fraud, emphasizing the imbalance.
- **Key Features** (11 total):
  - `step`: Integer timestep (1-744, simulating hours).
  - `type`: Categorical (`TRANSFER` or `CASH_OUT`).
  - `amount`: Float transaction value (highly skewed, frauds often larger).
  - `nameOrig`: Sender account ID (string, anonymized).
  - `oldbalanceOrg`, `newbalanceOrig`: Sender's balance before/after (floats; frauds often drain to zero).
  - `nameDest`: Recipient account ID (string, anonymized).
  - `oldbalanceDest`, `newbalanceDest`: Recipient's balance before/after (floats; inconsistencies signal fraud).
  - `isFraud`: Binary target (1 for fraud).
  - `isFlaggedFraud`: Binary (flagged if amount >200k in TRANSFER; mostly 0).
- **Notes**: Synthetic nature ensures privacy while mimicking real distributions. No missing values, but zero balances and inconsistencies require handling. Dataset is scaled down (1/4 of original in some versions) for manageability.

##  Methodology
This project follows a structured CRISP-DM (Cross-Industry Standard Process for Data Mining) approach, ensuring reproducibility and thoroughness.

1. **Data Wrangling** (`01_data_wrangling.ipynb`):
   - Loaded raw CSV from Kaggle.
   - Filtered to `TRANSFER` and `CASH_OUT`.
   - Dropped anonymized IDs (`nameOrig`, `nameDest`) as non-predictive.
   - Handled zero balances and outliers (e.g., via IQR for anomaly detection).
   - Added error checks for balance consistency (e.g., `newbalanceOrig` should = `oldbalanceOrg` - `amount`).
   - Output: Cleaned CSV in `data/processed/`.

2. **Exploratory Data Analysis (EDA)** (`02_eda.ipynb`):
   - Descriptive stats: Means, medians, distributions (e.g., `amount` log-scaled for skewness).
   - Visualizations: Histograms, boxplots, pairplots to spot fraud clusters.
   - Correlation analysis: Heatmap showing strong links (e.g., `amount` positively correlates with fraud).
   - Imbalance visualization: Pie chart of fraud ratio.
   - Insights: Frauds often involve full account drains and large transfers.

3. **Feature Engineering** (`03_feature_engineering.ipynb`):
   - New features: `balanceDiffOrg` (`oldbalanceOrg - newbalanceOrig`), `balanceDiffDest`, transaction ratios (e.g., `amount / oldbalanceOrg`).
   - Log transformations for skewed variables.
   - One-hot encoding for `type`.
   - Standardization: `StandardScaler` for numericals to aid ML algorithms.
   - Dimensionality check: No PCA needed (low collinearity).

4. **Model Building** (`04_modeling.ipynb`):
   - Train/Test Split: 80/20 stratified to preserve fraud ratio.
   - Models:
     - **Logistic Regression**: Baseline, interpretable; tuned with L1/L2 regularization.
     - **Random Forest**: Ensemble for nonlinearity; tuned n_estimators, max_depth.
     - **XGBoost**: Gradient boosting for imbalance; tuned learning_rate, scale_pos_weight.
   - Hyperparameter tuning: GridSearchCV with cross-validation.
   - Handling Imbalance: Class weights and SMOTE (tested).

5. **Model Evaluation** (`05_evaluation.ipynb`):
   - Metrics: Precision, Recall, F1, ROC-AUC, Confusion Matrix.
   - Cross-validation: Stratified K-Fold.
   - Leakage Check: Time-series split on `step` to simulate real deployment.
   - Final Selection: Based on recall-priority.

6. **Visualization & Storytelling**:
   - Plots saved in `results/visualizations/`.
   - Narrative: Used in report to explain "how fraud hides in large transfers and balance anomalies."

##  Results
After tuning and leakage mitigation, XGBoost outperformed others in balancing detection and efficiency.

| Model               | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.88      | 0.84   | 0.86     | 0.93    |
| Random Forest       | 0.94      | 0.89   | 0.91     | 0.97    |
| **XGBoost**         | **0.96**  | **0.92** | **0.94** | **0.98** |

**Final Model:** XGBoost (saved as `fraud_model.pkl` in `results/`).
- **Performance Insights**: Achieves 92% recall on holdout, catching most frauds with minimal false positives.
- **Feature Importance**: `amount`, `oldbalanceOrg`, and `balanceDiffOrg` top contributors.

##  Key Visualizations
- **Feature Importance (XGBoost)**: Bar plot highlighting top predictors.
   <!-- Replace with actual image -->
- **Confusion Matrix**: Heatmap showing TP/TN/FP/FN.
  
- **ROC Curve**: Curve with AUC=0.98 for XGBoost.
  
- **Fraud Distribution by Amount**: Boxplot comparing fraud vs. non-fraud amounts.
  

All visuals are generated in notebooks and saved in `results/visualizations/`.

##  Challenges and Learnings
- **Imbalance**: Addressed with weighting/SMOTE; learned sampling techniques.
- **Data Leakage**: Initial perfect scores from engineered features; fixed with time-based splits.
- **Synthetic Data**: Limits real-world generalizability; learned to validate on holdouts.
- **Compute**: Downsampled for efficiency; scalable with cloud tools.
- **Key Learning**: Feature engineering (e.g., diffs) boosts recall by 15-20%.

##  Future Work
- Integrate real-time streaming (e.g., Kafka) for production.
- Add advanced features: Network graphs of accounts, temporal patterns.
- Ensemble models or AutoML for further tuning.
- Test on real datasets (with privacy compliance).
- Deploy via Flask/API for bank integration.

##  Repository Structure

## Repository Structure
```
capstone-two/
│
├── notebooks/
│   ├── 01_data_wrangling.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│
├── data/
│   ├── raw/        # original dataset(s)
│   ├── processed/  # cleaned/feature-engineered datasets
│
├── results/
│   ├── model_metrics.csv
│   ├── visualizations/
│
├── report/
│   ├── Capstone_Two_Report.pdf
│
├── requirements.txt
└── README.md
```
## Deliverables
- Capstone_Two_Report.pdf: Main written report covering problem, process, results, and recommendations (in report/).
- model_metrics.csv: CSV with model comparison table (in results/).
- Jupyter Notebooks: Clean, commented, and executable (in notebooks/).
- README.md: Project summary, navigation guide, and setup instructions (this file).

## References
- PaySim Dataset: Lopez-Rojas et al. (2016). "PaySim: A financial mobile money simulator for fraud detection."
- Kaggle Page: ealaxi/paysim1
- Libraries: Scikit-learn docs, XGBoost paper (Chen & Guestrin, 2016).

## Acknowledgements
- Dataset Creators: Edgar Lopez-Rojas for PaySim.
- Mentors & Program: Springboard Data Science Career Track team.
- Open-Source Tools: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn.
- Community: Kaggle and Medium articles for inspiration.
