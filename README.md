# Capstone Two: Fraud Detection with XGBoost

Welcome to my fraud detection capstone project, built using Python, XGBoost, and pandas. This project aims to accurately identify fraudulent transactions from anonymized financial data.

---

## Dataset

- The dataset contains anonymized transaction data, including numerical and engineered features.
- Fraud cases (`isFraud = 1`) represent approximately **10%** of all transactions, leading to a class imbalance problem.
- Time-based information (`step`) was used to create realistic holdout and test splits.

---

## Objective

The primary goal is to **detect fraudulent transactions** while minimizing false negatives.  
This project emphasizes:
- **High recall** (catching all frauds)
- **Handling class imbalance**
- **Preventing data leakage**
- **Maintaining model interpretability**

---

## Methodology

- Preprocessing & Feature Engineering (`03_preprocessing.ipynb`)
- Model training & evaluation (`04_modeling.ipynb`)
- Class balancing using `scale_pos_weight` and SMOTE
- Cross-validation with stratified and time-based splits
- Model interpretability using feature importance and SHAP (optional)
- Holdout set evaluation to simulate production behavior

---

## Final Model Recommendation

- ✅ **Model:** XGBoost (best trade-off between recall, F1, and ROC-AUC)
- ✅ **No leakage features**
- ✅ **scale_pos_weight = 9.0** for class imbalance
- ✅ **Recall on Holdout Set:** 0.98
- ✅ **F1 on Holdout Set:** 0.99
- ✅ **ROC-AUC:** 1.00

> See [`final_model_recommendation.md`](./final_model_recommendation.md) for full details.

---

## Project Structure

```
capstone-two/
│
├── data/                          # Raw and cleaned datasets
├── notebooks/
│   ├── 03_preprocessing.ipynb     # Data cleaning and feature engineering
│   ├── 04_modeling.ipynb          # Model training, tuning, and evaluation
│   └── fraud_model.pkl            # Final trained model
│
├── final_model_recommendation.md  # Model justification and deployment notes
└── README.md                      # Project overview
```

---

## How to Run

```bash
git clone https://github.com/mesfin-k/capstone-two.git
cd capstone-two
pip install -r requirements.txt  # Optional if you include one
jupyter notebook notebooks/04_modeling.ipynb
```

---

## Future Work

- Add SHAP explanations for individual fraud predictions
- Deploy as a Flask API or streamlit app
- Monitor model drift over time with live data
- Use ensemble methods (e.g., XGBoost + Logistic Regression stacking)

---

## Author

**Mesfin Kebede**  
📍 San Leandro, CA  
🔗 [LinkedIn](https://www.linkedin.com/in/mesfin-kebede)  
📧 mesfin.k.kebede@gmail.com

---
