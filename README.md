## 🛡️ Capstone Two: Sentinel Fraud Detection (Data Wrangling Phase)

This notebook focuses on cleaning and exploring a filtered PaySim dataset containing 1,000 bank transactions (100 fraudulent). The goal is to prepare the data for machine learning-based fraud detection.

### 🔍 Key Steps:
- Data loaded and preprocessed from TRANSFER and CASH_OUT only
- Outliers analyzed using the IQR method
- Balance shifts evaluated for anomalies
- Fraud transaction patterns explored and visualized
- Top fraud indicators identified: `oldbalanceOrg`, `amount`
- Cleaned dataset saved for modeling phase

📄 View full notebook:  
👉 [sentinel_data_wrangling.ipynb](https://github.com/mesfin-k/capstone-two/blob/main/notebooks/sentinel_data_wrangling.ipynb)
