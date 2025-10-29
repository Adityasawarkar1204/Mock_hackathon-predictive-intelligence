Hackathon Report – Predictive Intelligence (HR • Finance • Sales)

1. Project Overview

This project demonstrates the application of AI and Data Science in three business-critical areas:

HR Attrition Prediction – Predicting if an employee is likely to leave.

Finance Fraud Detection – Identifying fraudulent financial transactions.

Sales Forecasting – Forecasting future sales using time series analysis.

The models are integrated into an interactive Streamlit dashboard for real-time predictions.

2. Data Sources

HR Dataset – IBM HR Attrition dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv)

Finance Dataset – Synthetic Financial Transactions (Synthetic_Financial_datasets_log.csv)

Sales Dataset – Store Sales Forecasting dataset (stores_sales_forecasting.csv)

3. Workflow
Step 1: Data Preprocessing & Feature Engineering

Handled missing values using imputation and rolling mean.

Encoded categorical variables (HR dataset).

Created lag features (lag-1, lag-7) and rolling averages for sales.

Derived transaction_diff and percent_change features for finance fraud.

Step 2: Model Training

HR Attrition → Logistic Regression vs Random Forest (Random Forest selected).

Finance Fraud → Logistic Regression achieved ROC-AUC ~0.94, better than Random Forest.

Sales Forecasting → Linear Regression with lag features (MAE ≈ 291).

Step 3: Model Evaluation

Metrics used: Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, MAE.

HR: Random Forest showed better balance between precision & recall.

Finance: Logistic Regression gave higher ROC-AUC and faster performance.

Sales: Forecasting captured overall trends, though spikes were harder to predict.

Step 4: Model Saving

Models saved as .pkl files using joblib:

models/saved_models/
    hr_model.pkl
    finance_model.pkl
    sales_model.pkl

Step 5: Dashboard Integration

Built a Streamlit Dashboard with 3 tabs:

HR Attrition → Predict employee attrition risk.

Finance Fraud → Detect fraudulent transactions.

Sales Forecast → Visualize past vs predicted sales + next-day forecast.

4. Observations & Results
HR Attrition (Random Forest)

Confusion Matrix showed model captures majority of attrition cases but still some false negatives.

ROC Curve improved over Logistic Regression baseline.

Best Model: Random Forest (class_weight="balanced").

Finance Fraud (Logistic Regression)

Confusion Matrix showed strong detection of fraud cases with fewer false negatives.

ROC Curve area ≈ 0.94, indicating high discriminative ability.

Best Model: Logistic Regression.

Sales Forecasting (Linear Regression)

Predictions followed sales trends but struggled with sudden spikes.

Mean Absolute Error (MAE): ~292 units.

Good baseline, can be improved using ARIMA, Prophet, or LSTM.

5. Conclusion

This project successfully demonstrates how AI and Predictive Analytics can solve real-world enterprise problems across HR, Finance, and Sales.

HR Attrition: Helps in proactive employee retention.

Finance Fraud: Detects anomalies, reducing financial risk.

Sales Forecasting: Aids in demand planning and inventory management.

The interactive Streamlit dashboard makes the system easy to use by business stakeholders.

6. Future Improvements

Use XGBoost / LightGBM for HR and Finance to improve recall on minority classes.

Deploy models via REST API (FastAPI/Flask) for enterprise integration.

Use advanced time-series models (Prophet, LSTM) for more accurate sales forecasting.

Add MLOps pipeline for automation of retraining and monitoring.