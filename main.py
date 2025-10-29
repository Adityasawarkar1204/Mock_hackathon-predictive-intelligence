import os, streamlit as st, pandas as pd
from hr_model import predict_hr
from finance_model import predict_finance
from sales_model import prepare_series, backtest_predictions, predict_last_window

st.set_page_config(page_title="IEAP – AI Demo", layout="wide")
st.title(" Intelligent Enterprise – Predictive Intelligence (HR • Finance • Sales)")

# ------- HR TAB -------
tab_hr, tab_fin, tab_sales = st.tabs([" HR Attrition", " Finance Fraud", " Sales Forecast"])

with tab_hr:
    st.subheader("Attrition Prediction")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 18, 70, 30)
    income = c2.number_input("Monthly Income", 0, 200000, 50000)
    years = c3.number_input("Years at Company", 0, 40, 3)
    joblevel = c1.selectbox("Job Level", [1,2,3,4,5], index=0)
    total_years = c2.number_input("Total Working Years", 0, 50, 6)

    if st.button("Predict Attrition", type="primary"):
        pred = predict_hr(age, income, years, joblevel, total_years)
        st.success("Result: Will Leave " if pred==1 else "Result: Will Stay ✅")

with tab_fin:
    st.subheader("Invoice Fraud Detection")
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Amount", 0.0, 1_000_000.0, 5000.0)
    oldbal = c2.number_input("Old Balance (Origin)", 0.0, 1_000_000.0, 2000.0)
    newbal = c3.number_input("New Balance (Origin)", 0.0, 1_000_000.0, 1000.0)

    if st.button("Check Fraud", type="primary"):
        pred = predict_finance(amount, oldbal, newbal)
        st.success("Prediction: FRAUD " if pred==1 else "Prediction: Legit ")

with tab_sales:
    st.subheader("Sales Forecast (backtest + next-day)")

    
    data_path = os.path.join(os.path.dirname(__file__), "Data", "stores_sales_forecasting.csv")
    date_col_guess = "Order Date"   
    sales_col = "Sales"

    
    date_col = st.text_input("Date column name", value=date_col_guess)
    if st.button("Run Forecast", type="primary"):
        df = prepare_series(data_path, date_col=date_col, sales_col=sales_col)
        out, mae = backtest_predictions(df, date_col=date_col, sales_col=sales_col, split_ratio=0.8)
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.line_chart(out.set_index(date_col)[[sales_col, "Predicted"]])
        next_pred = predict_last_window(df, sales_col=sales_col)
        st.info(f"Next-day forecast (one step ahead): **{next_pred:.2f}**")


