import os, joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "sales_model.pkl")

if os.path.exists(MODEL_PATH):
    _model = joblib.load(MODEL_PATH)
else:
    _model = LinearRegression()

def prepare_series(path, date_col="Order Date", sales_col="Sales"):
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    df['lag_1'] = df[sales_col].shift(1).fillna(0)
    df['lag_7'] = df[sales_col].shift(7).fillna(0)
    df['rolling_mean_7'] = df[sales_col].rolling(7, min_periods=1).mean()
    return df

def backtest_predictions(df, date_col="Order Date", sales_col="Sales", split_ratio=0.8):
    X = df[['lag_1','lag_7','rolling_mean_7']]
    y = df[sales_col]
    split = int(len(df)*split_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    out = df.iloc[split:].copy()
    out["Predicted"] = preds
    mae = np.mean(np.abs(y_test - preds))
    return out, mae

def predict_last_window(df, sales_col="Sales"):
    last_row = df[['lag_1','lag_7','rolling_mean_7']].iloc[-1].values.reshape(1,-1)
    return _model.predict(last_row)[0]

