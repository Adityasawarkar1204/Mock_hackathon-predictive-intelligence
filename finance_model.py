import os, joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "finance_model.pkl")

_model = joblib.load(MODEL_PATH)

def predict_finance(amount, oldbal, newbal):
    transaction_diff = oldbal - newbal
    percent_change = (transaction_diff / oldbal) if oldbal > 0 else 0
    X = np.array([[amount, oldbal, newbal, transaction_diff, percent_change]])
    return _model.predict(X)[0]




