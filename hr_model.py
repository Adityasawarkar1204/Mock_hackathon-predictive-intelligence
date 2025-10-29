import os, joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "hr_model.pkl")

_model = joblib.load(MODEL_PATH)

def predict_hr(age, income, years, joblevel, total_years):
    X = np.array([[age, income, years, joblevel, total_years]])
    return _model.predict(X)[0]




