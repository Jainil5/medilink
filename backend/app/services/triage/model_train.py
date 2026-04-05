import os
import pandas as pd
import numpy as np
import joblib

from nhanes.load import load_NHANES_data

from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# =============================
# PATHS
# =============================
BASE_PATH = "backend/app/services/triage"
MODEL_PATH = os.path.join(BASE_PATH, "models")

os.makedirs(MODEL_PATH, exist_ok=True)

# =============================
# LOAD NHANES
# =============================
def load_nhanes():

    demo = load_NHANES_data("DEMO")
    bp = load_NHANES_data("BPX")
    lab = load_NHANES_data("LAB")

    df = demo.merge(bp, on="SEQN")
    df = df.merge(lab, on="SEQN")

    df = df.rename(columns={
        "RIDAGEYR": "Age",
        "RIAGENDR": "Gender",
        "BMXBMI": "BMI",
        "BPXSY1": "SystolicBP",
        "BPXDI1": "DiastolicBP",
        "LBXGLU": "Glucose",
        "LBXHGB": "Hemoglobin",
        "LBXTC": "Cholesterol",
        "LBXIRON": "Iron",
        "SMQ020": "Smoking"
    })

    cols = [
        "Age","Gender","BMI","SystolicBP","DiastolicBP",
        "Glucose","Hemoglobin","Cholesterol","Iron","Smoking"
    ]

    return df[cols]

# =============================
# CLEAN DATA
# =============================
def clean_data(df):

    df["Smoking"] = df["Smoking"].replace({1:1, 2:0})
    df["Gender"] = df["Gender"].replace({1:1, 2:0})

    df = df.fillna(df.median())

    return df

# =============================
# CREATE TARGETS
# =============================
def create_targets(df):

    y = pd.DataFrame()

    y["Diabetes"] = (df["Glucose"] > 126).astype(int)
    y["Hypertension"] = ((df["SystolicBP"] > 140) | (df["DiastolicBP"] > 90)).astype(int)
    y["Obesity"] = (df["BMI"] > 30).astype(int)

    y["Anemia"] = (
        ((df["Gender"] == 1) & (df["Hemoglobin"] < 13)) |
        ((df["Gender"] == 0) & (df["Hemoglobin"] < 12))
    ).astype(int)

    y["IronDeficiency"] = (df["Iron"] < 60).astype(int)
    y["Dyslipidemia"] = (df["Cholesterol"] > 240).astype(int)

    abnormal = (
        y["Diabetes"] + y["Hypertension"] +
        y["Obesity"] + y["Dyslipidemia"]
    )

    y["MetabolicSyndrome"] = (abnormal >= 3).astype(int)

    cardio_score = (
        (df["Age"] > 50).astype(int) +
        y["Hypertension"] +
        y["Dyslipidemia"] +
        df["Smoking"]
    )

    y["CardioRisk"] = (cardio_score >= 2).astype(int)

    return y

# =============================
# TRAIN MODEL
# =============================
def train():

    df = load_nhanes()
    df = clean_data(df)

    X = df.copy()
    y = create_targets(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models
    rf = RandomForestClassifier(n_estimators=100)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    mlp = MLPClassifier(max_iter=300)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("mlp", mlp)],
        voting="soft"
    )

    model = MultiOutputClassifier(ensemble)
    model.fit(X_scaled, y)

    # Save artifacts
    joblib.dump(model, os.path.join(MODEL_PATH, "cdss_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_PATH, "features.pkl"))
    joblib.dump(list(y.columns), os.path.join(MODEL_PATH, "targets.pkl"))

if __name__ == "__main__":
    train()