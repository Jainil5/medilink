import pandas as pd
import joblib


# ==========================================
# LOAD MODEL
# ==========================================
data = joblib.load("clinical_model.pkl")

model = data["model"]
features = data["features"]
targets = data["targets"]


# ==========================================
# TRIAGE SYSTEM
# ==========================================
def triage_score(pred):
    score = 3*pred[0] + 2*pred[1] + 1*pred[2]

    if score >= 4:
        return "HIGH"
    elif score >= 2:
        return "MEDIUM"
    return "LOW"


# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_patient(patient_dict):
    df = pd.DataFrame([patient_dict])

    pred = model.predict(df)[0]

    probabilities = [
        model.estimators_[i].predict_proba(df)[0][1]
        for i in range(len(targets))
    ]

    diseases = {}

    for i, disease in enumerate(targets):
        diseases[disease] = {
            "prediction": int(pred[i]),
            "probability": round(float(probabilities[i]), 3)
        }

    return {
        "risk": triage_score(pred),
        "diseases": diseases
    }


# ==========================================
# SAMPLE PATIENTS
# ==========================================
patients = {
    "high_risk": {
        "Age": 65,
        "Gender": 1,
        "BMI": 32,
        "SystolicBP": 150,
        "DiastolicBP": 95,
        "Glucose": 180,
        "Hemoglobin": 11,
        "Cholesterol": 260,
        "Iron": 50,
        "Smoking": 1
    },
    "medium_risk": {
        "Age": 45,
        "Gender": 1,
        "BMI": 27,
        "SystolicBP": 135,
        "DiastolicBP": 85,
        "Glucose": 115,
        "Hemoglobin": 13,
        "Cholesterol": 210,
        "Iron": 70,
        "Smoking": 0
    },
    "low_risk": {
        "Age": 25,
        "Gender": 0,
        "BMI": 22,
        "SystolicBP": 110,
        "DiastolicBP": 70,
        "Glucose": 90,
        "Hemoglobin": 14,
        "Cholesterol": 180,
        "Iron": 80,
        "Smoking": 0
    }
}


# ==========================================
# RUN TEST
# ==========================================
results = {k: predict_patient(v) for k, v in patients.items()}

print(results)