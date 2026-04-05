import pandas as pd
import numpy as np
import shap
import joblib


# ==========================================
# LOAD TRAINED MODEL
# ==========================================
data = joblib.load("clinical_model.pkl")

model = data["model"]
features = data["features"]
targets = data["targets"]


# ==========================================
# TRIAGE FUNCTION
# ==========================================
def triage_score(pred):
    score = 3*pred[0] + 2*pred[1] + 1*pred[2]
    if score >= 4:
        return "HIGH"
    elif score >= 2:
        return "MEDIUM"
    return "LOW"


# ==========================================
# LOAD CSV
# ==========================================
df = pd.read_csv("backend/app/services/triage/data.csv")


# ==========================================
# PROCESS EACH PATIENT
# ==========================================
output_rows = []

for _, row in df.iterrows():
    
    patient_id = row["PatientID"]
    patient_data = row.drop("PatientID").to_dict()
    
    patient_df = pd.DataFrame([patient_data])

    # Predictions
    pred = model.predict(patient_df)[0]

    probabilities = [
        model.estimators_[i].predict_proba(patient_df)[0][1]
        for i in range(len(targets))
    ]

    # Build output row
    result = {
        "PatientID": patient_id,
        "Risk": triage_score(pred)
    }

    # Add original data
    for f in features:
        result[f] = patient_data[f]

    # Add predictions + probabilities
    for i, disease in enumerate(targets):
        result[f"{disease}_Prediction"] = int(pred[i])
        result[f"{disease}_Probability"] = round(float(probabilities[i]), 3)

    output_rows.append(result)


# ==========================================
# SAVE OUTPUT CSV
# ==========================================
output_df = pd.DataFrame(output_rows)

output_df.to_csv("backend/app/services/triage/patients_predictions.csv", index=False)