import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# BUILD MODEL
# ==========================================
def build_model():
    np.random.seed(42)

    n = 3000
    df = pd.DataFrame({
        "Age": np.random.randint(18, 80, n),
        "Gender": np.random.choice([0, 1], n),
        "BMI": np.random.normal(27, 5, n),
        "SystolicBP": np.random.normal(120, 15, n),
        "DiastolicBP": np.random.normal(80, 10, n),
        "Glucose": np.random.normal(100, 30, n),
        "Hemoglobin": np.random.normal(14, 2, n),
        "Cholesterol": np.random.normal(200, 40, n),
        "Iron": np.random.normal(80, 20, n),
        "Smoking": np.random.choice([0, 1], n)
    })

    df["Diabetes"] = (df["Glucose"] > 126).astype(int)
    df["Hypertension"] = ((df["SystolicBP"] > 140) | (df["DiastolicBP"] > 90)).astype(int)
    df["Obesity"] = (df["BMI"] > 30).astype(int)

    features = [
        "Age", "Gender", "BMI",
        "SystolicBP", "DiastolicBP",
        "Glucose", "Hemoglobin",
        "Cholesterol", "Iron", "Smoking"
    ]

    targets = ["Diabetes", "Hypertension", "Obesity"]

    X = df[features]
    y = df[targets]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    explainers = {
        target: shap.TreeExplainer(model.estimators_[i])
        for i, target in enumerate(targets)
    }

    return model, explainers, features, targets


# ==========================================
# TRIAGE
# ==========================================
def triage_score(pred):
    score = 3*pred[0] + 2*pred[1] + 1*pred[2]
    if score >= 4:
        return "HIGH"
    elif score >= 2:
        return "MEDIUM"
    return "LOW"


# ==========================================
# SHAP EXTRACTION
# ==========================================
def extract_shap(explainer, patient_df, features, mode="increase", top_n=3):
    shap_values = explainer.shap_values(patient_df)

    if isinstance(shap_values, list):
        contrib = shap_values[1][0]
    elif len(np.shape(shap_values)) == 3:
        contrib = shap_values[0, :, 1]
    else:
        contrib = shap_values[0]

    results = []
    for i, val in enumerate(contrib):
        if abs(val) > 0.01:
            if mode == "increase" and val > 0:
                results.append((features[i], float(val)))
            elif mode == "decrease" and val < 0:
                results.append((features[i], float(val)))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results[:top_n]


# ==========================================
# LLM-STYLE EXPLANATION GENERATOR
# ==========================================
def generate_explanation(disease, prob, drivers, protective):
    
    sentence = f"{disease} risk is {'high' if prob > 0.7 else 'moderate' if prob > 0.4 else 'low'} ({round(prob*100,1)}%). "

    if drivers:
        driver_text = ", ".join([f"{f[0]}" for f in drivers])
        sentence += f"Key factors increasing risk include {driver_text}. "

    if protective:
        protect_text = ", ".join([f"{f[0]}" for f in protective])
        sentence += f"Protective factors include {protect_text}."

    return sentence


# ==========================================
# MAIN FUNCTION
# ==========================================
def predict_patient(patient_dict, model, explainers, features, targets):
    
    patient_df = pd.DataFrame([patient_dict])
    pred = model.predict(patient_df)[0]

    output = {
        "risk": None,
        "diseases": {}
    }

    output["risk"] = triage_score(pred)

    for i, disease in enumerate(targets):
        
        proba = model.estimators_[i].predict_proba(patient_df)[0][1]

        drivers = extract_shap(explainers[disease], patient_df, features, "increase")
        protective = extract_shap(explainers[disease], patient_df, features, "decrease")

        explanation = generate_explanation(disease, proba, drivers, protective)

        output["diseases"][disease] = {
            "prediction": int(pred[i]),
            "probability": round(float(proba), 3),
            "drivers": drivers,
            "protective": protective,
            "explanation": explanation
        }

    return output


# ==========================================
# INIT
# ==========================================
model, explainers, features, targets = build_model()


# ==========================================
# TEST PATIENTS
# ==========================================

high_risk_patient = {
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
}

low_risk_patient = {
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


# ==========================================
# OUTPUT
# ==========================================
high_output = predict_patient(high_risk_patient, model, explainers, features, targets)
low_output = predict_patient(low_risk_patient, model, explainers, features, targets)

print(high_output)
print(low_output)
