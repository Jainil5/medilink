import pandas as pd
import pyreadstat
import os

# =============================
# PATHS
# =============================
BASE_PATH = "backend/app/services/triage"
RAW_PATH = os.path.join(BASE_PATH, "data/raw")
OUTPUT_PATH = os.path.join(BASE_PATH, "data")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =============================
# LOAD XPT FILE
# =============================
def load_xpt(file_path):
    df, _ = pyreadstat.read_xport(file_path)
    return df

# =============================
# LOAD NHANES DATA
# =============================
def load_nhanes():

    demo = load_xpt(os.path.join(RAW_PATH, "DEMO.xpt"))
    bmx = load_xpt(os.path.join(RAW_PATH, "BMX.xpt"))
    bpx = load_xpt(os.path.join(RAW_PATH, "BPX.xpt"))
    lab = load_xpt(os.path.join(RAW_PATH, "BIOPRO.xpt"))

    # Merge on SEQN (patient ID)
    df = demo.merge(bmx, on="SEQN", how="inner")
    df = df.merge(bpx, on="SEQN", how="inner")
    df = df.merge(lab, on="SEQN", how="inner")

    return df

# =============================
# CLEAN & SELECT FEATURES
# =============================
def clean_data(df):

    df = df.rename(columns={
        "RIDAGEYR": "Age",
        "RIAGENDR": "Gender",
        "BMXBMI": "BMI",
        "BPXSY1": "SystolicBP",
        "BPXDI1": "DiastolicBP",
        "LBXGLU": "Glucose",
        "LBXHGB": "Hemoglobin",
        "LBXTC": "Cholesterol",
        "LBXSIR": "Iron",
        "SMQ020": "Smoking"
    })

    cols = [
        "SEQN",
        "Age",
        "Gender",
        "BMI",
        "SystolicBP",
        "DiastolicBP",
        "Glucose",
        "Hemoglobin",
        "Cholesterol",
        "Iron",
        "Smoking"
    ]

    df = df[cols]

    # Convert categorical values
    df["Gender"] = df["Gender"].replace({1: 1, 2: 0})
    df["Smoking"] = df["Smoking"].replace({1: 1, 2: 0})

    # Remove unrealistic values
    df = df[(df["BMI"] > 10) & (df["BMI"] < 60)]

    # Handle missing values
    df = df.fillna(df.median())

    return df

# =============================
# MAIN FUNCTION
# =============================
def convert_to_csv():

    df = load_nhanes()
    df = clean_data(df)

    output_file = os.path.join(OUTPUT_PATH, "nhanes_cleaned.csv")

    df.to_csv(output_file, index=False)

    print(f"Saved cleaned NHANES dataset to: {output_file}")
    print(f"Shape: {df.shape}")

# =============================
# RUN
# =============================
if __name__ == "__main__":
    convert_to_csv()