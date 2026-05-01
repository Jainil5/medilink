import streamlit as st
import pandas as pd
import os
from pathlib import Path
import sys

# ================= PATH =================
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from services.main_service import generate_clinical_report

# ================= CONFIG =================
st.set_page_config(
    page_title="Medilink Clinical AI",
    page_icon="⚕️",
    layout="wide"
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= UI =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&display=swap');

/* Main Background and Text */
.stApp {
    background: linear-gradient(135deg, #fff0f5 0%, #ffe4e1 100%);
    font-family: 'Syne', sans-serif;
    color: #2d3748;
}

/* Titles and Headers */
h1, h2, h3, h4, h5, h6 {
    color: #d53f8c;
    font-family: 'Syne', sans-serif;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    gap: 15px;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255, 255, 255, 0.6);
    border-radius: 12px 12px 0px 0px;
    padding: 12px 24px;
    border: none;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #718096;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ed64a6, #d53f8c);
    color: white !important;
    border-bottom-color: transparent !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ed64a6, #d53f8c);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 12px 28px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(213, 63, 140, 0.3);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(213, 63, 140, 0.4);
    color: white;
    border-color: transparent;
}

/* Input Fields */
.stTextInput>div>div>input {
    border-radius: 12px;
    border: 2px solid #fed7e2;
    padding: 14px;
    background-color: rgba(255, 255, 255, 0.8);
    color: #2d3748;
    font-family: 'Syne', sans-serif;
    transition: all 0.3s;
}
.stTextInput>div>div>input:focus {
    border-color: #d53f8c;
    box-shadow: 0 0 0 2px rgba(213, 63, 140, 0.2);
}

/* Selectbox */
.stSelectbox>div>div>div {
    border-radius: 12px;
    border: 2px solid #fed7e2;
    background-color: rgba(255, 255, 255, 0.8);
    font-family: 'Syne', sans-serif;
}

/* Cards / Forms */
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255, 255, 255, 0.5);
    box-shadow: 0 8px 32px rgba(213, 63, 140, 0.1);
    backdrop-filter: blur(10px);
}

.chat-user {
    background: linear-gradient(90deg, #ed64a6, #d53f8c);
    color: white;
    padding: 12px 16px;
    border-radius: 16px 16px 4px 16px;
    box-shadow: 0 2px 8px rgba(213, 63, 140, 0.2);
    font-family: 'Syne', sans-serif;
}
.chat-bot {
    background: white;
    color: #2d3748;
    padding: 12px 16px;
    border-radius: 16px 16px 16px 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    font-family: 'Syne', sans-serif;
}
.metric-box {
    background: rgba(255, 255, 255, 0.8);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(213, 63, 140, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.6);
    font-family: 'Syne', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("⚕️ MEDILINK AI")
st.divider()

# ================= TABS =================
tab1, tab2, tab3 = st.tabs([
    "🧠 Skin Disease",
    "🚨 Patient Priority",
    "📊 Model Evaluation"
])

# =========================================================
# ================= TAB 1: SKIN DISEASE ====================
# =========================================================
with tab1:

    QUESTIONS = [
        "What is your age and gender?",
        "Where is the lesion and how long has it been present?",
        "Any symptoms like pain, itching, or bleeding?"
    ]

    DEMO_ANSWERS = [
        "25 Male",
        "Back, 1 year",
        "No pain or bleeding"
    ]

    DEMO_IMAGE = "demo.jpg"  # update path if needed

    if st.button("🎯 Load Demo Case"):
        st.session_state.demo_mode = True

    with st.form("patient_form"):

        col1, col2 = st.columns([2, 1])

        # LEFT INPUT
        with col1:
            st.subheader("🧾 Patient Intake")

            answers = []
            for i, q in enumerate(QUESTIONS):
                default_val = DEMO_ANSWERS[i] if st.session_state.get("demo_mode") else ""
                answers.append(st.text_input(q, value=default_val))

        # RIGHT IMAGE
        with col2:
            st.subheader("📸 Upload Image")

            uploaded_file = st.file_uploader(
                "Upload skin image",
                type=["png", "jpg", "jpeg"]
            )

            if st.session_state.get("demo_mode") and os.path.exists(DEMO_IMAGE):
                st.image(DEMO_IMAGE, caption="Demo Image")

            elif uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image")

        submitted = st.form_submit_button("Generate Clinical Report ➜")

    if submitted:

        conversation = []
        for i in range(len(QUESTIONS)):
            if answers[i]:
                conversation.append({"role": "doctor", "message": QUESTIONS[i]})
                conversation.append({"role": "patient", "message": answers[i]})

        # IMAGE HANDLING
        image_path = None
        if st.session_state.get("demo_mode"):
            image_path = DEMO_IMAGE
        elif uploaded_file:
            save_path = Path(UPLOAD_DIR) / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_path = str(save_path.resolve())

        # MODEL CALL
        with st.spinner("🧠 Analyzing..."):
            res = generate_clinical_report({"conversation": conversation}, image_path)

        # RESULTS
        if res:

            st.markdown("### 🧠 Clinical Report")

            col1, col2 = st.columns([2, 1])

            # LEFT
            with col1:
                confidence = res["confidence"]

                if confidence < 0.5:
                    st.error("Diagnosis: No disease predicted")
                else:
                    m1, m2 = st.columns(2)
                    m1.metric("Diagnosis", res["prediction"])
                    m2.metric("Confidence", f"{confidence:.2%}")

                    if image_path and os.path.exists(image_path):
                        st.image(image_path, caption="Input Image")

                    if os.path.exists(res.get("xai_image", "")):
                        st.image(res["xai_image"], caption="🔍 XAI Heatmap")

            # RIGHT
            with col2:
                st.markdown("### 🧾 SOAP Notes")
                st.markdown(f"""
                <div style="background:#f8fafc;padding:15px;border-radius:12px;">
                {res["soap_notes"]}
                </div>
                """, unsafe_allow_html=True)

# =========================================================
# ================= TAB 2: PATIENT PRIORITY ================
# =========================================================
with tab2:

    st.subheader("🚨 Patient Priority Queue")

    @st.cache_data
    def load_data():
        return pd.read_csv("/Users/jainil/Documents/development/medilink/app/services/details.csv")

    try:
        df = load_data()

        priority_order = {"high": 0, "medium": 1, "low": 2}
        df["priority_rank"] = df["critical"].map(priority_order)
        df = df.sort_values(by="priority_rank")

        def get_color(level):
            if level == "high":
                return "#ef4444"
            elif level == "medium":
                return "#f59e0b"
            else:
                return "#10b981"

        st.warning("⚠️ High-risk patients should be prioritized")

        col1, col2 = st.columns(2)

        with col1:
            priority_filter = st.selectbox(
                "Filter by Priority",
                ["All", "high", "medium", "low"]
            )

        with col2:
            search = st.text_input("Search by Disease")

        filtered_df = df.copy()

        if priority_filter != "All":
            filtered_df = filtered_df[
                filtered_df["critical"] == priority_filter
            ]

        if search:
            filtered_df = filtered_df[
                filtered_df["disease"].str.contains(search, case=False)
            ]

        st.markdown("---")

        cols = st.columns(3)

        for i, (_, row) in enumerate(filtered_df.iterrows()):
            color = get_color(row["critical"])

            with cols[i % 3]:
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.7);
                    padding: 15px;
                    border-radius: 15px;
                    border-left: 6px solid {color};
                    margin-bottom: 15px;
                ">
                    <h4>🆔 {row['id']}</h4>
                    <p><b>{row['disease']}</b></p>
                    <p>Confidence: {row['confidence']:.2f}</p>
                    <p style="color:{color}; font-weight:bold;">
                        {row['critical'].upper()}
                    </p>
                    <p style="font-size:13px;">
                        {row['soap_notes']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        if len(filtered_df) == 0:
            st.info("No patients match the selected filters.")

    except:
        st.error("❌ patients.csv not found")

# =========================================================
# ================= TAB 3: MODEL EVALUATION ================
# =========================================================
with tab3:

    st.subheader("📊 Ensemble Model Evaluation")

    try:
        df = pd.read_csv("/Users/jainil/Documents/development/medilink/app/services/outputs/metrics/final_report.csv")

        MODEL_INFO = {
            "tf_efficientnet_b5": "EfficientNet - optimized CNN balancing accuracy and efficiency",
            "convnext_base": "ConvNeXt - modern CNN inspired by Transformers",
            "densenet169": "DenseNet - dense feature reuse",
            "vit_base_patch16_224": "Vision Transformer - image as patches",
            "swin_small_patch4_window7_224": "Swin Transformer - hierarchical transformer",
            "ensemble": "Ensemble Model - combines all models"
        }

        best_model = df.loc[df["accuracy"].idxmax()]

        st.success(
            f"🏆 Best Model: {best_model['model']} ({best_model['accuracy']:.2%})"
        )

        st.markdown("### 🧠 Model Details")

        cols = st.columns(3)

        for i, row in df.iterrows():
            model = row["model"]

            with cols[i % 3]:
                border = "2px solid #2563eb" if model == "ensemble" else "1px solid #ddd"

                st.markdown(f"""
                <div style="
                    padding:15px;
                    border-radius:12px;
                    border:{border};
                    background:white;
                    margin-bottom:10px;
                ">
                    <h4>{model}</h4>
                    <p style="font-size:13px;">{MODEL_INFO.get(model,"")}</p>
                    <p><b>Accuracy:</b> {row['accuracy']:.2%}</p>
                    <p><b>F1 Score:</b> {row['f1_macro']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### 📈 Accuracy Comparison")
        st.bar_chart(df.set_index("model")["accuracy"])

        st.markdown("### 📈 F1 Score Comparison")
        st.bar_chart(df.set_index("model")["f1_macro"])

        st.markdown("### 📋 Full Table")
        st.dataframe(df)

    except:
        st.error("❌ final_report.csv not found")