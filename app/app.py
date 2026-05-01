import streamlit as st
import requests
import os
from pathlib import Path

st.set_page_config(
    page_title="Medilink Clinical AI",
    page_icon="⚕️",
    layout="wide"
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- CUSTOM UI ----------
st.markdown("""
<style>
.card {
    background: #ffffff;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}

.chat-user {
    background: #2563eb;
    color: white;
    padding: 10px;
    border-radius: 12px;
    margin: 6px 0;
    width: fit-content;
}

.chat-bot {
    background: #f1f5f9;
    padding: 10px;
    border-radius: 12px;
    margin: 6px 0;
}

.upload-box {
    border: 2px dashed #cbd5e1;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.metric-box {
    background: #f8fafc;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("⚕️ MEDILINK AI")
st.divider()

# ---------------- QUESTIONS ---------------- #
QUESTIONS = [
    "What is your age and gender?",
    "Where is the lesion and how long has it been present?",
    "Any symptoms like pain, itching, or bleeding?"
]

# ---------------- DEMO ---------------- #
DEMO_CHAT = [
    {"role": "doctor", "message": "What is your age and gender?"},
    {"role": "patient", "message": "I am 25 years old and male."},
    {"role": "doctor", "message": "How long has the patch been there?"},
    {"role": "patient", "message": "Almost a year."},
    {"role": "doctor", "message": "Any pain?"},
    {"role": "patient", "message": "No pain or bleeding."},
    {"role": "doctor", "message": "It appears rough and light brown."}
]

DEMO_IMAGE = "/Users/jainil/Documents/development/medilink/datasets/ddidiversedermatologyimages/images/000001.png"

# ---------------- FORM ---------------- #
with st.form("patient_form"):

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Patient Intake")

    answers = []

    for q in QUESTIONS:
        ans = st.text_input(q)
        answers.append(ans)

    st.markdown('</div>', unsafe_allow_html=True)

    # Upload UI
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📸 Upload Image")

    uploaded_file = st.file_uploader(
        "Upload skin image",
        type=["png", "jpg", "jpeg"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Generate Clinical Report ➜")

# ---------------- PROCESS ---------------- #
if submitted:

    conversation = []

    for i in range(len(QUESTIONS)):
        if answers[i]:
            conversation.append({"role": "doctor", "message": QUESTIONS[i]})
            conversation.append({"role": "patient", "message": answers[i]})

    # Save image
    image_path = None
    if uploaded_file:
        save_path = Path(UPLOAD_DIR) / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_path = str(save_path.resolve())

    payload = {
        "conversation": conversation,
        "image_path": image_path
    }

    # 🔥 Show conversation preview
    if conversation:
        st.markdown("### 💬 Consultation Summary")
        for msg in conversation:
            if msg["role"] == "doctor":
                st.markdown(f'<div class="chat-bot">👨‍⚕️ {msg["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-user">🧑 {msg["message"]}</div>', unsafe_allow_html=True)

    with st.spinner("Analyzing..."):
        r = requests.post("http://localhost:8000/generate-report", json=payload)

        if r.status_code == 200:
            res = r.json()

            st.markdown("### 🧠 Clinical Report")

            confidence = res["confidence"]

            col1, col2 = st.columns(2)

            if confidence < 0.5:
                st.error("Diagnosis: No disease predicted in image")
                st.warning("⚠️ Low confidence detection")
            else:
                col1.metric("Diagnosis", res["prediction"])
                col2.metric("Confidence", f"{confidence:.1%}")

                if os.path.exists(res.get("xai_image", "")):
                    st.image(res["xai_image"], caption="🔍 XAI Explanation Map")

                st.markdown("### 📝 SOAP Notes")
                st.markdown(f"""
                <div class="card">
                {res["soap_notes"]}
                </div>
                """, unsafe_allow_html=True)

        else:
            st.error("❌ API Error")

# ---------------- DEMO ---------------- #
st.markdown("---")

if st.button("🎯 Run Demo Case"):

    payload = {
        "conversation": DEMO_CHAT,
        "image_path": DEMO_IMAGE
    }

    # Chat preview
    st.markdown("### 💬 Demo Consultation")
    for msg in DEMO_CHAT:
        if msg["role"] == "doctor":
            st.markdown(f'<div class="chat-bot">👨‍⚕️ {msg["message"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-user">🧑 {msg["message"]}</div>', unsafe_allow_html=True)

    with st.spinner("Running demo..."):
        r = requests.post("http://localhost:8000/generate-report", json=payload)

        if r.status_code == 200:
            res = r.json()

            st.markdown("### 🧠 Demo Report")

            confidence = res["confidence"]

            col1, col2 = st.columns(2)

            if confidence < 0.5:
                st.error("Diagnosis: No disease predicted in image")
                st.warning("⚠️ Low confidence detection")
            else:
                col1.metric("Diagnosis", res["prediction"])
                col2.metric("Confidence", f"{confidence:.1%}")

                if os.path.exists(res.get("xai_image", "")):
                    st.image(res["xai_image"])

                st.markdown("### 📝 SOAP Notes")
                st.markdown(f"""
                <div class="card">
                {res["soap_notes"]}
                </div>
                """, unsafe_allow_html=True)

        else:
            st.error("❌ API Error")