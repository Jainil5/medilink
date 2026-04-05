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

    st.subheader("🧾 Patient Intake")

    answers = []

    for q in QUESTIONS:
        ans = st.text_input(q)
        answers.append(ans)

    st.subheader("📸 Upload Image")

    uploaded_file = st.file_uploader(
        "Upload skin image",
        type=["png", "jpg", "jpeg"]
    )

    submitted = st.form_submit_button("Generate Clinical Report ➜")

# ---------------- PROCESS ---------------- #
if submitted:

    # 🔥 Build conversation EXACTLY matching backend schema
    conversation = []

    for i in range(len(QUESTIONS)):
        if answers[i]:
            conversation.append({
                "role": "doctor",
                "message": QUESTIONS[i]
            })
            conversation.append({
                "role": "patient",
                "message": answers[i]
            })

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

    with st.spinner("Analyzing..."):
        r = requests.post("http://localhost:8000/generate-report", json=payload)

        if r.status_code == 200:
            res = r.json()

            st.subheader("🧠 Clinical Report")

            confidence = res["confidence"]

            # 🔥 CONFIDENCE LOGIC
            if confidence < 0.5:
                st.error("Diagnosis: No disease predicted in image")
                st.warning("⚠️ The AI could not confidently detect any specific skin condition.")
            else:
                st.metric("Diagnosis", res["prediction"])
                st.metric("Confidence", f"{confidence:.1%}")

                if os.path.exists(res.get("xai_image", "")):
                    st.image(res["xai_image"], caption="XAI Map")

                st.markdown(res["soap_notes"])

        else:
            st.error("❌ API Error")


# ---------------- DEMO BUTTON ---------------- #
st.markdown("---")

if st.button("🎯 Run Demo Case"):

    payload = {
        "conversation": DEMO_CHAT,
        "image_path": DEMO_IMAGE
    }

    with st.spinner("Running demo..."):
        r = requests.post("http://localhost:8000/generate-report", json=payload)

        if r.status_code == 200:
            res = r.json()

            st.subheader("🧠 Demo Report")

            confidence = res["confidence"]

            if confidence < 0.5:
                st.error("Diagnosis: No disease predicted in image")
                st.warning("⚠️ The AI could not confidently detect any specific skin condition.")
            else:
                st.metric("Diagnosis", res["prediction"])
                st.metric("Confidence", f"{confidence:.1%}")

                if os.path.exists(res.get("xai_image", "")):
                    st.image(res["xai_image"])

                st.markdown(res["soap_notes"])