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

st.markdown("""
<style>
.stApp{background:#F8FAFC;padding-top:0px}
.header{text-align:center;margin-bottom:2px}
.chat-row{display:flex;gap:10px;margin-bottom:10px}
.chat-row.user{justify-content:flex-end}
.chat-bubble{max-width:75%;padding:12px 16px;border-radius:14px}
.chat-ai{background:#DBEAFE;color:#1E3A8A}
.chat-user{background:#E5E7EB}
.avatar{width:38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#2563EB;color:white}
.avatar.user{background:#F1F5F9;color:#111827;border:1px solid #CBD5E1}
.upload-box{border:2px dashed #CBD5E1;padding:28px;border-radius:12px;background:white;text-align:center}
.stButton>button{background:#2563EB;color:white;border-radius:12px;padding:12px 18px;font-weight:600;border:none}
.stButton>button:hover{background:#1D4ED8}
[data-testid="stMetricValue"]{color:#2563EB;font-size:24px}
.report-card{border:1px solid #E2E8F0;border-radius:12px;padding:8px;background:white;height:100%}
</style>
""", unsafe_allow_html=True)

QUESTIONS = [
    "What is your age and gender?",
    "Where is the lesion and how long has it been present?",
    "Any symptoms like pain, itching, or bleeding?"
]

if "page" not in st.session_state:
    st.session_state.page = "intake"

if "conversation_data" not in st.session_state:
    st.session_state.conversation_data = [{"role":"doctor","message":QUESTIONS[0]}]
    st.session_state.q_idx = 0
    st.session_state.chat_done = False
    st.session_state.image_path = None

def reset_case():
    st.session_state.conversation_data = [{"role":"doctor","message":QUESTIONS[0]}]
    st.session_state.q_idx = 0
    st.session_state.chat_done = False
    st.session_state.image_path = None
    if "api_res" in st.session_state:
        del st.session_state.api_res

def load_demo():
    st.session_state.conversation_data = DEMO_CHAT.copy()
    st.session_state.chat_done = True
    st.session_state.image_path = DEMO_IMAGE

st.markdown("""
<div class="header">
<h1>⚕️ MEDILINK AI</h1>
<p style="color:#64748B;">Smart Clinical Decision Support</p>
</div>
""", unsafe_allow_html=True)

st.divider()


if st.session_state.page == "intake":

    col_chat, col_upload = st.columns([1,1.3], gap="large")

    with col_chat:
        st.subheader("🧾 Patient Intake")

        for msg in st.session_state.conversation_data:
            if msg["role"] == "doctor":
                st.markdown(f"""
                <div class="chat-row">
                    <div class="avatar">🧑‍⚕️</div>
                    <div class="chat-bubble chat-ai">{msg["message"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-row user">
                    <div class="chat-bubble chat-user">{msg["message"]}</div>
                    <div class="avatar user">👤</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if not st.session_state.chat_done:
            if reply := st.chat_input("Type patient response..."):
                st.session_state.conversation_data.append({"role":"patient","message":reply})
                st.session_state.q_idx += 1

                if st.session_state.q_idx < len(QUESTIONS):
                    st.session_state.conversation_data.append(
                        {"role":"doctor","message":QUESTIONS[st.session_state.q_idx]}
                    )
                else:
                    st.session_state.chat_done = True
                    st.session_state.conversation_data.append(
                        {"role":"doctor","message":"Intake completed."}
                    )
                st.rerun()

        if st.button("🎯 Run Demo Case"):
            load_demo()
            st.rerun()

    with col_upload:
        st.subheader("📸 Upload Image")

        uploaded_file = st.file_uploader(
            "Upload skin image",
            type=["png","jpg","jpeg"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            save_path = Path(UPLOAD_DIR) / uploaded_file.name
            with open(save_path,"wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.image_path = str(save_path.resolve())
            # st.image(uploaded_file, width=260)

        if st.button(
            "Generate Clinical Report ➜",
            use_container_width=True,
            disabled= not (st.session_state.chat_done and st.session_state.image_path)
        ):
            payload = {
                "conversation": st.session_state.conversation_data,
                "image_path": st.session_state.image_path
            }

            with st.spinner("AI analyzing..."):
                r = requests.post("http://localhost:8000/generate-report", json=payload)
                if r.status_code == 200:
                    st.session_state.api_res = r.json()
                    st.session_state.page = "report"
                    st.rerun()


if st.session_state.page == "report":

    res = st.session_state.api_res

    st.subheader("🧠 Clinical SOAP Report")

    left, right = st.columns([1.4,1], gap="small")

    with right:
        st.markdown(res["soap_notes"])

    with left:
        st.metric("Diagnosis", res["prediction"])
        st.metric("Confidence", f"{res['confidence']:.1%}")

        if os.path.exists(res.get("xai_image","")):
            st.image(res["xai_image"], caption="XAI Explainability Map")
        else:
            st.warning("XAI image not found")

    col_back, col_new = st.columns(2)

    with col_back:
        if st.button("⬅ Back to Intake", use_container_width=True):
            st.session_state.page = "intake"
            st.rerun()

    with col_new:
        if st.button("🆕 New Case", use_container_width=True):
            reset_case()
            st.session_state.page = "intake"
            st.rerun()
