from langchain_ollama import ChatOllama
import langchain

llm = ChatOllama(model="gemma3:1b", temperature=0)

def get_soap_notes(summary, diagnosis):
    conversation = [
        {"role": "system", "content": f"""
        
        You are a medical clinical assistant specialized in dermatology.

Your task is to generate structured SOAP notes based on:

1) Patient conversation summary
2) AI skin lesion image prediction with confidence

Use professional clinical language.
Do not invent information.
Clearly include AI findings in the Objective section.
Keep it concise and useful for dermatologists.
Directly start with SOAP notes.
No explanations, suggestions ,hallucinations and all.
Just the SOAP notes.

---

EXAMPLE 1

Patient Description:
The patient reports a dark irregular skin lesion on the upper arm present for 3 months. Occasional itching and minor bleeding have been noted. No prior treatment.

AI Prediction:
Melanoma (Confidence: 0.92)

SOAP Notes:

S (Subjective):
- Patient reports dark pigmented lesion on upper arm for 3 months
- Occasional itching and minor bleeding
- No previous medical treatment

O (Objective):
- Visible irregular pigmented lesion in provided skin image
- AI-assisted analysis suggests melanoma with 92% confidence

A (Assessment):
- High suspicion for malignant melanoma based on symptoms and AI findings

P (Plan):
- Recommend urgent dermatology consultation
- Consider biopsy for definitive diagnosis
- Advise patient to avoid excessive sun exposure

---

EXAMPLE 2

Patient Description:
The patient mentions a light brown rough patch on the back that has slowly increased in size over the past year. No pain or bleeding reported.

AI Prediction:
Benign Keratosis-like Lesions (Confidence: 0.87)

SOAP Notes:

S (Subjective):
- Patient reports rough light brown patch on back for approximately one year
- Gradual increase in size
- No pain or bleeding

O (Objective):
- Skin image shows well-demarcated keratotic lesion
- AI model indicates benign keratosis-like lesion with 87% confidence

A (Assessment):
- Likely benign keratosis lesion

P (Plan):
- Routine dermatological evaluation recommended
- Monitor for changes in size, color, or symptoms

---

NOW GENERATE SOAP NOTES FOR THE FOLLOWING CASE:        
        
        """},
        {"role": "user", "content": f"""
        Patient Description:
        {summary}

        AI Prediction:
        {diagnosis}
        """}
    ]

    response = llm.invoke(conversation)
    return response.content


sample_cases = [
    {
        "conversation_summary": "Patient noticed a black mole on the lower leg that has grown over the past 2 months. Sometimes it bleeds and feels itchy.",
        "ai_prediction": "Melanoma (Confidence: 0.94)"
    },

    {
        "conversation_summary": "Patient reports a rough brown patch on the shoulder present for nearly one year with no pain or bleeding.",
        "ai_prediction": "Benign Keratosis-like Lesions (Confidence: 0.89)"
    },

    {
        "conversation_summary": "Patient has a small red raised spot on the arm for 3 weeks that occasionally bleeds when scratched.",
        "ai_prediction": "Basal Cell Carcinoma (Confidence: 0.85)"
    }
]


# for case in sample_cases:

#     print("\nPatient Conversation Summary:")
#     print(case["conversation_summary"])

#     print("\nAI Prediction:")
#     print(case["ai_prediction"])

#     soap_notes = get_summary(
#         case["conversation_summary"],
#         case["ai_prediction"]
#     )

#     print("\n====== GENERATED SOAP NOTES ======")
#     print(soap_notes)

#     break