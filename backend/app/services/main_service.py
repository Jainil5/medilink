from .ham_test import run_test
from .summarize_chat import summarize_conversation
from .soap_notes_generator import get_soap_notes


def extract_patient_messages(conversation):
    return [
        msg.message
        for msg in conversation
        if msg.role == "patient"
    ]



def generate_clinical_report(conversation, image_path):

    if not conversation:
        raise ValueError("Conversation cannot be empty")

    patient_chats = extract_patient_messages(conversation)

    combined_text = " ".join(patient_chats)

    summary = summarize_conversation(combined_text)

    image_result = run_test(image_path)

    soap_notes = get_soap_notes(
        summary=summary,
        diagnosis=image_result["prediction"]
    )

    return {
        "summary": summary,
        "prediction": image_result["prediction"],
        "confidence": float(image_result["confidence"]),
        "xai_image": image_result["xai_image"],
        "soap_notes": soap_notes,
        "conversation": [
            {"role": msg.role, "message": msg.message}
            for msg in conversation
        ]
    }
