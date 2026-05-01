import sys
from pathlib import Path

# ================= PATH OPTIMIZATION =================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = BASE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from concurrent.futures import ThreadPoolExecutor
from services.ham_test import run_test
from services.summarize_chat import summarize_conversation
from services.soap_notes_generator import get_soap_notes


def generate_clinical_report(conversation, image_path):
    # Extract items regardless if it's a dict (native testing) or list (from FastAPI endpoint)
    items = conversation.get("conversation", []) if isinstance(conversation, dict) else conversation
    
    chats = []
    for i in items:
        # Extract role and message handling both dicts and Pydantic Message models
        role = i["role"] if isinstance(i, dict) else getattr(i, "role", "")
        message = i["message"] if isinstance(i, dict) else getattr(i, "message", "")
        
        if role == "patient":
            chats.append(message)
    
    with ThreadPoolExecutor() as executor:
        # Run independent tasks in parallel
        summary_future = executor.submit(summarize_conversation, str(chats))
        test_future = executor.submit(run_test, image_path)
        
        # Wait for results
        summary = summary_future.result()
        test_result = test_future.result()
        
    disease = test_result["prediction"]
    confidence = test_result["confidence"]
    xai_image = test_result.get("xai_image", "No XAI image available currently")
    
    # SOAP notes depend on summary and disease
    soap_notes = get_soap_notes(summary, disease)
    
    return {
        "summary": summary,
        "prediction": disease,
        "confidence": confidence,
        "xai_image": xai_image,
        "soap_notes": soap_notes,
        "conversation": conversation
    }



if __name__ == "__main__":
    chat = {
      "conversation": [
        {
          "role": "doctor",
          "message": "What is your age and gender?"
        },
        {
          "role": "patient",
          "message": "I am 25 years old and male."
        },
        {
          "role": "doctor",
          "message": "How long has the patch been there?"
        },
        {
          "role": "patient",
          "message": "Almost a year."
        },
        {
          "role": "doctor",
          "message": "Any pain?"
        },
        {
          "role": "patient",
          "message": "No pain or bleeding."
        },
        {
          "role": "doctor",
          "message": "It appears rough and light brown."
        }
      ]
    }

    image_path = "/Users/jainil/Documents/development/medilink/datasets/ddidiversedermatologyimages/images/000001.png"

    print(generate_clinical_report(chat, image_path))


# Output:
# {'summary': 'Patient is a 25-year-old male presenting with a lesion on the arm approximately one year prior to current visit. The patient reports no pain or bleeding.', 'prediction': 'Melanocytic Nevus', 'confidence': 0.9753084182739258, 'xai_image': 'backend/app/test/merged_result.png', 'soap_notes': 'Okay, here are the SOAP notes based on the provided information:\n\n---\n\n**S (Subjective):**\n\nPatient is a 25-year-old male presenting for a follow-up appointment. He reports a lesion on the arm approximately one year prior to the current visit. He denies any pain or bleeding.\n\n**O (Objective):**\n\n*   Patient states the lesion is located on the arm and has been present for approximately one year.\n*   Examination reveals a pigmented lesion approximately 1.5 cm in diameter.\n*   AI analysis indicates a Melanocytic Nevus (Melanocytic Nevus) with a high degree of confidence (95%).\n\n**A (Assessment):**\n\n*   Possible Melanocytic Nevus – Elevated suspicion for malignancy based on lesion characteristics and patient history.\n\n**P (Plan):**\n\n*   Recommend a full dermatological examination, including detailed skin imaging and biopsy evaluation if warranted.\n\n---\n\n**Important Note:** This is a preliminary assessment based on the limited information provided. A definitive diagnosis requires further investigation and potentially biopsy.', 'conversation': {'conversation': [{'role': 'doctor', 'message': 'What is your age and gender?'}, {'role': 'patient', 'message': 'I am 25 years old and male.'}, {'role': 'doctor', 'message': 'How long has the patch been there?'}, {'role': 'patient', 'message': 'Almost a year.'}, {'role': 'doctor', 'message': 'Any pain?'}, {'role': 'patient', 'message': 'No pain or bleeding.'}, {'role': 'doctor', 'message': 'It appears rough and light brown.'}]}}

