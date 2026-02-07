from .ham_test import run_test
from .summarize_chat import summarize_conversation
from .soap_notes_generator import get_soap_notes





def generate_clinical_report(conversation, image_path):
    chats = []
    for i in conversation["conversation"]:
        if i["role"] == "patient":
            chats.append(i["message"])
    # print(chats)        
    summary = summarize_conversation(str(chats))
    image_result = run_test(image_path)
    soap_notes = get_soap_notes(summary, image_result["prediction"])
    return {
        "summary": summary,
        "prediction": image_result["prediction"],
        "confidence": image_result["confidence"],
        "xai_image": image_result["xai_image"],
        "soap_notes": soap_notes,
        "conversation": conversation
    }


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

# print(generate_clinical_report(chat, image_path))


# Output:
# {'summary': 'Patient is a 25-year-old male presenting with a lesion on the arm approximately one year prior to current visit. The patient reports no pain or bleeding.', 'prediction': 'Melanocytic Nevus', 'confidence': 0.9753084182739258, 'xai_image': 'backend/app/test/merged_result.png', 'soap_notes': 'Okay, here are the SOAP notes based on the provided information:\n\n---\n\n**S (Subjective):**\n\nPatient is a 25-year-old male presenting for a follow-up appointment. He reports a lesion on the arm approximately one year prior to the current visit. He denies any pain or bleeding.\n\n**O (Objective):**\n\n*   Patient states the lesion is located on the arm and has been present for approximately one year.\n*   Examination reveals a pigmented lesion approximately 1.5 cm in diameter.\n*   AI analysis indicates a Melanocytic Nevus (Melanocytic Nevus) with a high degree of confidence (95%).\n\n**A (Assessment):**\n\n*   Possible Melanocytic Nevus – Elevated suspicion for malignancy based on lesion characteristics and patient history.\n\n**P (Plan):**\n\n*   Recommend a full dermatological examination, including detailed skin imaging and biopsy evaluation if warranted.\n\n---\n\n**Important Note:** This is a preliminary assessment based on the limited information provided. A definitive diagnosis requires further investigation and potentially biopsy.', 'conversation': {'conversation': [{'role': 'doctor', 'message': 'What is your age and gender?'}, {'role': 'patient', 'message': 'I am 25 years old and male.'}, {'role': 'doctor', 'message': 'How long has the patch been there?'}, {'role': 'patient', 'message': 'Almost a year.'}, {'role': 'doctor', 'message': 'Any pain?'}, {'role': 'patient', 'message': 'No pain or bleeding.'}, {'role': 'doctor', 'message': 'It appears rough and light brown.'}]}}