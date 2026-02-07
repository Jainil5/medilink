from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma3:1b", temperature=0)

def summarize_conversation(conversation_text):
    conversation = [
        {"role": "system", "content": f"""
        
            You are a medical clinical documentation assistant.

            Your task is to analyze a doctor-patient conversation and generate a concise clinical case summary using professional medical language.

            Follow these rules:
            - Extract key symptoms, duration, severity, and triggers
            - Include relevant medical history and medications
            - Include physical examination findings and vitals if mentioned
            - Only include terms mentioned in conversation. No assumptions.
            - Do NOT include casual dialogue
            - Do NOT invent information
            - Write in clear clinical style

            ---

            EXAMPLE 1

            Conversation:
            [doctor]: What brings you in today?
            [patient]: I’ve had a dark spot on my arm for about 3 months.
            [doctor]: Any pain or bleeding?
            [patient]: Sometimes it itches and bleeds a little.
            [doctor]: Any medical conditions?
            [patient]: No.

            Clinical Summary:
            Patient presents with a dark pigmented lesion on the arm present for approximately 3 months, associated with occasional itching and minor bleeding. No significant past medical history reported.

            ---

            NOW GENERATE A CLINICAL SUMMARY FOR THE FOLLOWING CONVERSATION:

        """},
        {"role": "user", "content": f"""
        Conversation:
        {conversation_text}
        Clinical Summary:
        """}
    ]

    response = llm.invoke(conversation)
    return response.content


inp = """EXAMPLE 2

            Conversation:
            doctor: What is your age and gender?
            patient: I am 25 years old and male.
            doctor: How long has the patch been there?
            patient: Almost a year.
            doctor: Any pain?
            patient: No pain or bleeding.
            doctor: It appears rough and light brown.

"""


print(summarize_conversation(inp))