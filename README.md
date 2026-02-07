# Medilink – Clinical Decision Support System

## Overview

Medilink is an AI-assisted clinical decision support system designed to help doctors analyze patient information and medical images.  
The system collects patient details through a chat-based intake process, accepts medical images, and generates a clinical report including diagnosis support, confidence score, SOAP notes, and explainable AI (XAI) visualizations.

This project is created for educational and demonstration purposes only.  
It does **not replace doctors** and should only be used as a support tool.

---

## Features

- Chat-based patient intake  
- Guided clinical questioning  
- Medical image upload  
- AI-assisted diagnosis prediction  
- Confidence score for predictions  
- SOAP notes generation  
- Explainable AI (XAI) heatmap visualization  
- Demo mode with sample patient data  
- Clean, hospital-style user interface  

---


---

## How the System Works

1. The doctor starts a patient intake chat  
2. The patient answers predefined medical questions  
3. A medical image is uploaded  
4. The frontend sends the conversation and image path to the backend API  
5. The backend:
   - Summarizes the patient conversation  
   - Analyzes the medical image  
   - Predicts possible medical conditions  
   - Generates SOAP notes  
6. The frontend displays:
   - SOAP notes on the left side  
   - Diagnosis, confidence score, and XAI image on the right side  

---

## Demo Mode

The application includes a **Run Demo Case** option.

Demo mode automatically loads:
- A sample patient conversation  
- A predefined medical image  

This is useful for:
- Testing the application  
- Academic demonstrations  
- Presentations  

---

## Technologies Used

### Frontend
- Python  
- Streamlit  
- HTML and CSS (custom UI styling)

### Backend
- Python  
- FastAPI  
- Pydantic  
- Machine Learning models  
- Explainable AI (XAI)

---


## Ethical Considerations

Ethics is a critical part of this project, especially because it deals with healthcare data and medical decision support.

### 1 .Human Oversight

- The AI system does not make final medical decisions

- Doctors always remain in control of diagnosis and treatment

- AI outputs are only suggestions and support tools

### 2. Patient Privacy and Data Protection

- Patient data should be anonymized before use

- No personally identifiable information is stored permanently

- Medical images and conversations must be handled securely

### 3. Transparency and Explainability

- The system provides confidence scores with predictions

- Explainable AI (XAI) heatmaps help doctors understand model behavior

- Clinical reports (SOAP notes) are generated in readable format

### 4. Fairness and Bias Reduction

- Training data should represent diverse patient groups

- The model must be tested for bias across age, gender, and skin types

- Continuous monitoring is required to avoid unfair predictions

### 5. Accountability

- Hospital management and system owners are accountable for AI usage

- Developers are responsible for model quality and system reliability

- Clear responsibility is defined for errors or system failures

### 6. Safety and Reliability

- The system should be tested thoroughly before deployment

- Performance must be continuously monitored after deployment

- Any unexpected behavior must be addressed immediately

### 7. Responsible Use and Limitations

- The system is not a certified medical device

- It should not be used for real-world diagnosis without proper validation

- Results should always be verified by qualified medical professionals

# Disclaimer

This project is developed strictly for educational and demonstration purposes.
It is not a certified medical system and should not be used for real-world diagnosis, treatment, or patient care.

---

## How to Run the Project

### 1. Start the Backend

```bash
uvicorn app.main:app --reload
```
### 2. Run the frontend 
```bash
streamlit run app.py
```


