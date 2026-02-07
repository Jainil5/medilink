from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from services.main_service import generate_clinical_report

app = FastAPI(title="Medilink Backend API")


class Message(BaseModel):
    role: str
    message: str


class ReportRequest(BaseModel):
    conversation: List[Message]
    image_path: str


@app.post("/generate-report")
async def generate_report(data: ReportRequest):

    result = generate_clinical_report(
        data.conversation,
        data.image_path
    )

    return result


@app.get("/")
def root():
    return {"status": "Backend running"}
