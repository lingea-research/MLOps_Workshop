from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel


app = FastAPI()


class SourceText(BaseModel):
    full_text: str


class DetectedLanguage(BaseModel):
    language: str
    probability: float


@app.get("/language-detection")
async def language_detection(source: SourceText) -> DetectedLanguage:
    detected_language = DetectedLanguage()
    return detected_language


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=4987)
