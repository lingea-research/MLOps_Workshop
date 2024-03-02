from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import fasttext


app = FastAPI()
model = fasttext.load_model('models/normalized_model4.bin')


class SourceText(BaseModel):
    full_text: str


class DetectedLanguage(BaseModel):
    language: str
    probability: float


@app.post("/language-detection")
async def language_detection(source: SourceText) -> DetectedLanguage:
    prediction = model.predict(source.full_text)
    detected_language = DetectedLanguage(
        language=prediction[0][0].strip('__label__'),
        probability=prediction[1][0]
    )
    return detected_language


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=4987)
