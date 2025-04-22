from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from utils import find_entities
from models import load_models
from train_module import train_from_csv
import os

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "*",  # WARNING: This allows all origins. Use with caution in production.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_DIR = "model"
try:
    vectorizer, classifier = load_models(MODEL_DIR)
except FileNotFoundError:
    print("Model not found. Training from CSV...")
    vectorizer, classifier, _ = train_from_csv("combined_emails_with_natural_pii.csv")


class EmailRequest(BaseModel):
    email_text: str


@app.post("/classify")
async def classify_email(data: EmailRequest) -> Dict[str, Any]:
    try:
        detected_entities, masked_email = find_entities(data.email_text)
        X_input = vectorizer.transform([masked_email])
        category = classifier.predict(X_input)[0]

        return {
            "input_email_body": data.email_text,
            "list_of_masked_entities": detected_entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }
    except Exception as e:
        return {"error": str(e)}