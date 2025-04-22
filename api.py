from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from utils import find_entities
from models import load_models
from train_module import train_from_csv
import os

# 🚀 Initialize FastAPI app
app = FastAPI(title="Email Classifier API", version="1.0")

# 🔐 CORS settings
origins = [
    "http://localhost",
    "http://localhost:8080",
    "*"  # WARNING: Use specific domains in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Load or train model
MODEL_DIR = "model"
CSV_PATH = "combined_emails_with_natural_pii.csv"

try:
    vectorizer, classifier = load_models(MODEL_DIR)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("⚠️ Model not found. Training from CSV...")
    vectorizer, classifier = train_from_csv(CSV_PATH)


# ✅ Health Check Root Route
@app.get("/")
def root():
    return {"status": "Email Classifier API is running 🚀"}


# 📬 Request model
class EmailRequest(BaseModel):
    email_text: str


# 🔍 Email classification endpoint
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
        return {"error": f"Something went wrong: {str(e)}"}
