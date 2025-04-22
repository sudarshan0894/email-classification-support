import os
import sys
import pandas as pd
import gradio as gr
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import load_models
from utils import find_entities
from train_module import train_from_csv

MODEL_DIR = 'model'
try:
    vectorizer, classifier = load_models(MODEL_DIR)
except FileNotFoundError:
    print("Model not found. Training from CSV...")
    vectorizer, classifier, _ = train_from_csv("combined_emails_with_natural_pii.csv")

def process_email(email_text: str) -> Dict[str, Any]:
    try:
        # Detect entities and mask email
        detected_entities_raw, masked_email = find_entities(email_text)

        # Ensure entities are in the expected list format
        if isinstance(detected_entities_raw, dict):
            detected_entities_raw = list(detected_entities_raw.values())

        formatted_entities = []
        for entity in detected_entities_raw:
            pos = entity.get("position", [0, 0])
            formatted_entities.append({
                "position": [int(pos[0]), int(pos[1])],
                "classification": entity.get("classification", ""),
                "entity": entity.get("entity", "")
            })

        # Classify email
        X_input = vectorizer.transform([masked_email])
        category = classifier.predict(X_input)[0]

        return {
            "input_email_body": email_text,
            "list_of_masked_entities": formatted_entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }

    except Exception as e:
        return {"error": str(e)}

def classify_email_ui(email_text: str) -> Dict[str, Any]:
    return process_email(email_text)

# Define Gradio interface
demo = gr.Interface(
    fn=classify_email_ui,
    inputs=gr.Textbox(lines=10, placeholder="Enter email text here..."),
    outputs=gr.JSON(),
    title="Email Classification System",
    description="Classify support emails and mask personal information",
    examples=[
        ["Hello, my name is John Doe and my email is johndoe@example.com. I'm having issues with my billing."],
        ["I need technical support for my account. My phone number is +91-9876543210."],
        ["Please update my account. My date of birth is 01/15/1990 and card number is 1234567812345678."]
    ]
)

if __name__ == "__main__":
    demo.launch()
