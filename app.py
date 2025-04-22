import os
import sys
from typing import Dict, Any

import pandas as pd
import gradio as gr

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from models import load_models
from utils import find_entities
from train_module import train_from_csv


MODEL_DIR = 'model'

# Load trained models if available; otherwise train from CSV
try:
    vectorizer, classifier = load_models(MODEL_DIR)
except FileNotFoundError:
    print("Model not found. Training from CSV...")
    vectorizer, classifier, _ = train_from_csv("combined_emails_with_natural_pii.csv")


def process_email(email_text: str) -> Dict[str, Any]:
    """
    Detect entities in the email text, mask PII, and classify the email category.

    Args:
        email_text (str): Raw email content.

    Returns:
        Dict[str, Any]: Dictionary with input, masked entities, masked email,
                        and classification.
    """
    try:
        # Detect entities and mask the email
        detected_entities_raw, masked_email = find_entities(email_text)

        # Convert entity format if necessary
        if isinstance(detected_entities_raw, dict):
            detected_entities_raw = list(detected_entities_raw.values())

        # Format entities for output
        formatted_entities = []
        for entity in detected_entities_raw:
            pos = entity.get("position", [0, 0])
            formatted_entities.append({
                "position": [int(pos[0]), int(pos[1])],
                "classification": entity.get("classification", ""),
                "entity": entity.get("entity", "")
            })

        # Classify the masked email
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
    """
    Wrapper function for Gradio to process email input.

    Args:
        email_text (str): Email content entered by user.

    Returns:
        Dict[str, Any]: Processed output from process_email().
    """
    return process_email(email_text)


# Define Gradio interface for the app
demo = gr.Interface(
    fn=classify_email_ui,
    inputs=gr.Textbox(lines=10, placeholder="Enter email text here..."),
    outputs=gr.JSON(),
    title="Email Classification System",
    description="Classify support emails and mask personal information",
    examples=[
        ["Hello, My name is John Doe and my email is johndoe@example.com. "
         "I'm having issues with my billing."],
        ["I need technical support for my account. "
         "My phone number is +91-9876543210."],
        ["Please update my account. My date of birth is 01/15/1990 and "
         "card number is 1234567812345678."]
    ]
)

if __name__ == "__main__":
    demo.launch()
