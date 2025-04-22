import re
from typing import List, Tuple, Dict, Any


def find_entities(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Detect personally identifiable information (PII) in the input text and replace them
    with labeled placeholders like [email], [dob], etc.

    Args:
        text (str): Raw email text to be processed.

    Returns:
        Tuple[List[Dict[str, Any]], str]: A tuple containing:
            - A list of detected entity metadata with position, classification, and value.
            - The masked version of the input text.
    """

    # Define regex patterns for common PII entities
    patterns = {
        'dob': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b'
               r'|\b\d{2}/\d{2}/\d{4}\b',
        'credit_debit_no': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'aadhar_num': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'cvv_no': r'\b\d{3}\b(?=\s*(?:cvv|cvv number)?)',
        'expiry_no': r'\b(?:0[1-9]|1[0-2])/\d{2}\b',
        'phone_number': r'(\+?\d{1,3}[-\s]?\d{4,5}[-\s]?\d{4,5})',
        'email': r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
        'full_name': r'(?:(?:My name is|I am|This is|Sincerely,|Best regards,|Regards,|Hello|Hi|Dear|Full Name[:\s]*)\s*)'
                     r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)(?=\s|,|\n|$)'
    }

    detected_entities: List[Dict[str, Any]] = []
    masked_text = text
    offset = 0  # Adjust for text length changes after replacements

    for entity_type, pattern in patterns.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                # Extract the matched value and its position
                if entity_type == "full_name":
                    value = match.group(1).strip()
                    if len(value.split()) > 4 or "submitting" in value.lower():
                        continue
                    start, end = match.start(1), match.end(1)
                elif entity_type in ["cvv_no", "phone_number"] and match.groups():
                    value = match.group(1)
                    start, end = match.start(1), match.end(1)
                else:
                    value = match.group()
                    start, end = match.start(), match.end()

                # Adjust indices due to previous replacements
                adjusted_start = start + offset
                adjusted_end = end + offset

                # Replace the detected value with a placeholder
                mask_token = f"[{entity_type}]"
                masked_text = (
                    masked_text[:adjusted_start] + mask_token + masked_text[adjusted_end:]
                )

                # Update offset due to masking
                offset += len(mask_token) - (end - start)

                # Store metadata for this entity
                detected_entities.append({
                    "position": [adjusted_start, adjusted_start + len(mask_token)],
                    "classification": entity_type,
                    "entity": value
                })

            except Exception as e:
                print(f"Error processing {entity_type}: {e}")

    return detected_entities, masked_text


def prepare_output(email_body: str, category: str) -> Dict[str, Any]:
    """
    Prepares structured output by detecting and masking PII in email text,
    and attaching its classified category.

    Args:
        email_body (str): Original email text.
        category (str): Predicted email category.

    Returns:
        Dict[str, Any]: Structured response with email body, detected entities,
                        masked version of the email, and category.
    """
    entities, masked_email = find_entities(email_body)
    return {
        "input_email_body": email_body,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }