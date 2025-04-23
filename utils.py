import re
from typing import List, Tuple, Dict, Any


def find_entities(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Detect and mask sensitive entities in the input text.

    :param text: Input string that may contain sensitive information.
    :return: A tuple with a list of detected entities and a version of the input text with masked entities.
    """
    patterns = {
        # Date of Birth (matches formats like 'Jan 15, 2021' or '01/15/2021')
        'dob': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b',
        
        # Credit/Debit Card Number (matches patterns like '1234-5678-1234-5678')
        'credit_debit_no': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        
        # Aadhar Number (matches Indian Aadhar format like '1234-5678-9012')
        'aadhar_num': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        
        # CVV Number (matches a 3-digit number that could be associated with CVV, considering the format)
        'cvv_no': r'(?i)(cvv(?: number)?|cvc2)[\s:]*\d{3}',
        
         # Expiry Date (matches formats like '12/23' or '01/22')
        'expiry_no': r'\b(0[1-9]|1[0-2])/\d{2}\b',
        
        # Email Address (matches email addresses like user@example.com)
        'email': r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
        
        # Full Name (matches patterns like 'John Doe' or 'My name is John Doe')
        'full_name': r'(?:(?:My name is|My Name:|Full Name|I am|This is|Sincerely,|Best regards,|Regards,|Hello,\s?|Hi,\s?|Dear,\s?|My Name:\s*|Full Name[:\s]*)\s*)([A-ZÀ-Ý][a-zà-ÿ]+(?:\s[A-ZÀ-Ý][a-zà-ÿ]+)+)(?=[\s.,\n]|$)'
    }
    
    detected_entities: List[Dict[str, Any]] = []
    masked_text = text
    offset = 0

    # Special handling for phone numbers using digit length checks
    phone_pattern = r'\+(\d{1,4})[-\s]?\(?(\d{1,4})\)?[-\s]?(\d{1,4})[-\s]?(\d{1,4})'

    for match in re.finditer(phone_pattern, text):
        full_number = match.group(0)
        total_digits = sum(len(g) for g in match.groups() if g)

        if total_digits >= 11:
            start, end = match.start(), match.end()
            adjusted_start = start + offset
            adjusted_end = end + offset

            mask_token = '[phone_number]'
            masked_text = (
                masked_text[:adjusted_start] + mask_token + masked_text[adjusted_end:]
            )
            offset += len(mask_token) - (end - start)

            detected_entities.append({
                "position": [adjusted_start, adjusted_start + len(mask_token)],
                "classification": "phone_number",
                "entity": full_number
            })

    # Loop through remaining patterns
    for entity_type, pattern in patterns.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                if entity_type == "full_name":
                    value = match.group(1).strip()
                    if len(value.split()) > 4 or "submitting" in value.lower():
                        continue
                    start, end = match.start(1), match.end(1)
                else:
                    value = match.group()
                    start, end = match.start(), match.end()

                adjusted_start = start + offset
                adjusted_end = end + offset

                mask_token = f"[{entity_type}]"
                masked_text = (
                    masked_text[:adjusted_start] + mask_token + masked_text[adjusted_end:]
                )
                offset += len(mask_token) - (end - start)

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
    Prepare a dictionary output containing original and masked text along with detected entities.

    :param email_body: Raw email content.
    :param category: Category or label for the email.
    :return: Dictionary with input, masked version, entities, and category.
    """
    entities, masked_email = find_entities(email_body)

    return {
        "input_email_body": email_body,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
