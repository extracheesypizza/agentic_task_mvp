# backend/services/vision.py
import base64
from openai import OpenAI

client = OpenAI()

def extract_text_from_image(image_bytes: bytes) -> str:
    """Send medical document image to GPT-4o, get raw text back."""
    b64 = base64.b64encode(image_bytes).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical document OCR assistant. "
                    "Extract ALL text from this image exactly as written. "
                    "Preserve numbers, units, reference ranges. "
                    "Output raw text only, no commentary."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Extract all text from this medical document."},
                ],
            },
        ],
        max_tokens=4096,
        temperature=0,
    )
    return response.choices[0].message.content