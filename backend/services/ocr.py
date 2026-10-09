# backend/services/ocr.py
import pytesseract
from PIL import Image
import io

def extract_text_from_image(image_bytes: bytes, lang: str = "eng+rus") -> str:
    """Local OCR via Tesseract. No API call, no cost."""
    image = Image.open(io.BytesIO(image_bytes))

    # Pre-processing: grayscale + contrast boost for better accuracy
    image = image.convert("L")  # grayscale

    text = pytesseract.image_to_string(image, lang=lang)
    return text.strip()