# backend/services/ocr_pdf.py  (for the final defense PDF support)
from pdf2image import convert_from_bytes
from backend.services.ocr import extract_text_from_image

def extract_text_from_pdf(pdf_bytes: bytes, lang: str = "eng+rus") -> str:
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    texts = []
    for page in pages:
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        texts.append(extract_text_from_image(buf.getvalue(), lang))
    return "\n---PAGE BREAK---\n".join(texts)