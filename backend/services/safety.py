# backend/services/safety.py
import re

BANNED_PATTERNS = [
    r"you (?:have|are suffering from|are diagnosed with)",
    r"your diagnosis is",
    r"you (?:need|must|should) take",
    r"i (?:diagnose|recommend you take)",
]

DISCLAIMER = (
    "\n\n---\n⚕️ **Disclaimer:** This is an informational explanation only. "
    "It is NOT a diagnosis and NOT medical advice. "
    "Always consult a qualified physician about your results."
)

def sanitize(text: str) -> str:
    for pattern in BANNED_PATTERNS:
        text = re.sub(pattern, "[redacted — consult your doctor]", text, flags=re.IGNORECASE)
    return text + DISCLAIMER