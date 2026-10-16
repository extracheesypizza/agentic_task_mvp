# MedTranslate

Translates medical jargon into plain Russian. You upload a photo of a lab result or prescription, and the app explains what everything means in normal language.

Made as a group project for the MAS course, 2026/27. Google Presentation link: [click](https://docs.google.com/presentation/d/1fO4Bg-KvhK54_chnVvfkShuPnoOIKov-wPjIMQHEg1U/edit?usp=sharing)

> This is not medical advice. Always talk to your doctor.

---

## What it does

1. You upload a photo of a medical document (lab results, prescription, etc.)
2. OCR reads the text from the image
3. An LLM extracts the structured data (parameters, values, units)
4. RAG pulls relevant definitions from our medical glossary
5. A second LLM call writes a plain-language explanation in Russian
6. A safety filter makes sure nothing looks like a diagnosis

---

## Why LLM

Medical documents are messy. Different clinics use different formats, abbreviations, layouts. You can't write a regex parser for all of that. The LLM handles the variability of real-world medical text, and RAG keeps it grounded so it doesn't hallucinate reference ranges.

We tried OpenRouter free models first but kept getting 403 errors because cloud providers block medical content. So we moved to a local model via Ollama, which solved the problem and also costs nothing.

---

## The model

We use **Google MedGemma:4B** running locally through **Ollama**.

It does two things:

- **Extraction**: takes OCR text and outputs structured JSON (parameter, value, unit, reference range, status)
- **Explanation**: takes the JSON + RAG context and writes a simple explanation in Russian

We chose a local model because:

- Free cloud APIs kept blocking medical content [403 errors]
- No rate limits, no API keys needed
- Anyone who clones the repo gets the same result

Embeddings for [future] RAG [will] use `sentence-transformers/all-MiniLM-L6-v2` [runs locally too].

---

## How to run

### With Docker

```Shell
git clone https://github.com//med-translator.git
cd med-translator
cp .env.example .env

# first time only: start ollama and pull the model (~4.7 GB)
docker compose up -d
docker exec -it medtranslate-ollama ollama pull qwen2.5:7b

# then run everything
docker compose up
```

- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

### GPU (optional, faster)

```Shell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

### Without Docker (for development)

```Shell
pip install -r backend/requirements.txt
ollama pull qwen2.5:7b
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/app.py
```

---

## Project structure

```
    ├── backend/
    │   ├── main.py              # FastAPI app
    │   ├── config.py            # settings from .env
    │   ├── routers/             # API endpoints
    │   ├── services/            # OCR, extraction, explanation, safety
    │   ├── prompts/             # prompt templates
    │   └── rag/                 # glossary, index building, retrieval
    ├── frontend/
    │   ├── app.py               # Streamlit UI
    │   └── components/          # UI components
    ├── scripts/                 # scraping and data pipeline scripts
    ├── data/                    # scraped data, sample images
    ├── notebooks/               # experiments
    ├── Dockerfile
    ├── docker-compose.yml
    └── .env.example
```

---

## Team

| Name                    | GitHub               |
| ----------------------- | -------------------- |
| *Ryazanov Stanislaw*  | @flowwwie            |
| *Gusev Maksim*        | @maxigacy            |
| *Privezentsev Daniil* | @Daniil_Privezentsev |
| *Bozhik Maksim*       | @bozzhik             |

---

## Known issues

- OCR doesn't work super well with handwritten documents
- Only Russian for now

---

## Environment variables

See `.env.example`. The main ones:

- `OLLAMA_BASE_URL` — where Ollama is running (default `http://localhost:11434/v1`)
- `MODEL_EXTRACTION` / `MODEL_EXPLANATION` — which model to use (default `qwen2.5:7b`)
- `TESSERACT_LANG` — OCR languages (default `eng+rus`)
- `EMBEDDING_MODEL` — embedding model for RAG
