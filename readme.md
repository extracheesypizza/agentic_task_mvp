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
6. A basic safety filter redacts some English phrases and appends a disclaimer; it does not guarantee that diagnoses or treatment advice are absent

---

## Why LLM

Medical documents are messy. Different clinics use different formats, abbreviations, layouts. You can't write a regex parser for all of that. The LLM handles the variability of real-world medical text, and RAG supplies glossary context for the explanation. Reference ranges should come from the uploaded document; RAG does not guarantee factual correctness.

We tried OpenRouter free models first but kept getting 403 errors. So we moved to a local model via Ollama, avoiding dependence on an external inference API.

---

## The model

The current configuration uses **MedGemma 4B** (`medgemma:4b`) running locally through **Ollama**.

It does two things:

- **Extraction**: takes OCR text and outputs structured JSON (parameter, value, unit, reference range, status)
- **Explanation**: takes the JSON + RAG context and writes a simple explanation in Russian

We chose a local model because:

- Earlier experiments with free cloud APIs returned 403 errors
- No external inference API key or per-request API charge is needed for local Ollama
- The model and configuration can be shared across the team; generated answers can still vary

Embeddings for RAG use `sentence-transformers/all-MiniLM-L6-v2` (runs locally too).

---

## How to run

### With Docker

Requirements: Docker with Compose, and enough memory and disk for the Ollama model and the Python/embedding image. The standard setup runs Ollama in a container and downloads `medgemma:4b` automatically on its first startup (several GB; allow time for the download and initial index build).

```sh
git clone https://github.com/extracheesypizza/agentic_task_mvp.git
cd agentic_task_mvp
# Keep an existing .env; review its values against the example.
cp -n .env.example .env
docker compose up --build
```

Compose starts Ollama, waits for it to become healthy, pulls `medgemma:4b`, then starts the API and Streamlit UI. The first build downloads the embedding model too. Open:

- UI: <http://localhost:8501>
- API documentation: <http://localhost:8000/docs>
- API health: <http://localhost:8000/health>
- Dependency readiness: <http://localhost:8000/health/ready> (inspect the JSON status; `degraded` still has HTTP 200)

Stop with `Ctrl+C`; use `docker compose down` to stop and remove the containers. The named `ollama_models` volume retains the model between runs. `docker compose down -v` also removes that model volume.

### GPU (optional, faster)

The optional `docker-compose.gpu.yml` adds an NVIDIA GPU reservation for Linux hosts with Docker's NVIDIA support configured:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

On macOS, the standard Docker setup runs the Ollama container without that NVIDIA reservation. For native Ollama execution, see the development setup below; this can avoid running Ollama inside a container, though Docker remains an option for the API and UI.

### Without Docker (for development)

Use Python 3.11. Tesseract and its English/Russian language data are native system dependencies. On macOS with Homebrew:

```sh
brew install tesseract tesseract-lang
```

On Debian/Ubuntu, install `tesseract-ocr`, `tesseract-ocr-eng`, and `tesseract-ocr-rus`. PDF conversion through Poppler is not used by the current upload endpoint.

Create an environment, install the backend's full requirements, configure the app, and pull the configured model into a separately installed/running Ollama instance:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
cp -n .env.example .env
```

If Ollama is not installed, one macOS option is `brew install ollama`. Start the server in another terminal with `ollama serve` (unless the Ollama app/service is already running). Once it is reachable, download the model:

```sh
ollama pull medgemma:4b
```

Run the backend and frontend in separate terminals from the repository root (with the virtual environment activated in each):

```sh
uvicorn backend.main:app --reload --port 8000
```

```sh
streamlit run frontend/app.py --server.port 8501
```

For this native setup, the example `OLLAMA_BASE_URL` points to `http://localhost:11434/v1`. The frontend defaults to the local API at `http://localhost:8000`. If using a custom API URL locally, export `BACKEND_URL` in the frontend terminal; the frontend reads process environment variables.

### Tests without Ollama

The test environment is intentionally separate from the full application requirements, so tests do not install ChromaDB, Sentence Transformers, or PyTorch. From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest tests
```

These tests use substitutes for external services and check application behavior; they do not measure OCR or model answer quality. They do not require Ollama. A real upload-to-answer run requires Ollama with the configured model available, in addition to the native OCR dependency or the Docker environment.

### Read-only corpus check

Run `python -m scripts.check_glossary` from the repository root, or pass a JSON file as the final argument. The command uses the glossary builder's record validation rules and reports sources, duplicate terms, empty/unusable definitions, and HTTP(S) source URL coverage. It does not modify files, make network requests, load models, or require extra dependencies. Missing source labels or URLs produce warnings; malformed or empty input, invalid records, and duplicate terms return exit code `1`. Otherwise it returns `0`. This checks structure and provenance metadata, not medical accuracy or whether a URL is reachable.

### Manual model connection check

With Ollama running and the configured model available, run `python -m backend.test` from the repository root. This explicitly calls the model selected by `MODEL_EXPLANATION` and prints its response. Exit code is `0` for a non-empty response and `1` for a model-service error or empty response. This checks model communication only, not OCR, retrieval, or answer quality. Importing the module and running the offline tests do not make model requests.

### Optional glossary generation

Glossary generation is a manual preparation step using free LLMs via OpenRouter. It is not needed for offline tests. Copy `.env.glossary.example` to `.env.glossary.local` with `cp -n .env.glossary.example .env.glossary.local`, then set `OPENROUTER_API_KEY`. The separate `GLOSSARY_MODEL` overrides the original free primary model; the original free fallback list is preserved. Availability of these historical model IDs has not been verified. The script writes `data/raw/llm_generated/llm_glossary.json`. Run `python scripts/generate_glossary_llm.py` only when you intend to make external requests. Generation does not merge or reindex the glossary. See [the RAG roadmap](docs/rag-roadmap.md) for the current limits and proposed next steps.

---

## Project structure

```text
    ├── backend/
    │   ├── main.py              # FastAPI app
    │   ├── config.py            # settings from .env
    │   ├── schemas.py           # extraction output validation
    │   ├── routers/             # API endpoints
    │   ├── services/            # OCR, extraction, explanation, safety
    │   └── rag/                 # glossary, index building, retrieval
    ├── frontend/
    │   ├── app.py               # Streamlit UI
    │   └── components/          # UI components
    ├── scripts/                 # scraping and data pipeline scripts
    ├── data/                    # project data
    ├── tests/                   # regression tests with service substitutes
    ├── requirements-test.txt
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
- The current upload API accepts PNG/JPG documents; PDF uploads and interpretation of X-ray/CT images are not implemented
- The glossary and safety filter are limited; the app is an educational prototype

---

## Environment variables

See `.env.example` for all settings. The main ones:

- `OLLAMA_BASE_URL` — where Ollama is running (default `http://localhost:11434/v1`)
- `MODEL_EXTRACTION` / `MODEL_EXPLANATION` — which model to use (default `medgemma:4b`)
- `TESSERACT_LANG` — OCR languages (default `eng+rus`)
- `EMBEDDING_MODEL` — embedding model for RAG

Docker Compose overrides `OLLAMA_BASE_URL` for the internal Ollama service and `BACKEND_URL` for the frontend. The default init service pulls `medgemma:4b`; custom model names must be pulled separately.

The client uses Ollama's OpenAI-compatible API with Ollama-specific options. Changing the URL alone does not guarantee support for another provider. `OLLAMA_API_KEY=ollama` is a local placeholder, not an external provider credential.

The repository already tracks `.env`; adding it to `.gitignore` does not untrack that existing file. Keep real credentials out of it. This change does not remove any file from the Git index or history.
