# Glossary generation and RAG roadmap

## Current pipeline

The app processes an uploaded document through OCR, LLM extraction, glossary retrieval, and LLM explanation. The glossary currently contains about 103 entries derived from Wikipedia. A basic safety filter adds a disclaimer, but it does not guarantee factual correctness or prevent every unsafe answer.

The glossary generator is a separate, explicitly invoked preparation step. It writes `data/raw/llm_generated/llm_glossary.json`; a separate build step merges available glossary sources for the app's index. Generation requires OpenRouter credentials and supports a separate `GLOSSARY_MODEL` setting. It makes external inference requests when run. It is not part of application startup or offline tests. The generator's output is an educational draft: the current term-only prompt does not provide source material, so it cannot guarantee citations or source-grounded claims. The app is not medical advice.

## Configuration contract

- Generator-only optional file: `.env.glossary.local` at the repository root. Only the generator loads it; the app's `.env` and settings stay independent.
- Process environment variables take precedence over values in that file.
- `OPENROUTER_API_KEY` is required. `OPENROUTER_BASE_URL` defaults to `https://openrouter.ai/api/v1`.
- `GLOSSARY_MODEL` overrides the original free default, `mistralai/mistral-7b-instruct:free`, independently of the app's `MODEL_EXPLANATION`. The original free fallback list and order are preserved. Each of up to three passes tries the primary model followed by those fallbacks. These are historical model IDs; their current availability has not been verified.
- Missing credentials or model configuration should produce a clear error before any request is made.
- Generated output is kept separate at `data/raw/llm_generated/llm_glossary.json`; generation, merge, and reindex are explicit manual operations.
- If generation produces no entries, or the merge has no usable records, it fails without replacing the existing output. Review generated drafts before merging: the existing LLM → Wikipedia → Wikidata precedence is preserved and does not establish source quality.
- The Wikipedia/Wikidata collectors likewise exit with code `1` without creating or overwriting output files when no records are collected. Successful and partially successful runs retain the existing save behavior.

## Next steps

Source URLs are now optionally retained by the Wikipedia/Wikidata collectors, the glossary merge, and Chroma metadata during future explicit runs. Legacy records without URLs remain supported. This does not backfill the checked-in corpus, rebuild the index, change retrieval or prompts, or add citations to the UI. URLs emitted by the term-only LLM generator are not treated as verified sources or inherited from lower-priority duplicate records.

1. **Improve source grounding.** Build a curated corpus with source URL, title, date, and content. Have the generator transform provided source material, retain its provenance, and keep generated drafts separate from source records.
2. **Make retrieval auditable.** Preserve source metadata through chunking and indexing so retrieved context can be inspected and attributed. Define how the app presents source information before claiming that explanations are citation-backed.
3. **Evaluate retrieval and explanations.** Create a small reviewed set covering abbreviations, common terms, unknown terms, and missing context. Check retrieval relevance, unsupported statements, and safe handling of uncertainty.
4. **Validate the full runtime path.** Separately exercise real OCR and model calls with representative documents, alongside the existing offline tests. Record limitations and review results before making quality claims.

Corpus preparation and retrieval tests can begin without Ollama. Real vector search needs Chroma and an embedding model; generation and a full upload-to-answer check also need a working LLM endpoint.
