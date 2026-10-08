# Local web demo

From the repository root, run:

```sh
.venv/bin/python -m arabic_poc.web
```

Open **http://127.0.0.1:8765**. Add `--port 8766` for another port. The server binds to localhost and uses the configured model endpoints in `.env`. A local Ollama/Qwen server is needed only when that backend is selected.

## Walkthrough

1. Explore the Library; filter by images, videos, audio or documents.
2. Search `أرني صور الملك فيصل` or `Show me photos of King Faisal`.
3. Inspect descriptions, entities and person provenance below the media.
4. Search `أرني فيديو الملك فيصل في بريطانيا` with the video filter.
5. Use matching-moment buttons or answer citations to seek to a timestamp. “About this item” is file-level publisher metadata, not a scene timestamp.
6. Compare `أرني القبة الخضراء` and `Show me the green dome` to demonstrate query expansion.

The UI requests the `library` dataset. Other historical dataset IDs remain in the API for compatibility. `arabic_poc/active_indexes.json` selects local storage; this workspace uses `rag_storage_arabic_remote_v2`. The top-level Library count reflects unique Library files, not the sum of overlapping API collections.

Search retrieves up to 20 candidate files and reranks to at most 5 source cards. The web process caches BGE-M3 and serializes questions. Reloading the browser does not cancel a running request. No upload/ingestion screen is implemented.

New catalog content appears after a page refresh. Restart the server after changing aliases, prompts, backend/model configuration or the active-index mapping. See [README.md](README.md) for description/indexing commands.

## What the evidence means

Descriptions are generated interpretations; OCR and transcripts are extracted text. Face-gallery matches are separate from names on posters and publisher labels. A video label does not establish that a person appears in every frame. Frames are sampled and may miss short events. King Faisal's newly added videos have scene descriptions, not newly generated transcripts.

Answers validate citation IDs and exact quote substrings. That check does not guarantee factual correctness or semantic support for every claim. When generation fails, the app retains retrieved sources for inspection.

## HTTP checks

```sh
.venv/bin/python -m unittest arabic_poc.web_checks arabic_poc.visual_checks -v
```

The web tests need the local demo files and localhost socket permission. They check catalog access, private-path rejection, byte-range playback, origin/host checks and serialized jobs. Retrieval routing is mocked and should not download a model.
