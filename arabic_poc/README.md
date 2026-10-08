# Arabic and English multimedia search

A local web app for searching images, video moments, speech, OCR and document text. It adds bilingual person aliases, Arabic query expansion, optional reference-face matches, generated descriptions and grounded answers to the RAG-Anything workspace.

## Start the app

Run from the repository root:

```sh
.venv/bin/python -m arabic_poc.web
```

Open **http://127.0.0.1:8765**. Use `--port 8766` for another port. The UI has one Library with image, video, audio and document filters. It serves existing indexed files; ingestion runs separately through the CLI. See [WEB_DEMO.md](WEB_DEMO.md).

## Setup

Python 3.10+ is declared by this project; available wheels for optional OCR/face backends depend on the Python version and platform. Use the existing `.venv` when working in this checkout.

```sh
# For a new checkout/environment only
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -r arabic_poc/requirements.txt
# Optional face enrichment
.venv/bin/python -m pip install -e '.[face]'
# macOS media tools
brew install ffmpeg
```

Merge the required settings from [`env.example`](env.example) into your root `.env`; do not overwrite an existing configuration. Remote text/vision/ASR and local OCR can be selected independently. Embeddings remain local with `BGE_MODEL=BAAI/bge-m3` by default. Local models may download weights on first use.

| Setting | Purpose |
| --- | --- |
| `MODEL_BACKEND` | Default backend, `local` or `remote` |
| `TEXT_BACKEND`, `VISION_BACKEND`, `ASR_BACKEND`, `OCR_BACKEND` | Per-role overrides |
| `REMOTE_BASE_URL`, `REMOTE_API_KEY` | OpenAI-compatible gateway configuration |
| `REMOTE_TEXT_MODEL` | Query rewriting, translation, reranking and answer model |
| `REMOTE_VISION_MODEL` | Image/frame description model; remote OCR when selected |
| `REMOTE_ASR_MODEL` | Remote transcription model |
| `REMOTE_MAX_TOKENS`, `REMOTE_TIMEOUT` | Remote response limit and per-request timeout |
| `BGE_MODEL` | BGE-M3 model ID or compatible local model directory |
| `QWEN_BASE_URL`, `QWEN_MODEL`, `QWEN_API_KEY` | Local text-generation endpoint |
| `FACE_GALLERY_PATH` | Enables optional reference-gallery face enrichment |

The recent demo used `cerebras/gpt-oss-120b` for text and `groq/qwen/qwen3.6-27b` for vision. These are configuration values, not a required or hardcoded pairing. The reranker uses the configured text LLM, not a dedicated cross-encoder.

## How search works

1. Preserve the original query. Resolve known Arabic/English aliases from [`person_aliases.json`](person_aliases.json); the text LLM can propose additional name-span resolutions against a bounded canonical-person shortlist. Ambiguous known aliases remain unresolved.
2. Add canonical-name variants and, for English queries, an Arabic translation when the remote text backend is enabled. Provider failures retain the original query and deterministic alias variants.
3. Embed query variants with BGE-M3. Combine dense similarity, BM25-style token scoring, entity matches and person aliases, then group evidence by source file.
4. Retrieve up to 20 candidate files and rerank with the text LLM. Show at most **5 files**, with up to **3 matching passages/moments per file**. A failed/unavailable reranker falls back to qualified hybrid results.
5. Generate an answer from up to 24 selected evidence passages. Validate quoted substrings and citation IDs; retain source cards if generation fails.

`search_service.py` keeps the embedding model loaded once per web process. Queries are serialized. Index JSON is read at query time; catalogs refresh on page reload. Restart the app after changing prompts, aliases, backend/model settings or the active-index configuration.

Generated descriptions, OCR/visible text, transcript mentions, publisher labels and face-gallery matches have distinct provenance. A publisher label describes the whole file and does not prove a person is visible at every timestamp. Face matches and descriptions are fallible. Exact quotation validation does not establish that every answer claim follows from the evidence.

## Storage and entry points

[`active_indexes.json`](active_indexes.json) is local runtime configuration and is intentionally not committed. In this workspace it points to `rag_storage_arabic_remote_v2`. Without it, older collection defaults in `video_search.py` and `web.py` apply.

| Location | Contents |
| --- | --- |
| `data/` | Sample media and adjacent `*.metadata.json` attribution |
| `reference_faces/<canonical_id>/` | Reference photos and `sources.json`; keep separate from sample photos |
| `<storage>/sources/*.json` | Extracted passages, entities, source paths and anchors |
| `<storage>/index.json` | Combined library records and parallel dense vectors |
| `<storage>/visual-index.json` | Visual description records and vectors |
| `<storage>/index/` | Separate LightRAG graph/vector stores |
| `<storage>/image-additions-backup-*` | Image-index rollback snapshots |
| `<storage>/video-refresh-*/` | Resumable frame results and pre-publish backups |
| `reference_faces/.face_gallery.npz` | Rebuildable reference-face embedding cache |

This is a local JSON-backed POC, not a hosted vector database. Searches load and score the index in memory. Moving to a database/object store is future work. Keep one index writer running per storage directory; the individual JSON writes are atomic, but publishing several index files is not a single transaction.

| Module | Responsibility |
| --- | --- |
| `web.py`, `web_assets/` | Local HTTP API and UI |
| `search_service.py`, `video_search.py` | Cached encoder, hybrid retrieval, grouping and reranking |
| `query_rewriting.py`, `query_translation.py`, `person_aliases.py` | Query expansion and canonical names |
| `remote.py`, `models.py`, `face_identifier.py` | Remote/local model adapters and gallery enrichment |
| `__main__.py` | Extraction, source records and graph CLI |
| `index_images.py`, `index_videos.py` | Incremental description updates for the app |
| `refresh_corpus.py`, `refresh_indexes.py` | Full-corpus refresh and graph/index activation |
| `answers.py`, `prompt_loader.py`, `prompts.yaml` | Grounded answers and application prompts |
| `evaluate*.py`, `*checks.py` | Evaluation and regression checks |

## Add or refresh image descriptions

These commands update the **existing active app indexes**; they require its `index.json` and `visual-index.json`. They do not update the LightRAG graph.

```sh
# New images in this directory only
.venv/bin/python -m arabic_poc.index_images data/demo_additions/king_faisal_images

# Regenerate descriptions/entities for all images in the selected directory
.venv/bin/python -m arabic_poc.index_images data/demo_additions/king_faisal_images --refresh
```

The image command excludes paths containing `visual_samples`. Other generic ingestion commands do not share this exclusion: pass specific sample paths, not the whole `data/` tree. Reference-gallery photos must never double as sample content.

## Add or refresh video scene descriptions

```sh
.venv/bin/python -m arabic_poc.index_videos --frame-seconds 10 \
  data/demo_additions/king_faisal/king_faisal_1967.mp4 \
  data/demo_additions/king_faisal/king_faisal_royal_welcome_1967.mp4

# Explicitly refresh every indexed video
.venv/bin/python -m arabic_poc.index_videos --refresh-library --frame-seconds 10
```

Use `--resume <printed-run-directory>` with the same video paths and sampling interval after an interruption. This command generates **scene descriptions only**: it preserves previously indexed speech/OCR but does not transcribe newly added videos. Sampling can miss brief appearances; a frame timestamp is a point, not proof covering the next ten seconds. The King Faisal videos were added with scene descriptions, without newly generated transcripts.

## New corpus, OCR, transcription and graph CLI

For a fresh corpus, use a new storage directory and explicit input paths:

```sh
# Extraction only: local/remote OCR, ASR and optional descriptions
.venv/bin/python -m arabic_poc --storage rag_storage_new extract path/to/media --describe-images

# Extraction plus LightRAG graph ingestion
.venv/bin/python -m arabic_poc --storage rag_storage_new ingest path/to/media --describe-images

# Graph-based question answering (different from the app's hybrid search)
.venv/bin/python -m arabic_poc --storage rag_storage_new ask 'ما الذي يظهر في المصادر؟' --top-k 5
```

`--ocr-engine qwen --qwen-vl-model models/qwen2.5-vl-3b` selects local Qwen OCR; PaddleOCR Arabic is the local default. `--asr-model small` is the default local Whisper option; `--asr-model large-v3` is another generic Whisper checkpoint. Pass a repository ID/path for a compatible custom ASR checkpoint. Do not change embedding models on an existing index without rebuilding.

For the full refresh/activation workflow, consult [REFRESH.md](REFRESH.md) and `python -m arabic_poc.refresh_indexes --help`. That older workflow is distinct from the incremental description commands above; review its cache and face-enrichment limitations in [the code review](../docs/CODE_REVIEW.md).

## People and prompts

Put clear, single-person reference photos under `reference_faces/<canonical_id>/`, and store their source attribution in `sources.json`. Configure:

```env
FACE_GALLERY_PATH=reference_faces
FACE_MODEL=buffalo_l
FACE_MATCH_THRESHOLD=0.5
MIN_FACE_SIZE=40
MIN_DETECTION_SCORE=0.6
```

Add the same canonical ID to `person_aliases.json`, with `name_ar`, `name_en` and `aliases`. Avoid generic ambiguous given names. InsightFace skips unsuitable reference photos and caches valid embeddings; rebuild descriptions with `--refresh` after changing the gallery. Inspect source rights and model licensing for your deployment.

All application LLM instructions live in [`prompts.yaml`](prompts.yaml): `vision.*`, `context.*`, `query.*`, `retrieval.*`, and `answers.*`. Preserve their output schemas and `${payload}` placeholders. Framework prompts remain in RAG-Anything/LightRAG. Query/answer prompt edits need an app restart; description-prompt edits require regenerated descriptions to affect existing media.

## Tests and evaluation

```sh
.venv/bin/python -m pip install pytest pytest-asyncio
.venv/bin/python -m unittest discover -s arabic_poc -t . -p '*checks.py'
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest tests -q
```

The framework command disables automatic `.env` loading so local model/parser overrides do not change tests of default settings. The web checks currently need the local demo corpus and permission to bind localhost; most other checks mock model calls. See [the review](../docs/CODE_REVIEW.md) for known test limitations.

```sh
# Calls configured models; compares the same candidate pool with/without reranking
.venv/bin/python -m arabic_poc.evaluate_search --limit 4
# Full provisional Arabic/English set
.venv/bin/python -m arabic_poc.evaluate_search
```

[`validation/search_cases.json`](validation/search_cases.json) contains provisional expected files and timestamps that require human review. This evaluator deliberately measures **Recall@3**, reciprocal rank, timestamp recall and negative-query correctness; the app displays up to **5** sources. `evaluate.py` separately scores supplied extraction/retrieval JSONL (CER/WER and retrieval metrics).

Useful demo queries:

- `أرني صور الملك فيصل` / `Show me photos of King Faisal`
- `أرني فيديو الملك فيصل في بريطانيا`
- `أرني القبة الخضراء` / `Show me the green dome`
- `أرني صور عبد الرحمن بن عبد العزيز السديس`

Older experiment walkthroughs remain in this directory for context; this README and WEB_DEMO.md describe the current app workflow.
