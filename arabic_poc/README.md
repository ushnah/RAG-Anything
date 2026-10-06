# Arabic RAG CLI

Default extraction policy: PaddleOCR Arabic for every PDF page, image, and sampled video frame; Whisper (Arabic, `small` by default) for audio and video speech; BGE-M3 dense embeddings; RAG-Anything/LightRAG graph + vector retrieval; Qwen through an OpenAI-compatible endpoint. Includes offline extraction and retrieval evaluation.

## Setup

Use the existing Python 3.11 environment:

```sh
.venv311/bin/python -m pip install -e .
.venv311/bin/python -m pip install -r arabic_poc/requirements.txt
# macOS, needed for audio/video:
brew install ffmpeg
```

Add the settings in `arabic_poc/env.example` to your existing `.env`. Start your Qwen server or configure a hosted Qwen endpoint. Local extraction and embeddings may download model weights on first use. Speech runs on CPU for Mac compatibility.

## Run

```sh
# Extraction only: no Qwen or embedding model required
.venv311/bin/python -m arabic_poc extract data/

# Extract/cache and index (or reuse cached extraction)
.venv311/bin/python -m arabic_poc ingest data/

.venv311/bin/python -m arabic_poc ask 'ما الذي تذكره المصادر عن هذا الموضوع؟'

# Select one consistent ASR model and sampling interval for a corpus
.venv311/bin/python -m arabic_poc --storage rag_storage_arabic_large ingest data/ --asr-model large-v3 --frame-seconds 15
```

Default storage is `rag_storage_arabic/`, separate from your existing experiment. `sources/*.json` contains original extraction, normalized retrieval text, source path, stable passage ID, extraction engine, and page or time anchor. `index/` contains the LightRAG stores. Repeat ingestion uses stable document IDs; extraction is cached. If a source or extraction settings change, use a new storage directory to avoid stale graph evidence. Run one ingestion process at a time per storage directory.

Optional metadata: create `book.pdf.metadata.json` beside `book.pdf`, for example:

```json
{"title": "عنوان الكتاب", "author": "المؤلف", "volume": "1", "chapter": "الباب الأول"}
```

Pages are one-based physical PDF pages, not printed page labels. Speech anchors contain start/end seconds. Frame anchors indicate the sampling grid (approximate visual timestamps); frames are sampled every 30 seconds by default. PDFs preserve recognized text. Images optionally include separately labeled generated visual descriptions. Tables follow OCR reading order. Confidence is null because the existing parser discards confidence scores. Metadata is document-level; chapters are not inferred.

## Answers and fidelity

Retrieval uses normalized Arabic; generation receives original extracted passages. Output includes the Arabic answer, citations with passage IDs and verbatim quotes, and retrieved source records with anchors. Invalid/missing quotations suppress the generated answer. This validates quote substrings, not whether every claim follows from the evidence. OCR/ASR output is not a canonical Quran/Hadith edition: inspect the linked page or recording before treating it as an exact religious quotation. No canonical-text verification is implemented.

Video includes speech and sampled on-screen text; enable `--describe-images` to also describe visible content. Silent videos are supported. Brief on-screen text between samples may be missed.

## Lightweight checks

```sh
.venv311/bin/python -m unittest arabic_poc.checks -v
```


## Qwen OCR and visual understanding

Use your existing local Qwen2.5-VL model for difficult scans. This is the generic Qwen checkpoint, not an Arabic-fine-tuned model unless you supply one explicitly.

```sh
.venv311/bin/python -m arabic_poc --storage rag_storage_qwen ingest data/ --ocr-engine qwen --qwen-vl-model models/qwen2.5-vl-3b --describe-images
```

`--describe-images` adds Arabic visual descriptions for standalone images and sampled video frames (not PDF figures). They are labeled `qwen-vl:generated` and `kind=visual_description`; these are model interpretations, not transcriptions. They must not be treated as exact source quotations. OCR and descriptions can load separate model instances; leave descriptions off on memory-constrained machines. Qwen uses CUDA, MPS, or CPU automatically. Pages reaching the output limit fail rather than silently indexing truncated output.

### Reference face identification

Face identification is a separate optional enrichment step. It runs on the same sampled video frames already sent to OCR and the visual model; it does not replace the remote VLM or ask it to identify faces. Install the optional backend with:

```sh
pip install -e '.[face]'
```

Create a gallery whose directory names are the canonical identifiers:

```text
reference_faces/
├── sheikh_x/
│   ├── 001.jpg
│   └── 002.jpg
└── sheikh_y/
	└── 001.jpg
```

To seed the initial Saudi scholars and leaders from Wikimedia Commons, run this from the repository root when internet access is available:

```sh
python scripts/download_reference_faces.py --gallery reference_faces --limit 3
```

The downloader writes `sources.json` inside each person directory with the Commons page, image URL, artist, and reported license. Add or override a person with `--person folder_name=Search Name`. Review the downloaded candidates before using them; the face gallery loader will skip images with no face or multiple faces.

Configure it in `.env`:

```env
FACE_GALLERY_PATH=reference_faces
FACE_MODEL=buffalo_l
FACE_MATCH_THRESHOLD=0.5
MIN_FACE_SIZE=40
MIN_DETECTION_SCORE=0.6
```

InsightFace/ArcFace generates and caches one normalized embedding per valid reference image. Images with no face or multiple faces are skipped. Each sampled frame receives a separate `person_matches` field such as `{"name":"sheikh_x","similarity":0.83,"source":"face_recognition"}`; unknown faces use `"name": null`. Known names are appended to the searchable normalized text, while OCR text, VLM entities, timestamps, source paths, and document metadata remain unchanged. The threshold is a starting configuration, not a universal value; calibrate it with held-out images from the intended camera, pose, and lighting conditions.

This identifies only people represented in the local gallery and should not be treated as ground truth without review. InsightFace model-pack licensing and any gallery image permissions must be checked for the deployment; the package and pretrained model assets may have separate licensing terms.

For a single image, run `./.venv/bin/python -m arabic_poc.sheikh_poc path/to/frame.jpg`. The interactive walkthrough is `arabic_poc/sheikh_poc.ipynb`; it prints the name-only `{"people": [...]}` result and the enriched visual description separately.

To update face entities in an already-saved visual-lab result after changing the reference gallery, without regenerating descriptions, run `./.venv/bin/python -m arabic_poc.sheikh_poc --refresh-results arabic_poc/validation/your-visual-lab.json`. This reuses the frame paths stored in the JSON and updates `entities.people`, `person_matches`, and the top-level `people` list only.

## Arabic-specialized ASR

`--asr-model large-v3` uses generic Whisper with Arabic decoding. To use an Arabic-fine-tuned Whisper checkpoint, pass its Hugging Face repository ID or local directory path containing `/`. Choose a checkpoint you have evaluated on the client's audio; the shorthand “whisper-large-v3-ar” does not identify a unique model. Hugging Face ASR runs on CPU and preserves returned segment timestamps.

## Demo and prerequisite check

```sh
.venv311/bin/python -m arabic_poc.doctor
.venv311/bin/python -m arabic_poc ingest arabic_poc/demo/
.venv311/bin/python -m arabic_poc ask 'من يشرف على فهرسة المخطوطات؟'
```

The demo text is explicitly fictional. Configure generation using `env.example`; `BGE_MODEL` can point to a locally downloaded BGE-M3 directory. Extraction runs locally. Indexing sends extracted text to the configured generation endpoint for graph construction; answering sends retrieved source passages there too.

Passages retain exact original substrings, character offsets, and overlapping windows of at most 1,800 characters. This bounds per-source context while retaining page/media anchors. The extraction cache schema changed: use a new storage directory for caches created by earlier versions.

## Evaluation

Create UTF-8 JSONL with manually verified references. For extraction, use one line per page/segment:

```json
{"id":"page-1","reference":"نص عربي","hypothesis":"نص عربى"}
```

For retrieval, copy ordered passage IDs from the query's `sources` output and compare with manually labeled relevant passage IDs:

```json
{"id":"question-1","expected_ids":["passage-a"],"retrieved_ids":["passage-b","passage-a"]}
```

```sh
.venv311/bin/python -m arabic_poc.evaluate evaluation.jsonl
```

Reports per-case and macro-averaged CER/WER (raw and normalized), Recall@5/10, MRR, and Hit@10. Run extraction in separate storage directories for PaddleOCR and Qwen to compare identical inputs. This is an offline scorer, not an automatic corpus/question generator. Faithfulness and canonical religious-text verification still require human review. Kraken, MMORE, and production concurrency are outside this first implementation.

Model API references: [BGE-M3](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/README.md), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

If the default LightRAG tokenizer download is unavailable, set `RAG_TOKENIZER_MODEL=models/qwen2.5-vl-3b` in `.env` to use the existing local Qwen tokenizer. Keep tokenizer and embedding settings consistent throughout the lifetime of an index. Token counts from this tokenizer approximate generation budgets when the generation model differs.

### Integration design

The Arabic adapters convert each modality into source-anchored text before calling RAG-Anything's `insert_content_list`. LightRAG builds and queries the graph and dense vector stores in `mix` mode. This first POC does not enable RAG-Anything's native multimodal processors or BGE-M3 sparse vectors. Visual descriptions are indexed alongside OCR and ASR records through the same source-preserving path. Final generation requests JSON and gets one repair attempt if exact citation validation fails.


## Local demo web UI

Run `.venv311/bin/python -m arabic_poc.web` and open http://127.0.0.1:8765. See [WEB_DEMO.md](WEB_DEMO.md) for the team-demo walkthrough.

## Image and video scene search

See [VISUAL_DEMO.md](VISUAL_DEMO.md) for the **Images & scenes** collection: Qwen visual descriptions, BGE-M3 retrieval, source labels, and video timestamp playback. This separate media-search path returns candidates without generating an answer. It is a description-based baseline, not direct pixel embedding or verified landmark recognition.

## LLM prompt configuration

Application prompts live in [`prompts.yaml`](prompts.yaml), grouped by key prefix:
`vision.*`, `context.*`, `query.*`, `retrieval.*`, and `answers.*`.
The code reads them with `get_prompt()` from `prompt_loader.py`. Dynamic questions,
evidence, images and person data remain structured payloads in Python.

Edit the YAML to change instructions; preserve JSON output schemas and the
`${payload}` placeholder in person-context templates. YAML folded blocks (`>-`)
join wrapped lines with spaces. The loader preserves literal JSON braces and
substitutes only explicit template placeholders. Prompts are cached per process,
so restart the app or notebook kernel after editing them. No reindexing is needed
for query/answer changes; to apply description-prompt changes to existing media,
rerun description extraction. Framework-managed LightRAG/RAG-Anything prompts
remain managed by those frameworks.
