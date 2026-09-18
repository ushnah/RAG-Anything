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
