# Real Arabic audio/video verification

## Downloaded samples

| Local file | Source and attribution | Test input |
| --- | --- | --- |
| `data/media_samples/arabic_greeting.ogg` | [السلام عليكم](https://commons.wikimedia.org/wiki/File:السلام_عليكم.ogg), TwoThirty, public domain | Complete 1.335-second recording |
| `data/media_samples/airplane_arabic_60_90s.wav` | [Airplane article in Arabic Wikipedia](https://commons.wikimedia.org/wiki/File:Airplane_article_in_Arabic_Wikipedia.ogg), Riadismet, public domain | Seconds 60–90, converted to mono 16 kHz WAV |
| `data/media_samples/internet_in_a_box_arabic.webm` | [Internet-in-a-Box (Arabic)](https://commons.wikimedia.org/wiki/File:Internet-in-a-Box_(Arabic).webm), Ala'a Najjar, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | Complete 40.754-second video, unchanged |

Each media file has a `.metadata.json` sidecar with source URL, attribution, SHA-256, and transformations. The complete spoken article is retained in `data/media_originals/airplane_arabic.ogg`. An initial 0–30-second intro excerpt is also retained there; the narration verification uses 60–90 seconds instead. Clip timestamps are relative to the local WAV; add its `original_start_seconds` metadata (60) to locate the same passage in the complete recording.

## Extraction results

Actual models: generic OpenAI Whisper `small`, Arabic decoding, CPU; PaddleOCR Arabic PP-OCRv5. These are not Arabic-fine-tuned Whisper or Qwen OCR benchmarks.

- Greeting: one segment, exact text `السلام عليكم`, timestamp 0–1.335147 seconds.
- Spoken article excerpt: nine segments, all timestamps within 0–30 seconds. The opening segment reads `الطائرة هي مركبة جوية أثقل من الهواء` at 0–4 seconds.
- Video: three OCR records at 0, 15, and 30 seconds. The 15-second record includes `يحتوي على موسوعة ويكيبيديا`, checked against the actual frame.
- Video ASR also emitted `ath` at 0–3.7 seconds; earlier runs emitted different non-Arabic fragments. Treat this segment as unverified and unsuitable as speech evidence. An audio stream does not imply intelligible speech, and the present ASR path does not reliably reject music/non-speech hallucinations.

Full passages, metadata, and structural checks are saved in [validation/media_extraction.json](validation/media_extraction.json).

## Fixes discovered by these inputs

1. Clamp ASR segment ends to the probed media duration; discard segments outside the media.
2. Stop traversing arbitrary PaddleOCR metadata as recognized text. Temporary file paths and model settings no longer enter source passages.
3. Use `fps=...:round=up` so sampled frames align with the labeled sampling grid instead of drifting toward the next half interval. The 15-second frame was visually compared with a direct seek to 15 seconds. Timestamps remain frame-quantized.

11 POC checks and 15 parser regression tests passed after these fixes. Extraction cache schema is now 5: use a new storage directory for older caches.

## Reproduce extraction

```sh
.venv311/bin/python -m arabic_poc --storage rag_storage_arabic_media_final extract data/media_samples --frame-seconds 15
```

## Quality limits

This verifies real media handling and source anchors, not production accuracy. The narrated excerpt contains recognition errors (for example `تستخدم كوزلة نقل جوئي`), video OCR includes noisy recognition of small/mixed-language text, and video ASR produced unstable fragments. No manually aligned CER/WER benchmark or full audio timestamp alignment audit has been completed. The advanced Qwen visual-description and Arabic-specialized ASR paths were not exercised by this run.

## Reproduce indexing and queries

With the configured Qwen server running:

```sh
RAG_TOKENIZER_MODEL=models/qwen2.5-vl-3b .venv311/bin/python -m arabic_poc --storage rag_storage_arabic_media_final ingest data/media_samples --frame-seconds 15
RAG_TOKENIZER_MODEL=models/qwen2.5-vl-3b .venv311/bin/python -m arabic_poc --storage rag_storage_arabic_media_final ask 'كيف يعرّف التسجيل الطائرة؟'
RAG_TOKENIZER_MODEL=models/qwen2.5-vl-3b .venv311/bin/python -m arabic_poc --storage rag_storage_arabic_media_final ask 'ما المحتوى الموجود في جهاز الإنترنت في صندوق كما يوضح الفيديو؟'
```

## Indexing result

All 14 source passages reached `processed` status; see [validation/media_index_status.json](validation/media_index_status.json). The local `qwen2:7b` endpoint initially timed out on the large video OCR passage. Generation now uses a configurable `QWEN_MAX_TOKENS` limit (default 1536) and no automatic HTTP retries. Resuming ingestion completed successfully.

The video passage hit the output cap, so its graph extraction may be incomplete. Original extracted text is still stored for vector retrieval and citation validation. This is an engineering smoke-test success, not evidence of complete graph extraction or high OCR/ASR accuracy.

## End-to-end question results: quality checks failed

Both queries ran against the real media index using graph + vector retrieval. Both retrieved the expected supporting passage within the three returned passages, and their citation quotes were exact substrings of the stored extraction. These mechanical checks do **not** establish answer faithfulness.

| Question | Verified evidence | Generation review |
| --- | --- | --- |
| كيف يعرّف التسجيل الطائرة؟ | Narration at 0–4 seconds of the excerpt (60–64 seconds of the original recording); exact quote `الطائرة هي مركبة جوية أثقل من الهواء` | **Failed:** the answer mixes Chinese into Arabic. |
| ما المحتوى الموجود في جهاز الإنترنت في صندوق كما يوضح الفيديو؟ | Expected 15-second frame retrieved; the generated answer instead cited the 30-second frame | **Failed:** the answer claims a 4 GB capacity absent from the retrieved extracted text, and its chosen citation does not support the full answer. |

Saved evidence:

- [Audio answer and sources](validation/media_audio_answer.json)
- [Video answer and sources](validation/media_video_answer.json)
- [Query checks and review verdicts](validation/media_query_checks.json)

**Conclusion:** downloads, decoding, ASR/OCR execution, indexing, retrieval, and source anchors were exercised on real files. The current generic Whisper-small/PaddleOCR/Qwen2:7b combination is **not yet client-ready for reliable Arabic answers**. Speech/non-speech handling, extraction quality, Arabic generation, and claim-to-citation support still need improvement. Exact-quote validation alone is insufficient; these saved outputs intentionally retain the failures for review.
