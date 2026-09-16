# POC validation

- 11 Arabic POC checks passed: normalization, verbatim quotes, citation rendering, metadata, page/media anchors, extraction caching IDs, bounded original passages, visual provenance, metrics, and failed-index propagation.
- 15 existing parser regression tests passed (`testparser_wiring.py`, `testpaddleocr_parser.py`).
- Live fictional Arabic text ingestion passed with cached BGE-M3, the local Qwen tokenizer, and the configured `qwen2:7b` endpoint. LightRAG persisted 15 graph nodes, 7 edges, and the source chunk.
- Live graph + vector query “من يشرف على فهرسة المخطوطات؟” retrieved the correct passage and answered with مريم, citing the exact original quote “تشرف مريم على فهرسة المخطوطات”.
- Verified demo index saved to `rag_storage_arabic_demo/` (gitignored).
- Real audio/video smoke tests have now been run: see [MEDIA_VALIDATION.md](MEDIA_VALIDATION.md). The representative client corpus has not been benchmarked; no overall quality scores or canonical Quran/Hadith verification are claimed.

Run the saved demo from the repository root:

```sh
RAG_TOKENIZER_MODEL=models/qwen2.5-vl-3b .venv311/bin/python -m arabic_poc --storage rag_storage_arabic_demo ask 'من يشرف على فهرسة المخطوطات؟'
```

The configured Qwen server must be reachable. The local tokenizer avoids a failed default tiktoken download observed during testing. Generation initially copied normalized text into quotations; normalized text is now excluded from the generation prompt. Exact-quote validation remains mandatory, with one repair attempt and safe abstention on failure.

Real media verification found generation quality failures despite valid quote substrings: see [MEDIA_VALIDATION.md](MEDIA_VALIDATION.md). This POC is not yet validated as client-ready for faithful Arabic answers.
