# Refreshing the demo corpus

The refresh retains original PaddleOCR passages and text, regenerates speech with remote Whisper and sampled frame/image descriptions with remote Gemma, then builds a fresh LightRAG graph using remote GPT-OSS and local BGE-M3 embeddings.

```sh
.venv311/bin/python -m arabic_poc.refresh_corpus rag_storage_arabic_remote_v2
.venv311/bin/python -m arabic_poc.refresh_indexes rag_storage_arabic_remote_v2 --ingest
# After query checks succeed, publish the storage mapping:
.venv311/bin/python -m arabic_poc.refresh_indexes rag_storage_arabic_remote_v2 --activate
.venv311/bin/python -m arabic_poc.web
```

If remote vision is unavailable, pass `--local-vision-fallback` to the corpus refresh. It reuses successful Gemma responses and generates missing descriptions with local Qwen; concise English fallback descriptions are translated by remote GPT-OSS. Engine fields record this distinction. This is an explicit refresh option, not an automatic change to the configured provider.

Extraction caches each completed remote response. Audio streams are converted to mono 16 kHz WAV before upload; segment timestamps remain relative to the local media. Video descriptions are sampled every 20 seconds by default. Existing OCR timestamps retain their original sampling grid.

The complete-source manifest and every passage's `processed` graph status are checked before building or activating indexes. The new storage contains the source records, LightRAG graph/vector stores, visual vector index, unified hybrid search index and a `refresh-report.json` with counts and model provenance.

`arabic_poc/active_indexes.json` is the local deployment pointer. Both the web server and its search subprocess read it. All legacy graph-query endpoints use the full refreshed corpus after activation; the UI's media filters still select source types. Restart the server after changing the pointer.

Previous storage directories are retained. To roll back, rename the deployment pointer and restart the server; the original default paths become active again.

Descriptions and ASR outputs are model-generated evidence, not verified truth. Music/non-speech transcription hallucinations and incorrect landmark descriptions remain possible. Rebuilding does not by itself establish better answer quality.

## Completed refresh

Active storage: `rag_storage_arabic_remote_v2`. All 12 sources and 85 passages completed graph ingestion using remote GPT-OSS. The unified search index contains 91 records, including separately attributed source labels; the visual index contains 27 descriptions. BGE-M3 remains local.

- 54 new remote Whisper speech segments.
- 5 remote Gemma descriptions.
- 22 regenerated local Qwen fallback descriptions after Gemma read timeouts; 21 used English descriptions translated by remote GPT-OSS.
- 3 retained PaddleOCR passages and the original text document.

All media anchors are within the local file durations. 24 regression checks passed. Search checks found Makkah sources, the green-dome video and the airplane recording, and returned no match for an unrelated submarine query. Arabic definite-article normalization and a stricter semantic threshold for short speech fragments address regressions found during this refresh. These heuristics still require broader evaluation.

A rebuilt-graph answer check returned the airplane definition with an exact supporting quote. These smoke tests do not establish comprehensive faithfulness or ASR/description accuracy. The remote vision provider remains configured for future extraction; this refresh used an explicit local fallback where needed.

Reports in the active storage: `refresh-report.json`, `extraction-validation.json`, `search-validation.json`. Older storage directories remain available for rollback.
