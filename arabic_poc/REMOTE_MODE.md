# Remote models with the original retrieval

The demo keeps its existing retrieval and indexes. Model providers are configured independently in the git-ignored `.env`; credentials never enter browser JavaScript.

| Function | Configured model | Runs |
| --- | --- | --- |
| Embeddings | BGE-M3 | Locally |
| Arabic OCR | PaddleOCR | Locally |
| Arabic answers and LightRAG graph extraction | `cerebras/gpt-oss-120b` | Arbisoft gateway |
| Image and sampled video-frame descriptions | `inhouse/gemma-4-12b` | Arbisoft gateway |
| Audio and video speech transcription | `whisper` | Arbisoft gateway |

The gateway model list did not expose an embedding model, so BGE-M3 remains local. PaddleOCR remains local because the Gemma OCR probe hallucinated text. Remote vision is used for descriptions, which remain generated interpretations rather than exact source quotations. Gateway aliases do not establish the underlying hosting location.

## Retrieval paths

- **UI Library:** existing BGE-M3 semantic similarity plus Arabic keyword/BM25 matching, weak-match rejection, source grouping and time-aligned evidence. It reads `rag_storage_arabic_video_search/index.json` and returns source candidates.
- **Standard RAG `ingest` / `ask`:** RAG-Anything and LightRAG continue to build and query the graph and vector index. BGE-M3 supplies embeddings; remote GPT-OSS supplies graph extraction, query processing and answer generation. Original passages and literal citation validation remain.

The experimental remote relevance-scoring retrieval bypass has been removed. Historical `remote_search_queries.json` and `remote_http_query.json` reports describe that superseded experiment, not the active search implementation.

## Configuration

```dotenv
MODEL_BACKEND=remote
TEXT_BACKEND=remote
VISION_BACKEND=remote
ASR_BACKEND=remote
OCR_BACKEND=local
REMOTE_BASE_URL=https://litellm.arbisoft.com/v1
REMOTE_API_KEY=replace-locally
REMOTE_TEXT_MODEL=cerebras/gpt-oss-120b
REMOTE_VISION_MODEL=inhouse/gemma-4-12b
REMOTE_ASR_MODEL=whisper
REMOTE_MAX_TOKENS=4096
RAG_TOKENIZER_MODEL=models/qwen2.5-vl-3b
```

The local tokenizer performs token counting, not generation. Existing embeddings remain compatible because BGE-M3 is unchanged. Existing transcripts/descriptions keep their original engine provenance; switching providers does not regenerate or improve them. Re-extract into a fresh storage directory to compare providers, then ingest and rebuild the library index when adding new evidence.

```sh
.venv311/bin/python -m arabic_poc.web
.venv311/bin/python -m arabic_poc --storage rag_storage_arabic_demo ask 'من يشرف على فهرسة المخطوطات؟'
```

To use a local model for a role, set its `*_BACKEND=local` and retain the corresponding local model configuration. Set all four roles to local for a complete rollback, then restart the server. The UI stack labels describe the current mixed configuration.

## Verification and limits

- 23 regression checks passed, including retrieval, temporal alignment, citation handling, provider selection and local HTTP behavior.
- Live LightRAG + remote GPT-OSS query retrieved 12 entities, 7 relations and one chunk, answered `مريم`, and quoted the supporting source exactly. Saved in `validation/hybrid_remote_answer.json`.
- Restored BGE-M3 hybrid search returned two Makkah videos for `Show me video of Makkah`.
- Remote Gemma described a green dome and minarets in Arabic; remote Whisper transcribed `السلام عليكم` with a 0–1.24-second segment.
- Remote OCR emitted inaccurate text, so it is disabled in the active configuration.

These are smoke tests, not a comprehensive Arabic-quality benchmark. Older extraction errors remain in existing indexes. Exact substring citations do not alone establish answer faithfulness.

## Refreshed corpus

The active corpus has now been rebuilt. See [REFRESH.md](REFRESH.md) for counts, validation and the local vision fallback used after remote Gemma timeouts. Existing older indexes are retained for rollback.
