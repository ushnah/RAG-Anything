# Local web demo

From the repository root:

```sh
.venv311/bin/python -m arabic_poc.web
```

Open **http://127.0.0.1:8765**. Keep this terminal and your Ollama server running. Stop the web server with Ctrl+C. For another port, add `--port 8766`.

The UI uses `rag_storage_arabic_demo`, `rag_storage_arabic_media_final`, and `rag_storage_arabic_spoken`. The new **Spoken Arabic** collection contains a tutorial lecture and a Syrian-Arabic speaker; see [SPOKEN_VIDEOS.md](SPOKEN_VIDEOS.md). It reads each index's tokenizer/embedding settings automatically and uses the Qwen configuration in `.env`. No additional web dependencies are required.

## Suggested team-lead walkthrough

1. Choose **Arabic library**, select the manuscript-cataloguing question, and click **Ask a question**.
2. Expand the answer's citations and compare them with the original extracted text.
3. Choose **Audio & video**. Inspect a narration segment and play it from its timestamp.
4. Ask one of the suggested media questions. Show the retrieved evidence and distinguish a verified quote substring from a faithful answer.
5. If live generation is slow, open a **Recorded media example**. The UI labels saved results and their known quality failures explicitly.

Live generation can take several minutes. The UI handles one live question at a time to avoid loading competing model processes. Reloading the page does not cancel a running query; wait for it to finish before sending another.

## Scope

- Runs only on this Mac; it binds to `127.0.0.1`.
- No public hosting or credentials in browser code.
- Questions, saved results, original passages, citations, and source playback.
- No upload or ingestion screen: add/index new files through the existing CLI before the demo.
- Current model failures are unchanged: mixed-language answers, unsupported claims, OCR/ASR errors, and incomplete graph extraction remain possible.
- Five HTTP tests passed for collection access, byte-range playback, rejected file/path/origin requests, saved-result labeling, and serialized jobs. JavaScript syntax and Python compilation were checked. Browser visual testing was not performed.

```sh
.venv311/bin/python -m unittest arabic_poc.web_checks -v
```

## Live verification

A real question was submitted through `/api/ask` and polled through `/api/jobs/...`. It completed in 36.4 seconds, identified مريم, and returned the exact quote `تشرف مريم على فهرسة المخطوطات` with the correct source. This verifies the web backend → existing CLI → local Qwen → citation response path; it does not establish general answer quality.

## Visual media search

The Images & scenes collection adds generated image/frame descriptions and source labels, with image/video filters. See [VISUAL_DEMO.md](VISUAL_DEMO.md) for setup, attribution, example queries, and limits.
