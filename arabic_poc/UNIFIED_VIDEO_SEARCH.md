# Unified video search

The web demo opens **All videos** by default. Ask `Show me video of Makkah` or `أرني فيديو يتحدث عن ويكي بيانات` without choosing between visual and spoken collections.

## How it works

`video_search.py` imports existing records from the visual, spoken-Arabic, and media-demo indexes. It filters for video files, assigns a shared video ID from path and content hash, deduplicates identical evidence, and embeds each evidence record with BGE-M3. Publisher labels become separate whole-video records instead of being appended to every frame description.

A query searches all evidence and ranks each video by its best evidence score. The UI shows one card per video and up to three evidence buttons, preserving speech start/end times, frame timestamps, and evidence type. A metadata match has no timestamp; its player starts at the beginning without implying a frame-level match. Buttons show each evidence's original extracted/generated text. Video players seek to the selected timestamp.

The current index has 32 evidence records from eight videos. Not every video has all modalities: existing visual-only videos remain visual-only and the spoken samples retain their existing transcripts. This merges retrieval, not automatic re-extraction or temporal fusion. It does not build a new knowledge graph, perform face identification, or verify candidates with a model at query time.

## Build / refresh

After extracting or updating source indexes:

```sh
.venv311/bin/python -m arabic_poc.video_search build
.venv311/bin/python -m arabic_poc.video_search search 'Show me video of Makkah'
.venv311/bin/python -m arabic_poc.web
```

Storage: `rag_storage_arabic_video_search/index.json`. Input index names are listed in `video_search.INPUTS`. Rebuild after input changes; it is an explicit snapshot, not a live watcher. The specialized collections and their question-answering behavior remain available.

## Description guidance

Visual descriptions use the configured vision backend to name visible objects and describe colors/positions; extract structured entities for people, landmarks, places, organizations, and events; name landmarks only when distinctive and supported; preserve names on readable signs/captions as visible text; and avoid inferring identity from faces or motion from a still image. Prompt adherence is not guaranteed.

Existing descriptions were not silently rewritten. New visual builds carry `description_prompt_version: 3`; the standard extraction fingerprint also records that version when descriptions are enabled. Re-extract into a fresh index to compare the changed prompt fairly. Old evaluation findings remain applicable to the old descriptions, not a benchmark of the revised prompt.

## Validation and limits

Live HTTP results are saved in `validation/unified_video_queries.json`. Tests check grouping, provenance, timestamps, and media range serving. Browser visual QA has not been performed.

For the English Makkah query, the first two results are Makkah videos, matching respectively through publisher metadata and a generated frame description at 160 seconds. A Madinah video also ranks as a candidate. Scores have no calibrated no-match threshold; do not treat the returned list as verified geographic or person identification. The UI labels the evidence and candidates accordingly.

## Hybrid retrieval, abstention and temporal evidence

Unified search now combines Arabic-normalized whole-word BM25 matching with dense similarity. Arabic diacritics and alef variants are normalized; this is not Arabic morphological stemming, and normalized matches are not literal quotations. Ranking uses 75% nonnegative cosine similarity and 25% bounded BM25 (`bm25/(bm25+2)`). Original evidence text remains unchanged.

The earlier always-return-candidates behavior is superseded: evidence must have cosine similarity >= 0.55, or match all non-stopword query terms with cosine >= 0.35. If none qualifies, the UI displays insufficient evidence and no video cards. Configure `VIDEO_MIN_SCORE` in `.env` or use `search --min-score`. This policy is heuristic, not a calibrated probability or verified absence claim.

Smoke tests in `validation/hybrid_video_queries.json`: Makkah returns the two Makkah videos; submarine and airplane queries abstain. **The Arabic Wikidata lecture query also abstains despite relevant content existing**, a known false negative caused by the conservative threshold and noisy transcription. Lower thresholds improve recall but admit weak matches. Do not claim these four development cases establish production accuracy; independent positive/negative calibration remains necessary.

Each returned evidence button can now link to speech/frame evidence from the same video. A sampled frame inside a speech interval is `overlap`; a frame within two seconds of that interval is `nearby`. These labels mean temporal proximity only, not semantic agreement. No interval is invented between successive sampled frames. Metadata does not participate. `--nearby-seconds 0` restricts links to overlap. Existing visual-only/spoken-only inputs do not acquire missing modalities automatically. This is temporal evidence navigation, not full temporal query reasoning.

Tests cover Arabic normalization, weak-match abstention, BM25 lexical recovery, same-video boundaries, exact overlap, nearby tolerance, and metadata exclusion. Existing live source evidence may contain OCR/ASR hallucinations; temporal association does not verify extraction quality.

Current corpus audit: only the captioned Internet-in-a-Box video has a speech/frame overlap in the merged records. Its 0–3.7-second ASR fragment was previously flagged as unverified/non-speech hallucination in `MEDIA_VALIDATION.md`. Therefore the current corpus does **not** establish useful real speech/visual alignment quality; the temporal boundary behavior is verified by controlled tests. More videos extracted with both modalities are needed for a quality benchmark.
