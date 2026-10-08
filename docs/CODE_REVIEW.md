# Code review and cleanup proposal

Reviewed 2026-10-07. The user subsequently approved the cleanup; all 26 listed generated paths and the four small code/ignore cleanups have now been removed. The pre-existing King Faisal alias change was preserved.

## Scope and approach

Inspected the application extraction, indexing, query expansion, retrieval, face enrichment, answer generation, web API/UI, packaging and documentation. Parsed 116 Python files across the application, framework, scripts, examples, tests and reproduction tools. Ran Pyflakes over application/framework/scripts/examples/reproduction code, examined entry points and references, and ran both test suites. This is a repository-wide review with deeper inspection of the app path, not a claim of exhaustive manual verification of every upstream parser or model backend.

## Fixes made

| Finding | Change and verification |
| --- | --- |
| Image refresh reused combined-index vectors by ID even when entity-enriched text changed. IDs can stay the same when description text stays constant. | Compare both ID and normalized text before reusing a vector, matching the video refresh behavior. Added `indexing_checks.py`: same description/ID, changed entity, regenerated vector and retained backup. Existing corpus was not reindexed as part of this review. |
| Package data included only prompt YAML. A normal wheel could omit required person aliases and browser assets. | Included aliases, web assets, demo text and evaluation cases. Built the wheel and verified all seven expected resources are present. |
| UI workspace count summed overlapping collections. | Count files in the Library rather than adding counts from all API datasets. |
| Visual routing test still mocked the old subprocess implementation, causing real model initialization. | Mock the current in-process search service; assert media filtering and no subprocess call. |
| Pipeline isolation tests loaded developer `.env` settings despite clearing environment variables. | Mock the remote module's dotenv loader so local-backend expectations remain deterministic. |
| Root/application documentation described different, outdated workflows. | Added a root app entry point; rewrote app README and web walkthrough with current configuration, commands, provenance, index layout and limits. Preserved upstream root documentation. |

## Remaining findings and boundaries

1. **Full refresh differs from incremental refresh.** `refresh_corpus.py` calls `RemoteVision` directly, without the `Extractor.describe()` face-gallery enrichment path. Its caches do not include prompt contents or gallery signatures. Existing cached source files are skipped. Do not use it as a drop-in replacement for regenerating face-enriched descriptions; use the documented incremental commands for that task. Consolidation needs explicit cache/versioning regression tests.
2. **Optional component failure boundary.** `Extractor.identify_faces()` handles initialization failure, but an exception escaping an already-initialized identifier can abort extraction. The existing `test_escaping_runtime_error_should_preserve_pipeline_output` remains an expected failure. The normal identifier handles its own operational exceptions; the caller boundary still deserves hardening.
3. **Face-cache settings.** Gallery cache validation uses file metadata and model name but not minimum face size/detection score. Tightening those thresholds can leave old cached references accepted until the cache is rebuilt.
4. **Multiple storage paths are intentional compatibility paths.** App library search uses JSON hybrid retrieval; the CLI `ask` and older API collections use LightRAG. Incremental image/video indexing does not rebuild the graph. The legacy visual-only builder bypasses face enrichment, and the older visual-only storage shape is not interchangeable with the combined search service. Do not delete those modules merely because the current UI calls `library`.
5. **Shared publishing logic remains duplicated.** Image/video indexers stage source files, embed, back up and publish separately. The image module also executes at import time. A shared, import-safe index writer would reduce drift, but should be done with tests for preserved OCR/transcripts, refreshes and failure recovery. The current change fixes the confirmed stale-vector bug without a broader rewrite.
6. **Scaling and concurrency.** Every search reloads index JSON, recomputes corpus token statistics and scans vectors. Index writers are not locked and multi-file publication is not transactional. A database/index service and object storage are architectural follow-ups, not unused-code deletion.
7. **Packaging duplication.** `setup.py`, `pyproject.toml` and `requirements.txt` disagree on Python/dependency bounds and extras. `setup.py` is still a build entry point; do not simply delete it. Consolidate metadata in `pyproject.toml`, retain a compatibility shim if needed, and revalidate editable/source/wheel builds separately.
8. **Tests and configuration tracking.** Root `.gitignore` has a broad `test_*` rule that hides newly added tests of that form unless already tracked. `active_indexes.json` is local/ignored deployment state, so a fresh clone does not automatically select the current corpus. The web tests also depend on local demo fixtures and saved results. A future CI pass should replace those dependencies with temporary fixtures and explicitly include app tests; the current CI invokes only `tests/`.
9. **Old experiment documentation.** The remaining per-experiment Markdown files include historical environment names and behavior. The updated app README/WEB_DEMO are authoritative for current usage. These older documents were retained as experiment history pending a separate consolidation decision.
10. **Diagnostic correctness/noise.** `doctor.py` checks local Qwen configuration only, so its generation flag can be misleading in remote mode. `remote.merge_person_context()` prints entities, sometimes twice per description because enrichment is idempotently applied twice. The debug print was subsequently removed in the approved cleanup.

## Static analysis

Pyflakes found no undefined names in the scanned production/example code. Its warnings include two genuinely unused application imports (`json` in `face_identifier.py`, `SimpleNamespace` in `face_checks.py`), a shadowed `enabled` name in `__main__.py`, framework package re-exports, and a Docling installation-probe import. The re-exports and installation probe have purposes and must not be removed by blanket autofix.

## Validation

- App unittest discovery: **79 tests run; 78 passed, 1 existing expected failure**.
- Framework pytest suite: **425 passed** with `PYTHON_DOTENV_DISABLED=1`. The first unisolated run had 3 failures because local parser/Whisper overrides changed default-setting expectations; no production defaults were changed to satisfy them.
- New offline indexing regression: entity changes invalidate cached combined-index embeddings even when evidence ID is unchanged.
- Wheel built using `pip wheel . --no-deps --no-build-isolation`; checked prompts, aliases, HTML/JS/CSS, demo text and evaluation cases inside the archive.
- `git diff --check`: passed.
- JavaScript runtime syntax check was unavailable because `node` is not installed/on PATH. The JS change is limited to the Library count expression; no browser visual regression run was performed.
- No live model evaluation, corpus regeneration, deployment or app restart was performed during this review. Test dependencies (`pytest`, `pytest-asyncio`, `pyflakes`) were installed in `.venv`.

## Approved cleanup — completed

The exact completed file/directory list and verification results are in [CLEANUP_CANDIDATES.md](CLEANUP_CANDIDATES.md). The user approved it before deletion.

Completed first pass: deleted only listed generated root logs, selected Python/test caches and `.DS_Store`; removed the two unused application imports and diagnostic `print(merged)`; collapsed the repeated `*.jpg` ignore entry. These do not implement application features. The cleanup list is a snapshot, not a command to recursively delete all matching paths.

Keep all media, attribution sidecars, reference photos, model weights, active and historical indexes, rollback snapshots, notebooks, validation records, manual diagnostic scripts and framework APIs. In particular, `sheikh_embeddings.pkl` appears unreferenced by current Python/notebook code, but its historical role is not established: retain it until separately approved. Keep `qwen_pages/`, which is used by `test_qwen_ocr.py`.
