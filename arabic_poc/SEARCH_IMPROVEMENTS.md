# Retrieval evaluation and serving

The library, images/scenes, and videos web routes now use `search_service.search`
in the server process. The embedding model loads lazily once, is retained between
requests, and is protected by a lock. Index JSON is read on each request so refreshed
content is visible without reloading weights. Changing the embedding model requires
a server restart and compatible index vectors. Legacy graph-QA routes still use
separate CLI processes.

Hybrid retrieval keeps up to 20 source candidates, including merged English/Arabic
query results. The configured remote text model scores their evidence for relevance;
the app shows the best five. Model failures or invalid scores retain hybrid order.
The reranking pool bypasses the final hybrid admission cutoff; scores below 0.5
are excluded after reranking. On failure, original hybrid admission rules apply. `reranking`, `candidate_count`, and `rerank_score` expose this behavior.

Person evidence distinguishes reference-gallery matches, names in generated visible
text, generic model mentions, publisher metadata, and transcript mentions. A name in
visible text is classified only when an explicit alias occurs in the description's
visible-text section; other visual mentions remain unverified. These labels do not
validate face identity or OCR correctness. The UI displays the origin under entities.

Run the paired evaluation:

```sh
.venv/bin/python -m arabic_poc.evaluate_search
```

The 32 bilingual cases in `validation/search_cases.json` include person aliases,
photos versus posters, landmarks, attributes, video timestamps, and negative queries.
Labels are provisional, based on existing saved evidence, and need human review.
They are a development regression set, not an independent accuracy benchmark.

The evaluation uses identical translations and embeddings for baseline and reranked
results. It reports macro Recall@3, reciprocal rank, timestamp recall (per-case
tolerance), negative-query correctness, per-case latency and reranker fallback status.
Reports are written incrementally to `validation/search_evaluation.json`.

## Aliases embedded in user queries

`person_aliases.json` is the reusable registry for Arabic/English display names,
canonical gallery IDs, and aliases. Add an entry with `name_ar`, `name_en`, and
an `aliases` list; no retrieval code edits are needed. Restart a running server
after registry edits, because the token trie is built once per process.

Before embedding or translation, `query_variants` retains the original request
and adds canonical English and Arabic name variants. Only matched name spans are
replaced: request words, constraints and punctuation are retained. Matching is
case-insensitive, folds Arabic spelling/diacritics, respects token boundaries,
and prefers the longest alias. It does not use fuzzy matches or partial given
names. The trie avoids scanning every person's aliases for every query token.

An alias shared by multiple IDs is deliberately unresolved. A longest ambiguous
match also blocks shorter aliases inside that span. Queries containing such a
match skip automatic LLM translation to avoid guessing which identity was meant.
Disambiguation requires an explicit longer name or a registry correction.

Hybrid retrieval also resolves explicit names in existing descriptions at search
time, so structured entity extraction is not required. A description mention is
not a face identity; provenance classification is unchanged. No gallery media,
descriptions or stored vectors are rewritten. Queries without known aliases keep
their existing translation/retrieval behavior.

### LLM-assisted person query resolution

`query_rewriting.py` runs before query translation/embedding. It supplies up to 30
canonical identities to the configured remote text model and asks for exact query
mentions mapped to supplied IDs. The model can resolve unlisted nicknames and
spelling variants. It receives canonical names rather than the alias lists.
Backend validation rejects unknown IDs, absent/overlapping spans and spans containing
request/negation terms or numbers. Only those spans are replaced; original queries
remain in retrieval. Invalid/ambiguous responses and provider failures retain the
existing deterministic alias/translation path. Explicit registry collisions remain
unresolved and are not sent for guessing.

Shortlisting uses existing alias hits, name similarity and English initials, not a
full semantic identity index. For very large galleries, obscure nicknames may fail
to shortlist their person; no identity is inferred outside the supplied candidates.
The model can still make a wrong choice among valid candidates, so identity ambiguity
requires evaluation. This adds one model call per query when the remote text backend
is enabled, including Arabic queries. Restart the app after code/registry changes.
