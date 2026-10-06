"""Process-local embedding worker and evidence-aware candidate reranking."""
from .prompt_loader import get_prompt
import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from .__main__ import normalize
from .person_aliases import resolve_people, name_key
from .query_translation import query_variants, search_variants
from .remote import chat, enabled
from .visual import embedding_model

_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def encoder():
    return embedding_model()


def provenance(row):
    kind = row.get('evidence_type', row.get('anchor', {}).get('kind'))
    people = resolve_people(row.get('entities', {}))
    matches = row.get('anchor', {}).get('person_matches', row.get('person_matches', []))
    matched = {p['id'] for m in matches if m.get('name')
               for p in resolve_people({'people': [m['name']]})}
    if kind == 'speech':
        return {'type': 'transcript_mention', 'people': people}
    if kind == 'metadata':
        return {'type': 'publisher_mention', 'people': people}
    if kind in ('frame', 'page'):
        return {'type': 'visible_text', 'people': people}
    text = row.get('original_text', '')
    visible = text.split('النص المرئي:', 1)[1].split('|', 1)[0] if 'النص المرئي:' in text else ''
    visible_key = ' ' + name_key(visible) + ' '
    return {'type': 'generated_visual_description', 'people': [dict(p, provenance=
            'face_gallery_match' if p['id'] in matched else
            'visible_text_mention' if any(' ' + name_key(a) + ' ' in visible_key for a in p['aliases']) else 'visual_model_mention') for p in people]}


def rerank(question, result, top_k=5):
    candidates = result['sources'][:20]
    for row in candidates:
        row['person_evidence'] = provenance(row)
        for match in row.get('matches', []):
            match['person_evidence'] = provenance(match)
    if not candidates or not enabled('text'):
        kept = [r for r in candidates if r.get('hybrid_qualified', True)][:top_k]
        return dict(result, sources=kept, no_match=not kept, answer=result.get('answer', '') if kept else 'لم أجد أدلة قوية كافية لهذا البحث. جرّب صياغة أخرى.', reranking='hybrid_fallback')
    payload = [{'id': str(i), 'document': r['document'], 'media_type': r['media_type'],
                'evidence': [{'text': m['original_text'][:1800], 'provenance': provenance(m)}
                             for m in [r, *r.get('matches', [])]]} for i, r in enumerate(candidates)]
    try:
        raw = chat([{'role': 'system', 'content':
            get_prompt('retrieval.reranking')},
            {'role': 'user', 'content': json.dumps({'query': question, 'candidates': payload}, ensure_ascii=False)}], json_output=True)
        scores = json.loads(raw)['scores']
        if not isinstance(scores, list) or len(scores) != len(candidates):
            raise ValueError('Incomplete scores')
        mapped = {s['id']: s['score'] for s in scores}
        if set(mapped) != {str(i) for i in range(len(candidates))}:
            raise ValueError('Invalid IDs')
        for i, row in enumerate(candidates):
            score = mapped[str(i)]
            if type(score) not in (int, float) or not 0 <= score <= 1:
                raise ValueError('Invalid score')
            row['rerank_score'] = score
        candidates.sort(key=lambda r: (r['rerank_score'], r['hybrid_score']), reverse=True)
        kept = [r for r in candidates if r['rerank_score'] >= .5][:top_k]
        return dict(result, sources=kept, no_match=not kept, answer=result.get('answer', '') if kept else 'لم أجد أدلة قوية كافية لهذا البحث. جرّب صياغة أخرى.', reranking='llm', candidate_count=len(candidates))
    except Exception:
        logging.getLogger(__name__).warning('Reranking unavailable; preserving hybrid order')
        kept = [r for r in result['sources'] if r.get('hybrid_qualified', True)][:top_k]
        return dict(result, sources=kept, no_match=not kept, answer=result.get('answer', '') if kept else 'لم أجد أدلة قوية كافية لهذا البحث. جرّب صياغة أخرى.', reranking='hybrid_fallback')


def search(question, storage, media_type='all', use_reranker=True):
    with _LOCK:
        model, name = encoder()
        index = json.loads((Path(storage) / 'index.json').read_text())
        if index['embedding_model'] != name:
            raise ValueError('Embedding model differs from index')
        variants = query_variants(question)
        vectors = model.encode([normalize(q) for q in variants], return_dense=True)['dense_vecs']
        result = search_variants(index, variants, vectors, media_type=media_type,
                                 top_k=20 if use_reranker else 5)
        return rerank(question, result) if use_reranker else result
