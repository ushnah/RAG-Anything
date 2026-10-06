"""Arabic retrieval expansion while preserving the user's original query."""
from .prompt_loader import get_prompt
import json
import logging
import re
from .remote import chat, enabled
from .person_aliases import canonical_query_variants, MATCHER
from .query_rewriting import rewrite_person_query

logger = logging.getLogger(__name__)


def query_variants(question):
    rewritten = rewrite_person_query(question)
    variants = list(dict.fromkeys([question, *canonical_query_variants(rewritten), *canonical_query_variants(question)]))
    if any(m['ambiguous'] for m in MATCHER.find(question)):
        return variants
    if not re.search(r'[A-Za-z]', question) or not enabled('text'):
        return variants
    try:
        raw = chat([
            {'role': 'system', 'content': get_prompt('query.arabic_translation')},
            {'role': 'user', 'content': MATCHER.rewrite(rewritten, 'en')},
        ], json_output=True)
        translated = json.loads(raw).get('query_ar')
        if (isinstance(translated, str) and translated.strip() and
                re.search(r'[\u0621-\u064a]', translated) and
                len(translated) <= max(200, len(question) * 5) and translated.strip() not in variants):
            variants.append(translated.strip())
    except Exception:
        logger.warning('Arabic query translation unavailable; using original query')
    return variants


def search_variants(index, variants, vectors, min_score=.55, tolerance=2.0, media_type='video', top_k=5):
    from .video_search import hybrid_search
    results = [hybrid_search(index, query, vector, min_score, tolerance, media_type, top_k=top_k)
               for query, vector in zip(variants, vectors)]
    groups = {}
    # Keep the strongest score per file, preserving distinct scene matches.
    for result in results:
        for row in result['sources']:
            key = row['video_id']
            if key not in groups:
                groups[key] = row
                continue
            previous = groups[key]
            best = row if row['hybrid_score'] > previous['hybrid_score'] else previous
            matches = {m['id']: m for m in best['matches']}
            other = previous if best is row else row
            for match in other['matches']:
                matches.setdefault(match['id'], match)
            groups[key] = dict(best, matches=list(matches.values())[:3])
    sources = sorted(groups.values(), key=lambda r: r['hybrid_score'], reverse=True)[:top_k]
    result = next((r for r in results if r['sources']), results[0])
    return dict(result, sources=sources, no_match=not sources, query_variants=variants)
