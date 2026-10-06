"""Resolve person references with an LLM; apply only validated name-span edits."""
from .prompt_loader import get_prompt
import json
import logging
import re
from difflib import SequenceMatcher
from .person_aliases import REGISTRY, MATCHER, name_key
from .remote import chat, enabled


def candidates(question, limit=30):
    """Bound the prompt; known matches first, then lexical name/alias similarity."""
    query = name_key(question)
    words = query.split()
    known = {m['id'] for m in MATCHER.find(question) if m['id']}
    ranked = []
    for identity, person in REGISTRY.items():
        names = [person['name_en'], person['name_ar'], identity, *person.get('aliases', [])]
        score = max((SequenceMatcher(None, query, name_key(n)).ratio() for n in names), default=0)
        # Acronym support without maintaining a separate nickname table.
        initials = ''.join(w[0] for w in name_key(person['name_en']).split() if w)
        score += int(initials in words)
        ranked.append((identity in known, score, identity))
    return [dict(id=identity, name_ar=REGISTRY[identity]['name_ar'], name_en=REGISTRY[identity]['name_en'])
            for _, _, identity in sorted(ranked, reverse=True)[:limit]]


def rewrite_person_query(question):
    if not enabled('text') or any(m['ambiguous'] for m in MATCHER.find(question)):
        return question
    people = candidates(question)
    if not people:
        return question
    try:
        response = json.loads(chat([
            {'role': 'system', 'content': get_prompt('query.person_resolution')},
            {'role': 'user', 'content': json.dumps({'query': question, 'people': people}, ensure_ascii=False)},
        ], json_output=True))
        resolutions = response['resolutions']
        if not isinstance(resolutions, list) or len(resolutions) > 10:
            return question
        allowed = {p['id']: p for p in people}
        edits = []
        for item in resolutions:
            if not isinstance(item, dict) or item.get('status') != 'resolved':
                continue
            mention, identity = item.get('mention'), item.get('person_id')
            if not isinstance(mention, str) or not mention.strip() or identity not in allowed:
                return question
            protected = {'show', 'find', 'picture', 'pictures', 'photo', 'photos', 'video', 'videos',
                         'not', 'without', 'except', 'only', 'ارني', 'اعرض', 'صورة', 'صور',
                         'فيديو', 'بدون', 'ليس', 'الا', 'فقط'}
            if set(name_key(mention).split()) & protected or any(c.isdigit() for c in mention):
                return question
            # Require a literal, whole-token span. Never execute free-form rewritten prose.
            matches = list(re.finditer(r'(?<!\w)' + re.escape(mention) + r'(?!\w)', question))
            if not matches:
                return question
            for match in matches:
                edits.append((match.start(), match.end(), allowed[identity]['name_en']))
        edits = sorted(set(edits))
        if any(a[1] > b[0] for a, b in zip(edits, edits[1:])):
            return question
        rewritten = question
        for start, end, name in reversed(edits):
            rewritten = rewritten[:start] + name + rewritten[end:]
        return rewritten
    except Exception:
        logging.getLogger(__name__).warning('Person query resolution unavailable; retaining original query')
        return question
