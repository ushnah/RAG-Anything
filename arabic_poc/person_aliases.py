"""Explicit bilingual person aliases; never infer identity from partial given names."""
import json
import re
from pathlib import Path
from .__main__ import normalize

REGISTRY = json.loads(Path(__file__).with_suffix('.json').read_text())


def name_key(name):
    return ' '.join(re.findall(r'[^\W_]+', normalize(name).casefold()))


def aliases(identity):
    person = REGISTRY[identity]
    return list(dict.fromkeys([person['name_ar'], person['name_en'], identity, *person['aliases']]))


class AliasMatcher:
    """Token trie: longest whole-name matches; collisions remain unresolved."""
    def __init__(self, registry):
        self.registry = registry
        self.trie = {}
        self.lookup = {}
        for identity, person in registry.items():
            for name in [identity, person['name_ar'], person['name_en'], *person.get('aliases', [])]:
                key = name_key(name)
                if not key:
                    continue
                self.lookup.setdefault(key, set()).add(identity)
                node = self.trie
                for token in key.split():
                    node = node.setdefault(token, {})
                node.setdefault(None, set()).add(identity)

    def find(self, text):
        # Keep offsets into the original query while folding spelling/diacritics.
        folded, offsets = [], []
        for i, char in enumerate(text):
            for part in normalize(char).casefold():
                folded.append(part)
                offsets.append(i)
            if char.isspace():
                folded.append(' ')
                offsets.append(i)
        spans = list(re.finditer(r"[^\W_]+", ''.join(folded)))
        found, i = [], 0
        while i < len(spans):
            node, j, best = self.trie, i, None
            while j < len(spans) and spans[j].group() in node:
                node = node[spans[j].group()]
                j += 1
                if None in node:
                    best = (j, node[None])
            if best is None:
                i += 1
                continue
            end, identities = best
            found.append({'start': offsets[spans[i].start()],
                          'end': offsets[spans[end-1].end()-1]+1,
                          'id': next(iter(identities)) if len(identities) == 1 else None,
                          'ambiguous': len(identities) != 1})
            i = end
        return found

    def rewrite(self, text, language='en'):
        result = text
        for match in reversed(self.find(text)):
            if match['id'] is not None:
                name = self.registry[match['id']]['name_' + language]
                result = result[:match['start']] + name + result[match['end']:]
        return result


MATCHER = AliasMatcher(REGISTRY)
_LOOKUP = {key: next(iter(ids)) for key, ids in MATCHER.lookup.items() if len(ids) == 1}


def person_ids_in_text(text):
    return {m['id'] for m in MATCHER.find(text) if m['id'] is not None}


def canonical_query_variants(question):
    return list(dict.fromkeys([question, MATCHER.rewrite(question, 'en'), MATCHER.rewrite(question, 'ar')]))


def resolve_people(entities):
    names = entities.get('people', []) if isinstance(entities, dict) else []
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list):
        return []
    identities = dict.fromkeys(_LOOKUP[name_key(name)] for name in names
                              if isinstance(name, str) and name_key(name) in _LOOKUP)
    return [dict(id=identity, name_ar=REGISTRY[identity]['name_ar'],
                 name_en=REGISTRY[identity]['name_en'], aliases=aliases(identity)) for identity in identities]


def retrieval_aliases(entities):
    return '\n'.join(alias for person in resolve_people(entities) for alias in person['aliases'])
