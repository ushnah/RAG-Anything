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


_LOOKUP = {}
for _identity in REGISTRY:
    for _name in aliases(_identity):
        _key = name_key(_name)
        if _key in _LOOKUP and _LOOKUP[_key] != _identity:
            raise ValueError(f'Ambiguous person alias: {_name}')
        _LOOKUP[_key] = _identity


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
