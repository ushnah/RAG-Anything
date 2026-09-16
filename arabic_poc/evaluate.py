"""Offline extraction/retrieval metrics: python -m arabic_poc.evaluate cases.jsonl."""
import argparse
import json
from pathlib import Path
from .__main__ import normalize


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = new
    return row[-1]


def extraction_metrics(reference, hypothesis):
    if not reference.strip():
        raise ValueError('Extraction reference must not be empty')
    return {'cer': distance(reference, hypothesis) / len(reference),
            'wer': distance(reference.split(), hypothesis.split()) / len(reference.split())}


def retrieval_metrics(expected, retrieved):
    expected = set(expected)
    retrieved = list(dict.fromkeys(retrieved))
    if not expected:
        raise ValueError('Provide at least one expected passage ID')
    return {**{f'recall@{k}': len(expected.intersection(retrieved[:k])) / len(expected) for k in (5, 10)},
            'mrr': next((1 / i for i, value in enumerate(retrieved, 1) if value in expected), 0),
            'hit@10': float(bool(expected.intersection(retrieved[:10])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path)
    args = parser.parse_args()
    results = []
    for line in args.cases.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        scores = {}
        if 'reference' in case:
            scores.update(extraction_metrics(case['reference'], case['hypothesis']))
            scores.update({'normalized_' + key: value for key, value in extraction_metrics(
                normalize(case['reference']), normalize(case['hypothesis'])).items()})
        if 'expected_ids' in case:
            scores.update(retrieval_metrics(case['expected_ids'], case['retrieved_ids']))
        if not scores:
            raise ValueError('Each case needs reference/hypothesis or expected_ids/retrieved_ids')
        results.append({'id': case.get('id', len(results)), **scores})
    keys = set().union(*(set(row) - {'id'} for row in results))
    means = {key: sum(row[key] for row in results if key in row) / sum(key in row for row in results) for key in sorted(keys)}
    print(json.dumps({'cases': results, 'mean': means}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
