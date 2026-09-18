"""Evaluate sampled-frame retrieval with and without publisher labels."""
import argparse
import json
from pathlib import Path
from .__main__ import normalize, save_json
from .visual import embedding_model, rank


def scores(expected, retrieved):
    expected = set(expected)
    if not expected:
        return {'returned_candidates_without_verified_support': bool(retrieved)}
    return {'hit@1': int(bool(expected.intersection(retrieved[:1]))),
            'hit@3': int(bool(expected.intersection(retrieved[:3]))),
            'recall@3': len(expected.intersection(retrieved[:3])) / len(expected),
            'mrr': next((1/i for i, key in enumerate(retrieved, 1) if key in expected), 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', type=Path, default=Path('rag_storage_arabic_visual_eval'))
    parser.add_argument('--labels', type=Path, default=Path('arabic_poc/validation/visual_eval_labels.json'))
    parser.add_argument('--output', type=Path, default=Path('arabic_poc/validation/visual_eval_retrieval.json'))
    args = parser.parse_args()
    index = json.loads((args.storage / 'visual-index.json').read_text())
    labels = json.loads(args.labels.read_text())
    encoder, model = embedding_model()
    if model != index['embedding_model']:
        raise ValueError('Embedding model does not match evaluation index.')
    annotations = {(r['document'], r['second']): r for r in labels['frames']}
    for row in index['records']:
        if (row['document'], row['anchor']['start']) not in annotations:
            raise ValueError('Missing frame annotation: ' + row['id'])
    vectors = encoder.encode([normalize(q['question']) for q in labels['queries']], return_dense=True)['dense_vecs']
    results = {}
    for mode in ('with_source_labels', 'descriptions_only'):
        variant = dict(index)
        if mode == 'descriptions_only':
            variant['vectors'] = encoder.encode([normalize(r['original_text']) for r in index['records']],
                                                batch_size=4, return_dense=True)['dense_vecs'].tolist()
        cases = []
        for query, vector in zip(labels['queries'], vectors):
            expected = [r['id'] for r in index['records'] if query['tag'] in
                        annotations[(r['document'], r['anchor']['start'])]['visible_objects']]
            retrieved = rank(variant, vector, 'video', len(index['records']))
            ids = [r['id'] for r in retrieved]
            cases.append(dict(query, expected_ids=expected, **scores(expected, ids),
                top3=[{'id': r['id'], 'document': r['document'], 'second': r['anchor']['start'],
                       'similarity': r['similarity'], 'description': r['original_text']} for r in retrieved[:3]]))
        positives = [c for c in cases if c['expected_ids']]
        results[mode] = {'cases': cases, 'positive_query_count': len(positives),
            'mean': {key: sum(c[key] for c in positives)/len(positives)
                     for key in ('hit@1', 'hit@3', 'recall@3', 'mrr')} if positives else {}}
    save_json(args.output, {'frame_count': len(index['records']), 'embedding_model': model,
        'caveat': 'Provisional assistant annotations; same short corpus, no held-out tuning. Negative queries test lack of abstention, not identity recognition.',
        'results': results})
    print(json.dumps({mode: result['mean'] for mode, result in results.items()}, indent=2))


if __name__ == '__main__':
    main()
