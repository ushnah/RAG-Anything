"""Paired hybrid/reranked evaluation using the same queries, translations and vectors."""
import argparse
import json
import time
from pathlib import Path
from .__main__ import normalize, save_json
from .query_translation import query_variants, search_variants
from .search_service import encoder, rerank
from .video_search import STORAGE


def metrics(case, result):
    rows = result['sources']
    expected = case['expected']
    if not expected:
        return {'negative_correct': not rows}
    documents = {e['document'] for e in expected}
    found = {r['document'] for r in rows}
    ranks = [i+1 for i,r in enumerate(rows) if r['document'] in documents]
    timestamps = [e for e in expected if 'start' in e]
    hits = sum(any(r['document'] == e['document'] and any(
        abs(m.get('anchor', {}).get('start', -10000) - e['start']) <= e.get('tolerance', 2)
        for m in [r, *r.get('matches', [])]) for r in rows) for e in timestamps)
    return {'recall_at_3': len(documents & found)/len(documents),
            'reciprocal_rank': 1/min(ranks) if ranks else 0,
            **({'timestamp_recall': hits/len(timestamps)} if timestamps else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('arabic_poc/validation/search_cases.json'))
    parser.add_argument('--output', type=Path, default=Path('arabic_poc/validation/search_evaluation.json'))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())['cases']
    if args.limit: cases = cases[:args.limit]
    index = json.loads((STORAGE/'index.json').read_text())
    model, name = encoder()
    if name != index['embedding_model']: raise ValueError('Embedding model mismatch')
    report = {'label_status': 'provisional: derived from saved evidence, needs human review', 'model': name, 'cases': []}
    for case in cases:
        started = time.monotonic()
        variants = query_variants(case['query'])
        vectors = model.encode([normalize(q) for q in variants], return_dense=True)['dense_vecs']
        pool = search_variants(index, variants, vectors, media_type=case['media_type'], top_k=20)
        baseline = search_variants(index, variants, vectors, media_type=case['media_type'], top_k=3)
        retrieval_seconds = time.monotonic()-started
        started = time.monotonic()
        improved = rerank(case['query'], pool, top_k=3)  # Preserve the explicit Recall@3 benchmark.
        result = dict(id=case['id'], query=case['query'], variants=variants,
                      baseline=metrics(case, baseline), reranked=metrics(case, improved),
                      baseline_results=[r['document'] for r in baseline['sources']],
                      reranked_results=[r['document'] for r in improved['sources']],
                      reranking=improved['reranking'], retrieval_seconds=retrieval_seconds,
                      rerank_seconds=time.monotonic()-started)
        report['cases'].append(result)
        save_json(args.output, report)
        print(f"{case['id']}: {result['baseline']} -> {result['reranked']}", flush=True)
    report['summary'] = {}
    for system in ['baseline', 'reranked']:
        report['summary'][system] = {metric: sum(values)/len(values) for metric in
            ['recall_at_3', 'reciprocal_rank', 'timestamp_recall', 'negative_correct']
            if (values := [r[system][metric] for r in report['cases'] if metric in r[system]])}
    save_json(args.output, report)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__': main()
