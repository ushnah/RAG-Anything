"""Unified, grouped video retrieval over existing visual and speech/OCR extractions."""
import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from .__main__ import VIDEO, AUDIO, IMAGES, normalize, save_json
from .visual import embedding_model, rank

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / 'rag_storage_arabic_video_search'
INPUTS = ['rag_storage_arabic_visual', 'rag_storage_arabic_spoken', 'rag_storage_arabic_media_final', 'rag_storage_arabic_demo']
ACTIVE_INDEXES = ROOT / 'arabic_poc' / 'active_indexes.json'
if ACTIVE_INDEXES.exists():
    active = json.loads(ACTIVE_INDEXES.read_text())
    STORAGE = ROOT / active['search_storage']
    INPUTS = active['inputs']


def collect():
    rows, seen, videos = [], set(), {}
    for folder in INPUTS:
        base = ROOT / folder
        visual = base / 'visual-index.json'
        records = json.loads(visual.read_text())['records'] if visual.exists() else []
        for file in sorted((base / 'sources').glob('*.json')):
            records.extend(json.loads(file.read_text())['records'])
        for row in records:
            path = Path(row['source']).resolve()
            if not path.is_file():
                continue
            if str(path) not in videos:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                videos[str(path)] = hashlib.sha256((str(path) + digest).encode()).hexdigest()[:24]
            video_id = videos[str(path)]
            media_type = ('video' if path.suffix.lower() in VIDEO else 'audio' if path.suffix.lower() in AUDIO else 'image' if path.suffix.lower() in IMAGES else 'document')
            kind = row['anchor']['kind']
            signature = (video_id, kind, row['anchor'].get('start'), row['anchor'].get('end'), row['original_text'])
            if signature in seen:
                continue
            seen.add(signature)
            evidence_id = hashlib.sha256(json.dumps(signature, ensure_ascii=False).encode()).hexdigest()[:24]
            rows.append(dict(row, id=evidence_id, video_id=video_id, media_type=media_type,
                             evidence_type=kind, source_label=''))
            label = row.get('source_label') or row.get('metadata', {}).get('visual_label')
            label_signature = (video_id, 'metadata', label)
            if label and label_signature not in seen:
                seen.add(label_signature)
                rows.append(dict(row, id=video_id+'-metadata', video_id=video_id,
                    original_text=label, normalized_text=normalize(label), source_label='',
                    anchor={'kind': 'metadata'}, evidence_type='metadata', engine='publisher-metadata',
                    media_type=media_type))
    return rows


def group_matches(matches, top_k=3):
    groups = {}
    for row in matches:
        key = row['video_id']
        if key not in groups:
            groups[key] = dict(row, matches=[])
        # One card per file; retain the best three pieces of evidence.
        if len(groups[key]['matches']) < 3:
            groups[key]['matches'].append({k: row[k] for k in
                ('id', 'anchor', 'original_text', 'evidence_type', 'similarity')})
    return list(groups.values())[:top_k]


# Match whole tokens, folding Arabic spelling and the definite article on
# words of at least five letters; do not match arbitrary substrings.
STOP = set(normalize('show me video videos of a the about find please أرني اعرض فيديو فيديوهات عن في من لي').split())


def tokens(text):
    words = [t for t in re.findall(r"[^\W_]+", normalize(text).lower()) if t not in STOP]
    words = [t[2:] if t.startswith("ال") and len(t) >= 5 else t for t in words]
    # Small, explicit variants for high-value Arabic landmark queries. This is
    # not a stemmer: it avoids turning unrelated words into a match.
    variants = {'بوابة': 'باب', 'بوابه': 'باب', 'منارة': 'ماذن', 'منارات': 'ماذن',
                'مئذنة': 'ماذن', 'مئذنتان': 'ماذن', 'مئذنتين': 'ماذن'}
    return [variants.get(word, word) for word in words]


def aligned_evidence(seed, rows, tolerance=2.0):
    """Frames are points, speech is an interval. Never extend a frame to next sample."""
    if seed['evidence_type'] not in ('speech', 'frame', 'visual_description'):
        return []
    anchor = seed['anchor']
    if 'start' not in anchor:
        return []
    result = []
    for other in rows:
        if other['video_id'] != seed['video_id'] or other['id'] == seed['id']:
            continue
        if (seed['evidence_type'] == 'speech') == (other['evidence_type'] == 'speech'):
            continue
        if other['evidence_type'] not in ('speech', 'frame', 'visual_description') or 'start' not in other['anchor']:
            continue
        speech, frame = (seed, other) if seed['evidence_type'] == 'speech' else (other, seed)
        start = speech['anchor']['start']
        end = speech['anchor'].get('end', start)
        point = frame['anchor']['start']
        gap = max(start-point, point-end, 0)
        if gap <= tolerance:
            result.append({k: other[k] for k in ('id', 'anchor', 'original_text', 'evidence_type')} |
                          {'relation': 'overlap' if gap == 0 else 'nearby', 'gap_seconds': gap})
    return result


def hybrid_search(index, question, vector, min_score=.55, tolerance=2.0, media_type='video'):
    query = set(tokens(question))
    documents = [tokens(r['original_text']) for r in index['records']]
    n = len(documents)
    average = sum(map(len, documents))/max(n, 1)
    df = Counter(t for doc in documents for t in set(doc))
    by_id = {r['id']: doc for r, doc in zip(index['records'], documents)}
    ranked = rank(index, vector, media_type, n)
    accepted = []
    for row in ranked:
        doc = by_id[row['id']]
        counts = Counter(doc)
        bm25 = 0.0
        for term in query:
            freq = counts[term]
            if freq:
                idf = math.log(1+(n-df[term]+.5)/(df[term]+.5))
                bm25 += idf*freq*2.2/(freq+1.2*(.25+.75*len(doc)/max(average, 1)))
        coverage = len(query.intersection(counts))/max(len(query), 1)
        # Named landmarks often share broad visual terms (for example, gate or
        # minaret). Reward precise Arabic term coverage so a complete name is
        # preferred over a semantically similar but differently named place.
        score = (.60*max(0, row['similarity']) + .25*(bm25/(bm25+2)) + .15*coverage)
        # Fixed, explicit policy; not a calibrated probability. Strong lexical
        # matches may pass only with semantic support as well.
        semantic_floor = max(min_score, .7) if row['evidence_type'] == 'speech' and len(doc) < 5 and coverage == 0 else min_score
        qualifies = (row['similarity'] >= semantic_floor or
                     (coverage >= .55 and row['similarity'] >= .45) or
                     (coverage == 1 and bool(query) and row['similarity'] >= .35))
        if qualifies:
            row.update(hybrid_score=round(score, 4), bm25=round(bm25, 4), keyword_coverage=coverage)
            accepted.append(row)
    accepted.sort(key=lambda r: r['hybrid_score'], reverse=True)
    groups = group_matches(accepted)
    by_row = {r['id']: r for r in index['records']}
    for group in groups:
        for match in group['matches']:
            match['aligned'] = aligned_evidence(by_row[match['id']], index['records'], tolerance)
    return {'answer': ('مصادر مرشحة. راجع الأدلة وتوقيتاتها.' if groups else
                       'لم أجد أدلة قوية كافية لهذا البحث. جرّب صياغة أخرى.'),
            'sources': groups, 'citations': [], 'retrieval': 'video_candidates' if media_type == 'video' else 'library_candidates',
            'no_match': not groups,
            'policy': {'min_semantic_score': min_score, 'nearby_seconds': tolerance,
                       'note': 'Heuristic threshold, not calibrated confidence; temporal proximity does not prove semantic agreement.'}}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('build')
    search = sub.add_parser('search')
    search.add_argument('question')
    search.add_argument('--media-type', choices=['all','video','audio','image','document'], default='video')
    search.add_argument('--min-score', type=float, default=float(os.getenv('VIDEO_MIN_SCORE', '.55')))
    search.add_argument('--nearby-seconds', type=float, default=2.0)
    args = parser.parse_args()
    target = STORAGE / 'index.json'
    encoder, name = embedding_model()
    if args.action == 'build':
        rows = collect()
        if not rows:
            raise ValueError('No extracted video evidence found.')
        vectors = encoder.encode([normalize(r['original_text']) for r in rows], batch_size=4,
                                 return_dense=True)['dense_vecs']
        save_json(target, {'embedding_model': name, 'records': rows, 'vectors': vectors.tolist()})
        print(f'Indexed {len(rows)} evidence records from {len({r["video_id"] for r in rows})} sources.')
        return
    index = json.loads(target.read_text())
    if name != index['embedding_model']:
        raise ValueError('Embedding model changed; rebuild unified video search.')
    vector = encoder.encode([normalize(args.question)], return_dense=True)['dense_vecs'][0]
    if not 0 <= args.min_score <= 1 or not math.isfinite(args.nearby_seconds) or args.nearby_seconds < 0:
        parser.error('min-score must be 0..1 and nearby-seconds finite and nonnegative')
    print(json.dumps(hybrid_search(index, args.question, vector, args.min_score, args.nearby_seconds, args.media_type), ensure_ascii=False))


if __name__ == '__main__':
    main()
