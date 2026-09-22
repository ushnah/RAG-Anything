"""Build fresh vector indexes from an extracted corpus; activate only on request.

Run standard `ingest` first to build and verify the LightRAG graph.
"""
import argparse
import asyncio
import os
from collections import Counter
import json
from pathlib import Path

from .__main__ import save_json, runtime
from . import video_search
from .visual import embedding_model, search_text


async def ingest(storage, rows):
    from dotenv import load_dotenv
    load_dotenv()
    settings = {key: os.getenv(key, default) for key, default in (
        ('BGE_MODEL','BAAI/bge-m3'), ('RAG_TOKENIZER_MODEL','tiktoken-default'))}
    save_json(storage / 'index-settings.json', settings)
    status_file = storage / 'index' / 'kv_store_doc_status.json'
    previous = json.loads(status_file.read_text()) if status_file.exists() else {}
    rag, llm, client = await runtime(storage)
    try:
        for index, row in enumerate(rows):
            if previous.get('doc-' + row['id'], {}).get('status') == 'processed':
                continue
            print(f'Indexing {index+1}/{len(rows)} {row["document"]} {row["anchor"]}', flush=True)
            await rag.insert_content_list(
                [{'type':'text','text':f'SOURCE_ID={row["id"]}\n{row["normalized_text"]}',
                  'page_idx':row['anchor'].get('page',1)-1}],
                file_path=f'anchor:{row["id"]}',doc_id=f'doc-{row["id"]}',display_stats=False)
    finally:
        await rag.lightrag.finalize_storages()
        await client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('storage', type=Path)
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--ingest', action='store_true')
    args = parser.parse_args()
    storage = args.storage.resolve()
    rows = [row for file in sorted((storage / 'sources').glob('*.json'))
            for row in json.loads(file.read_text())['records']]
    if not rows:
        raise ValueError('No extracted records')
    complete = json.loads((storage / 'extraction_complete.json').read_text())
    if {row['source'] for row in rows} != set(complete['sources']):
        raise ValueError('Extracted corpus does not match the completed source manifest')
    if args.ingest:
        asyncio.run(ingest(storage, rows))
    statuses = json.loads((storage / 'index' / 'kv_store_doc_status.json').read_text())
    failed = [row['id'] for row in rows if statuses.get('doc-' + row['id'], {}).get('status') != 'processed']
    if failed:
        raise ValueError(f'{len(failed)} passages have not completed graph ingestion')
    visual = [dict(row, media_type='video' if 'start' in row['anchor'] else 'image',
                   source_label=row.get('metadata', {}).get('visual_label', ''))
              for row in rows if row['anchor']['kind'] == 'visual_description']
    encoder, name = embedding_model()
    if visual:
        vectors = encoder.encode([search_text(row) for row in visual], batch_size=4, return_dense=True)['dense_vecs']
        save_json(storage / 'visual-index.json', {'schema': 1, 'embedding_model': name,
                  'records': visual, 'vectors': vectors.tolist()})
    video_search.INPUTS = [str(storage)]
    combined = video_search.collect()
    vectors = encoder.encode([row['normalized_text'] for row in combined], batch_size=4, return_dense=True)['dense_vecs']
    save_json(storage / 'index.json', {'embedding_model': name, 'records': combined, 'vectors': vectors.tolist()})
    report = {'storage': str(storage), 'sources': len({r['source'] for r in rows}),
              'passages': len(rows), 'search_records': len(combined), 'graph_processed': len(rows),
              'engines': dict(Counter(r['engine'] for r in rows))}
    save_json(storage / 'refresh-report.json', report)
    if args.activate:
        relative = str(storage.relative_to(video_search.ROOT))
        save_json(video_search.ACTIVE_INDEXES, {'search_storage': relative, 'inputs': [relative],
                  'datasets': {key: relative for key in ('library','videos','visual','text','spoken','media')}})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
