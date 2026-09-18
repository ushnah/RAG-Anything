"""Visual search over Qwen frame descriptions and separately attributed source labels.

python -m arabic_poc.visual build data/visual_samples
python -m arabic_poc.visual search 'أرني صورة باب الملك عبد العزيز' --media-type image
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .__main__ import IMAGES, VIDEO, command, normalize, save_json


def build(paths, storage, model, interval):
    from .models import QwenParser
    vision = QwenParser(model, describe=True)
    files = sorted({p.resolve() for root in paths for p in
                    (root.rglob('*') if root.is_dir() else [root])
                    if p.suffix.lower() in IMAGES | VIDEO})
    if not files:
        raise ValueError('No images or videos found.')
    rows = []
    for path in files:
        sidecar = path.with_name(path.name + '.metadata.json')
        metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source_id = hashlib.sha256((str(path) + digest).encode()).hexdigest()[:24]
        with tempfile.TemporaryDirectory() as folder:
            frames = [(path, None)]
            if path.suffix.lower() in VIDEO:
                duration = float(json.loads(command('ffprobe', '-v', 'error', '-show_format',
                    '-of', 'json', str(path)))['format']['duration'])
                command('ffmpeg', '-v', 'error', '-i', str(path), '-vf',
                    f'fps=1/{interval}:start_time=0:round=up,scale=640:-2', str(Path(folder) / '%06d.jpg'))
                frames = [(frame, i * interval) for i, frame in enumerate(sorted(Path(folder).glob('*.jpg')))
                          if i * interval < duration]
            for i, (frame, start) in enumerate(frames):
                print(f'Describing {path.name}: {start}', flush=True)
                description = vision.parse_image(frame)[0]['text']
                anchor = {'kind': 'visual_description'}
                if start is not None:
                    anchor['start'] = start
                rows.append(dict(id=f'{source_id}-{i:06d}', source=str(path), document=path.name,
                    original_text=description, normalized_text=normalize(description), anchor=anchor,
                    engine=f'qwen-vl:generated:{model}', confidence=None, metadata=metadata,
                    source_label=metadata.get('visual_label', ''),
                    media_type='video' if start is not None else 'image'))
    # Publish only after extraction and embedding both succeed.
    import gc
    import torch
    del vision
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import numpy as np
    encoder, model_id = embedding_model()
    vectors = encoder.encode([search_text(row) for row in rows], batch_size=4,
                             return_dense=True)['dense_vecs']
    save_json(storage / 'visual-index.json', {'schema': 1, 'embedding_model': model_id,
        'frame_seconds': interval, 'description_prompt_version': 2, 'records': rows, 'vectors': np.asarray(vectors).tolist()})


def embedding_model():
    from dotenv import load_dotenv
    from FlagEmbedding import BGEM3FlagModel
    load_dotenv()
    name = os.getenv('BGE_MODEL', 'BAAI/bge-m3')
    return BGEM3FlagModel(name, use_fp16=False), name


def search_text(row):
    # Labels describe the whole asset, not proof of its presence in every frame.
    return normalize(row['original_text'] + '\nSource label: ' + row.get('source_label', ''))


def rank(index, vector, media_type, top_k):
    import numpy as np
    query = np.asarray(vector)
    matches = []
    for row, stored in zip(index['records'], index['vectors']):
        if media_type != 'all' and row['media_type'] != media_type:
            continue
        embedding = np.asarray(stored)
        score = float(embedding @ query / max(float(np.linalg.norm(embedding) * np.linalg.norm(query)), 1e-12))
        matches.append(dict(row, similarity=round(score, 4)))
    # These are ranked candidates, never a claim of verified landmark identity.
    return sorted(matches, key=lambda row: row['similarity'], reverse=True)[:top_k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', type=Path, default=Path('rag_storage_arabic_visual'))
    commands = parser.add_subparsers(dest='action', required=True)
    build_args = commands.add_parser('build')
    build_args.add_argument('paths', nargs='+', type=Path)
    build_args.add_argument('--model', default='models/qwen2.5-vl-3b')
    build_args.add_argument('--frame-seconds', type=float, default=10)
    query = commands.add_parser('search')
    query.add_argument('question')
    query.add_argument('--media-type', choices=['all', 'image', 'video'], default='all')
    query.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()
    if args.action == 'build':
        import math
        if not math.isfinite(args.frame_seconds) or args.frame_seconds <= 0:
            parser.error('frame-seconds must be finite and positive')
        build(args.paths, args.storage, args.model, args.frame_seconds)
    else:
        if args.top_k <= 0:
            parser.error('top-k must be positive')
        index = json.loads((args.storage / 'visual-index.json').read_text())
        encoder, name = embedding_model()
        if index['embedding_model'] != name:
            raise ValueError('Embedding model changed; rebuild the visual index.')
        vector = encoder.encode([normalize(args.question)], return_dense=True)['dense_vecs'][0]
        sources = rank(index, vector, args.media_type, args.top_k)
        print(json.dumps({'answer': 'نتائج مرشحة حسب التشابه. راجع الصورة أو لقطة الفيديو للتحقق من المكان.',
                          'citations': [], 'sources': sources, 'retrieval': 'visual_candidates'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
