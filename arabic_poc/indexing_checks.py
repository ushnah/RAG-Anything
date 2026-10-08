"""Offline regression check for refreshing entities without changing description text."""
import contextlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from . import __main__ as pipeline, video_search, visual


class ImageRefreshChecks(unittest.TestCase):
    def test_entity_change_reembeds_existing_evidence(self):
        class OldDescription:
            def extract(self, path):
                yield 'A portrait.', {'kind': 'visual_description', 'entities': {'people': ['Old name']}}, 'mock'

        class NewDescription:
            def __init__(self, **kwargs):
                pass

            def describe(self, path):
                return {'description': 'A portrait.', 'entities': {'people': ['New name']}, 'person_matches': []}

            def description_engine(self):
                return 'mock'

        class Encoder:
            def __init__(self):
                self.inputs = []

            def encode(self, texts, **kwargs):
                self.inputs.extend(texts)
                return {'dense_vecs': np.array([[0.0, 1.0] for text in texts])}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            media = root / 'media'
            media.mkdir()
            photo = media / 'portrait.jpg'
            photo.write_bytes(b'mocked media')
            storage = root / 'storage'
            old_rows = pipeline.records(photo, OldDescription())
            pipeline.save_json(storage / 'sources' / 'original.json', {'records': old_rows})
            pipeline.save_json(storage / 'visual-index.json', {
                'embedding_model': 'mock', 'records': old_rows, 'vectors': [[1.0, 0.0]]})
            encoder = Encoder()
            with patch.object(video_search, 'STORAGE', storage), \
                    patch.object(video_search, 'INPUTS', [str(storage)]), \
                    patch.object(pipeline, 'Extractor', NewDescription), \
                    patch.object(visual, 'embedding_model', return_value=(encoder, 'mock')), \
                    patch('dotenv.load_dotenv'), \
                    patch.object(sys, 'argv', ['index_images', str(media), '--refresh']), \
                    contextlib.redirect_stdout(io.StringIO()):
                before = video_search.collect()
                pipeline.save_json(storage / 'index.json', {
                    'embedding_model': 'mock', 'records': before, 'vectors': [[1.0, 0.0]]})
                runpy.run_module('arabic_poc.index_images', run_name='__main__')
            after = json.loads((storage / 'index.json').read_text())
            self.assertEqual(after['records'][0]['id'], before[0]['id'])
            self.assertIn('New name', after['records'][0]['normalized_text'])
            self.assertEqual(after['vectors'], [[0.0, 1.0]])
            self.assertEqual(len(encoder.inputs), 2, 'Visual and combined records both need new embeddings')
            self.assertTrue(list(storage.glob('image-additions-backup-*')))


if __name__ == '__main__':
    unittest.main()
