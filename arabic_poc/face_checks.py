import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .face_identifier import FaceIdentifier
from .sheikh_poc import describe_with_matches, refresh_saved_results
from .__main__ import Extractor, records


class FakeFace:
    def __init__(self, embedding, bbox=(0, 0, 100, 100), det_score=0.99):
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.det_score = det_score


class FakeFaceApp:
    def __init__(self, faces):
        self.faces = faces

    def get(self, image):
        return self.faces


class FaceIdentifierChecks(unittest.TestCase):
    def test_known_and_unknown_matching(self):
        identifier = FaceIdentifier.__new__(FaceIdentifier)
        identifier.threshold = 0.8
        identifier.gallery = [('sheikh_x', FaceIdentifier.normalize_embedding([1, 0]))]
        self.assertEqual(identifier.identify_face([1, 0])['name'], 'sheikh_x')
        unknown = identifier.identify_face([0, 1])
        self.assertIsNone(unknown['name'])
        self.assertEqual(unknown['source'], 'face_recognition')

    def test_empty_gallery_and_multiple_reference_images(self):
        identifier = FaceIdentifier.__new__(FaceIdentifier)
        identifier.threshold = 0.8
        identifier.gallery = []
        self.assertIsNone(identifier.identify_face([1, 0])['name'])
        identifier.gallery = [
            ('sheikh_x', FaceIdentifier.normalize_embedding([1, 0])),
            ('sheikh_x', FaceIdentifier.normalize_embedding([0.9, 0.1])),
        ]
        self.assertEqual(identifier.identify_face([0.9, 0.1])['name'], 'sheikh_x')

    def test_gallery_skips_invalid_and_multiple_face_images(self):
        with tempfile.TemporaryDirectory() as folder:
            gallery = Path(folder)
            (gallery / 'sheikh_x').mkdir()
            (gallery / 'sheikh_y').mkdir()
            for path in (gallery / 'sheikh_x' / 'valid.jpg', gallery / 'sheikh_y' / 'multiple.jpg'):
                path.write_bytes(b'image')
            app = FakeFaceApp([FakeFace([1, 0]), FakeFace([1, 0])])
            identifier = FaceIdentifier(gallery, app=app)
            with patch('cv2.imread', return_value=object()):
                loaded = identifier.build_gallery()
            self.assertEqual(len(loaded), 0)

    def test_multiple_faces_in_frame_and_quality_filter(self):
        identifier = FaceIdentifier.__new__(FaceIdentifier)
        identifier.enabled = True
        identifier.app = FakeFaceApp([FakeFace([1, 0]), FakeFace([0, 1])])
        identifier.gallery = [('sheikh_x', FaceIdentifier.normalize_embedding([1, 0]))]
        identifier.threshold = 0.8
        identifier.min_face_size = 40
        identifier.min_detection_score = 0.6
        results = identifier.identify_faces(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual([item['name'] for item in results], ['sheikh_x', None])

    def test_extractor_preserves_vlm_entities_and_indexes_face_name(self):
        class FakeExtractor(Extractor):
            def extract(self, path):
                yield 'المشهد: رجل', {'kind': 'visual_description', 'start': 20,
                                      'entities': {'landmarks': ['الكعبة']},
                                      'person_matches': [{'name': 'Sheikh X', 'similarity': 0.91,
                                                          'source': 'face_recognition'}]}, 'remote-vision:generated:test'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'frame.jpg'
            path.write_bytes(b'dummy')
            row = records(path, FakeExtractor())[0]
            self.assertEqual(row['entities']['people'], [])
            self.assertEqual(row['anchor']['person_matches'][0]['name'], 'Sheikh X')
            self.assertIn('Sheikh X', row['normalized_text'])
            self.assertEqual(row['anchor']['start'], 20)

    def test_face_backend_failure_returns_empty_matches(self):
        with patch('arabic_poc.face_identifier.FaceIdentifier', side_effect=RuntimeError('model unavailable')):
            extractor = Extractor(face_gallery_path='reference_faces')
            self.assertEqual(extractor.identify_faces(Path('missing.jpg')), [])

    def test_matched_people_are_passed_to_vision_and_returned_cleanly(self):
        class FakeIdentifier:
            def identify_faces(self, image_path):
                return [
                    {'name': 'abdul_rahman_al_sudais', 'similarity': 0.91, 'source': 'face_recognition'},
                    {'name': 'mohammed_bin_salman', 'similarity': 0.88, 'source': 'face_recognition'},
                    {'name': None, 'similarity': 0.42, 'source': 'face_recognition'},
                ]

        class FakeVision:
            def parse_image(self, image_path, person_context=None):
                self.person_context = person_context
                return [{'description': 'وصف', 'entities': {'people': ['visible_name'], 'landmarks': ['الكعبة']}}]

        vision = FakeVision()
        result = describe_with_matches(Path('frame.jpg'), vision, FakeIdentifier())
        self.assertEqual(result['people'], {
            'people': ['abdul_rahman_al_sudais', 'mohammed_bin_salman']
        }['people'])
        self.assertEqual(vision.person_context['source'], 'face_recognition')
        self.assertEqual(vision.person_context['people'], result['people'])
        self.assertEqual(result['entities']['people'], [
            'abdul_rahman_al_sudais', 'mohammed_bin_salman', 'visible_name'
        ])
        self.assertEqual(result['entities']['landmarks'], ['الكعبة'])

    def test_extractor_passes_face_matches_before_visual_description(self):
        class FakeIdentifier:
            def identify_faces(self, image_path):
                return [{'name': 'mohammed_bin_salman', 'similarity': 0.9,
                         'source': 'face_recognition'}]

        class FakeVision:
            def parse_image(self, image_path, person_context=None):
                self.person_context = person_context
                return [{'text': 'وصف', 'description': 'وصف', 'entities': {'people': []}}]

        extractor = Extractor(describe_images=True)
        extractor.face_identifier = FakeIdentifier()
        extractor.face_identifier_initialized = True
        extractor.vision = FakeVision()
        result = extractor.describe(Path('frame.jpg'))
        self.assertEqual(extractor.vision.person_context['people'], ['mohammed_bin_salman'])
        self.assertEqual(result['entities']['people'], ['mohammed_bin_salman'])
        self.assertEqual(result['person_matches'][0]['source'], 'face_recognition')

    def test_refresh_saved_results_adds_people_without_changing_description(self):
        class FakeIdentifier:
            def identify_faces(self, frame):
                return [{'name': 'mohammed_bin_salman', 'similarity': 0.93,
                         'source': 'face_recognition'}]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'visual.json'
            original = {
                'people': [],
                'results': [{
                    'frame': '/retained/frames/frame-000001.jpg',
                    'timestamp_seconds': 20,
                    'description': 'Existing VLM description',
                    'entities': {'people': [], 'landmarks': ['الكعبة']},
                }],
            }
            path.write_text(json.dumps(original), encoding='utf-8')
            refreshed = refresh_saved_results(path, FakeIdentifier())
            row = refreshed['results'][0]
            self.assertEqual(row['description'], 'Existing VLM description')
            self.assertEqual(row['timestamp_seconds'], 20)
            self.assertEqual(row['entities']['people'], ['mohammed_bin_salman'])
            self.assertEqual(row['entities']['landmarks'], ['الكعبة'])
            self.assertEqual(row['person_matches'][0]['source'], 'face_recognition')
            self.assertEqual(refreshed['people'], ['mohammed_bin_salman'])


if __name__ == '__main__':
    unittest.main()
