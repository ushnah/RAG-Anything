"""Offline pipeline checks; identification and all model calls are mocked.

Run: python -m unittest arabic_poc.pipeline_isolation_checks -v
No reference photos, embeddings, model downloads, or network calls are used.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from . import __main__ as pipeline


class PipelineIsolationChecks(unittest.TestCase):
    def setUp(self):
        environment = patch.dict('os.environ', {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.extractor = pipeline.Extractor(describe_images=True, frame_seconds=10)
        self.visual = {
            'text': 'A doorway with a sign.',
            'description': 'A doorway with a sign.',
            'entities': {'people': [], 'landmarks': [], 'places': [],
                         'organizations': [], 'events': []},
        }
        self.extractor.ocr = Mock()
        self.extractor.ocr.parse_image.return_value = [{'text': 'النص الأصلي', 'page_idx': 0}]
        self.extractor.vision = Mock()
        self.extractor.vision.parse_image.return_value = [copy.deepcopy(self.visual)]
        self.identifier = Mock()
        self.identifier.identify_faces.return_value = []
        self.extractor.face_identifier = self.identifier
        self.extractor.face_identifier_initialized = True

    def test_empty_result_preserves_image_ocr_description_and_entities(self):
        path = Path('synthetic.png')
        rows = list(self.extractor.extract(path))
        self.assertEqual(rows[0], ('النص الأصلي', {'kind': 'page', 'page': 1}, 'paddleocr:ar'))
        self.assertEqual(rows[1], (self.visual['description'], {
            'kind': 'visual_description', 'entities': self.visual['entities'],
            'person_matches': []}, 'qwen-vl:generated'))
        self.assertEqual(self.extractor.vision.parse_image.return_value, [self.visual])

    def test_video_records_preserve_timestamps_text_and_source_metadata(self):
        self.extractor.asr = Mock()
        self.extractor.asr.transcribe.return_value = {'segments': [
            {'text': 'كلام مسموع', 'start': 1.5, 'end': 4.5}]}

        def fake_command(*args):
            if args[0] == 'ffprobe':
                return json.dumps({'streams': [{'codec_type': 'video'}, {'codec_type': 'audio'}],
                                   'format': {'duration': '20'}})
            for number in (1, 2):
                Path(args[-1].replace('%06d', f'{number:06d}')).write_bytes(b'mocked frame')
            return ''

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.mp4'
            path.write_bytes(b'mocked video')
            metadata = {'title': 'Synthetic fixture', 'source_url': 'https://example.test/video'}
            pipeline.save_json(path.with_name(path.name + '.metadata.json'), metadata)
            with patch.object(pipeline.shutil, 'which', return_value='/mock/tool'), \
                    patch.object(pipeline, 'command', side_effect=fake_command) as command:
                rows = pipeline.records(path, self.extractor)
            self.assertEqual(command.call_count, 2)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]['original_text'], 'كلام مسموع')
        self.assertEqual((rows[0]['anchor']['start'], rows[0]['anchor']['end']), (1.5, 4.5))
        self.assertEqual([row['anchor']['start'] for row in rows[1:]], [0, 0, 10, 10])
        self.assertEqual([row['anchor']['kind'] for row in rows[1:]],
                         ['frame', 'visual_description', 'frame', 'visual_description'])
        for row in rows:
            self.assertEqual(row['metadata'], metadata)
            self.assertEqual(row['source'], str(path.resolve()))
            self.assertEqual(row['anchor']['char_start'], 0)
            self.assertEqual(row['anchor']['char_end'], len(row['original_text']))
        for row in (rows[1], rows[3]):
            self.assertEqual(row['original_text'], 'النص الأصلي')
        for row in (rows[2], rows[4]):
            self.assertEqual(row['original_text'], self.visual['description'])
            self.assertEqual(row['entities'], self.visual['entities'])
            self.assertEqual(row['anchor']['person_matches'], [])

    def test_initialization_failure_preserves_pipeline_output(self):
        self.extractor.face_identifier = None
        self.extractor.face_identifier_initialized = False
        self.extractor.face_gallery_path = '/unused/mocked-gallery'
        module = ModuleType('arabic_poc.face_identifier')
        module.FaceIdentifier = Mock(side_effect=RuntimeError('synthetic initialization failure'))
        with patch.dict('sys.modules', {'arabic_poc.face_identifier': module}), \
                patch('builtins.print'):
            first = list(self.extractor.extract(Path('synthetic.png')))
            second = list(self.extractor.extract(Path('synthetic.png')))
        self.assertEqual(first, second)
        self.assertEqual(first[0][0], 'النص الأصلي')
        self.assertEqual(first[1][0], self.visual['description'])
        self.assertEqual(first[1][1]['person_matches'], [])
        module.FaceIdentifier.assert_called_once()

    @unittest.expectedFailure
    def test_escaping_runtime_error_should_preserve_pipeline_output(self):
        # Known caller-boundary gap: an exception escaping the mocked component
        # aborts extraction. This does not exercise recognition implementation.
        self.identifier.identify_faces.side_effect = RuntimeError('synthetic component failure')
        rows = list(self.extractor.extract(Path('synthetic.png')))
        self.assertEqual(rows[0][0], 'النص الأصلي')
        self.assertEqual(rows[1][0], self.visual['description'])
        self.assertEqual(rows[1][1]['person_matches'], [])


class IngestionBoundaryChecks(unittest.IsolatedAsyncioTestCase):
    async def test_empty_result_reaches_rag_with_source_link_and_saved_metadata(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {}, clear=True):
            root = Path(folder)
            path = root / 'synthetic.png'
            path.write_bytes(b'mocked image')
            pipeline.save_json(path.with_name(path.name + '.metadata.json'), {'title': 'Fixture'})
            extractor = Mock()
            extractor.extract.return_value = iter([
                ('النص الأصلي', {'kind': 'page', 'page': 1}, 'mock:ocr'),
                ('A doorway.', {'kind': 'visual_description', 'start': 10,
                                'entities': {}, 'person_matches': []}, 'mock:vision')])
            rag = SimpleNamespace(insert_content_list=AsyncMock(),
                                  lightrag=SimpleNamespace(finalize_storages=AsyncMock()))
            client = SimpleNamespace(close=AsyncMock())
            args = SimpleNamespace(action='ingest', storage=str(root / 'storage'), paths=[str(path)],
                                   asr_model='mock', frame_seconds=10, ocr_engine='mock',
                                   qwen_vl_model='mock', describe_images=True, speech_only=False)
            with patch.object(pipeline, 'Extractor', return_value=extractor), \
                    patch.object(pipeline, 'runtime', new=AsyncMock(return_value=(rag, None, client))), \
                    patch('dotenv.load_dotenv'), patch('builtins.print'):
                await pipeline.run(args)
            saved = list((root / 'storage' / 'sources').glob('*.json'))
            self.assertEqual(len(saved), 1)
            rows = json.loads(saved[0].read_text())['records']
            self.assertEqual(len(rows), 2)
            self.assertEqual(rag.insert_content_list.await_count, 2)
            for row, call in zip(rows, rag.insert_content_list.await_args_list):
                self.assertEqual(call.args[0], [{'type': 'text',
                    'text': f'SOURCE_ID={row["id"]}\n{row["normalized_text"]}', 'page_idx': 0}])
                self.assertEqual(call.kwargs['file_path'], f'anchor:{row["id"]}')
                self.assertEqual(call.kwargs['doc_id'], f'doc-{row["id"]}')
                self.assertEqual(row['metadata'], {'title': 'Fixture'})
            self.assertEqual(rows[1]['anchor']['person_matches'], [])
            self.assertEqual(rows[1]['anchor']['start'], 10)
            rag.lightrag.finalize_storages.assert_awaited_once()
            client.close.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
