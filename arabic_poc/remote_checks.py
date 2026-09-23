import unittest
from unittest.mock import patch
from .remote import RemoteASR, RemoteVision, chat, enabled, parse_visual_response

class RemoteChecks(unittest.TestCase):
    def test_structured_visual_response(self):
        description, entities = parse_visual_response('{"description":"المشهد: مسجد", "entities":{"people":["شخص"],"landmarks":["المسجد النبوي"],"places":["المدينة المنورة"],"organizations":["جامعة"],"events":["مؤتمر"]}}')
        self.assertEqual(description, 'المشهد: مسجد')
        self.assertEqual(entities['landmarks'], ['المسجد النبوي'])
        self.assertEqual(set(entities), {'people', 'landmarks', 'places', 'organizations', 'events'})

    def test_empty_and_missing_entity_fields(self):
        _, entities = parse_visual_response('{"description":"وصف", "entities":{"landmarks":[]}}')
        self.assertEqual(entities, {'people': [], 'landmarks': [], 'places': [], 'organizations': [], 'events': []})

    def test_malformed_json_preserves_description(self):
        description, entities = parse_visual_response('وصف عربي قديم')
        self.assertEqual(description, 'وصف عربي قديم')
        self.assertEqual(entities['people'], [])

    def test_remote_description_returns_entities(self):
        from tempfile import TemporaryDirectory
        from PIL import Image
        with TemporaryDirectory() as folder, patch.dict('os.environ', {'REMOTE_VISION_MODEL': 'vision-test'}):
            path = f'{folder}/image.jpg'
            Image.new('RGB', (8, 8), 'white').save(path)
            response = '{"description":"المشهد: قبة", "entities":{"landmarks":["قبة خضراء"]}}'
            with patch('arabic_poc.remote.chat', return_value=response) as request:
                result = RemoteVision(describe=True).parse_image(path)[0]
            self.assertEqual(result['text'], 'المشهد: قبة')
            self.assertEqual(result['entities']['landmarks'], ['قبة خضراء'])
            self.assertTrue(request.call_args.kwargs['json_output'])

    def test_remote_description_failure_falls_back_without_entities(self):
        from tempfile import TemporaryDirectory
        from PIL import Image
        with TemporaryDirectory() as folder, patch.dict('os.environ', {'REMOTE_VISION_MODEL': 'vision-test'}):
            path = f'{folder}/image.jpg'
            Image.new('RGB', (8, 8), 'white').save(path)
            with patch('arabic_poc.remote.chat', side_effect=[RuntimeError('unavailable'), 'وصف احتياطي']) as request:
                result = RemoteVision(describe=True).parse_image(path)[0]
            self.assertEqual(result['text'], 'وصف احتياطي')
            self.assertEqual(result['entities'], {'people': [], 'landmarks': [], 'places': [], 'organizations': [], 'events': []})
            self.assertFalse(request.call_args.kwargs['json_output'])

    def test_remote_ocr_remains_text_only(self):
        from tempfile import TemporaryDirectory
        from PIL import Image
        with TemporaryDirectory() as folder, patch.dict('os.environ', {'REMOTE_VISION_MODEL': 'vision-test'}):
            path = f'{folder}/image.jpg'
            Image.new('RGB', (8, 8), 'white').save(path)
            with patch('arabic_poc.remote.chat', return_value='{"text":"النص الأصلي"}'):
                result = RemoteVision().parse_image(path)[0]
            self.assertEqual(result, {'text': 'النص الأصلي', 'page_idx': 0})
    def test_roles_are_independent(self):
        with patch.dict('os.environ', {'MODEL_BACKEND':'remote','OCR_BACKEND':'local','VISION_BACKEND':'remote'}):
            self.assertFalse(enabled('ocr'))
            self.assertTrue(enabled('vision'))

    def test_generation_cap_is_not_silent(self):
        with patch('arabic_poc.remote.request', return_value={'choices':[{'finish_reason':'length','message':{'content':'partial'}}]}):
            with self.assertRaisesRegex(RuntimeError,'token limit'): chat([])

    def test_asr_requires_timestamps(self):
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile() as file, patch('arabic_poc.remote.request',return_value={'text':'hello'}):
            with self.assertRaisesRegex(RuntimeError,'timestamps'):RemoteASR().transcribe(file.name)

if __name__ == '__main__': unittest.main()
