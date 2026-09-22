import unittest
from unittest.mock import patch
from .remote import RemoteASR, chat, enabled

class RemoteChecks(unittest.TestCase):
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
