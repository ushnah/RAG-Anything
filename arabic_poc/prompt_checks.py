import json
import unittest
from .prompt_loader import get_prompt
from .remote import person_context_prompt


class PromptChecks(unittest.TestCase):
    def test_context_payload_is_not_interpreted_as_template(self):
        value = 'اسم ${payload} {"name":"test"}'
        result = person_context_prompt({'people': [value]})
        self.assertIn(json.dumps([value], ensure_ascii=False), result)
        self.assertIn('user-provided context', result)

    def test_missing_prompt_fails_clearly(self):
        with self.assertRaises(KeyError):
            get_prompt('missing.prompt')

    def test_json_examples_remain_literal(self):
        self.assertIn('"entities":{"people":[]', get_prompt('vision.remote_description'))
        self.assertIn('query_ar', get_prompt('query.arabic_translation'))
