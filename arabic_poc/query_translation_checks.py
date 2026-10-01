import unittest
from unittest.mock import patch
from .query_translation import query_variants, search_variants


class QueryTranslationChecks(unittest.TestCase):
    @patch('arabic_poc.query_translation.chat')
    def test_arabic_skips_translation(self, chat):
        self.assertEqual(query_variants('أرني القبة الخضراء'), ['أرني القبة الخضراء'])
        chat.assert_not_called()

    @patch('arabic_poc.query_translation.enabled', return_value=True)
    @patch('arabic_poc.query_translation.chat')
    def test_translation_and_failure_fallback(self, chat, enabled):
        chat.return_value = '{"query_ar":"أرني القبة الخضراء"}'
        self.assertEqual(query_variants('Show me the green dome'), ['Show me the green dome', 'أرني القبة الخضراء'])
        for raw in ['not json', '{"query_ar":123}', '{"query_ar":""}']:
            chat.return_value = raw
            self.assertEqual(query_variants('green dome'), ['green dome'])
        chat.side_effect = RuntimeError('unavailable')
        self.assertEqual(query_variants('green dome'), ['green dome'])

    def test_arabic_expansion_recovers_result_and_deduplicates(self):
        row = dict(id='dome', video_id='dome', media_type='image', original_text='القبة الخضراء',
                   evidence_type='visual_description', anchor={})
        index = {'records': [row], 'vectors': [[1, 0]]}
        result = search_variants(index, ['Show me the green dome', 'أرني القبة الخضراء'],
                                 [[0, 1], [1, 0]], media_type='image')
        self.assertEqual([r['id'] for r in result['sources']], ['dome'])
        result = search_variants(index, ['green dome', 'القبة الخضراء'], [[1, 0], [1, 0]], media_type='image')
        self.assertEqual(len(result['sources']), 1)
        self.assertEqual(len(result['sources'][0]['matches']), 1)
        self.assertTrue(search_variants(index, ['green dome'], [[1, 0]], media_type='video')['no_match'])
