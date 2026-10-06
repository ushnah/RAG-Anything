import unittest
from unittest.mock import patch
from .search_service import provenance, rerank, encoder


class SearchServiceChecks(unittest.TestCase):
    def test_person_provenance(self):
        row = {'entities': {'people': ['عبد الرحمن السديس']}, 'original_text': 'النص المرئي: Abdul Rahman Al-Sudais | المشهد: بوستر', 'anchor': {'kind': 'visual_description'}}
        self.assertEqual(provenance(row)['people'][0]['provenance'], 'visible_text_mention')
        row['anchor']['person_matches'] = [{'name': 'abdul_rahman_al_sudais'}]
        self.assertEqual(provenance(row)['people'][0]['provenance'], 'face_gallery_match')
        row['evidence_type'] = 'speech'
        self.assertEqual(provenance(row)['type'], 'transcript_mention')

    @patch('arabic_poc.search_service.embedding_model')
    def test_encoder_loaded_once(self, load):
        encoder.cache_clear()
        self.assertIs(encoder(), encoder())
        self.assertEqual(load.call_count, 1)
        encoder.cache_clear()

    @patch('arabic_poc.search_service.enabled', return_value=True)
    @patch('arabic_poc.search_service.chat')
    def test_reranker_can_promote_candidate_twenty(self, chat, enabled):
        rows = [dict(document=str(i), media_type='image', original_text='scene', hybrid_score=1-i/100, matches=[]) for i in range(20)]
        import json
        chat.return_value = json.dumps({'scores': [{'id': str(i), 'score': 1 if i == 19 else 0} for i in range(20)]})
        result = rerank('target', {'sources': rows})
        self.assertEqual(result['sources'][0]['document'], '19')
        self.assertEqual(len(result['sources']), 1)
        chat.return_value = '{"scores": []}'
        fallback = rerank('target', {'sources': rows})
        self.assertEqual(fallback['reranking'], 'hybrid_fallback')
        self.assertEqual(fallback['sources'][0]['document'], '0')

    @patch('arabic_poc.search_service.enabled', return_value=False)
    def test_failure_does_not_admit_unqualified_candidates(self, enabled):
        row = dict(document='irrelevant', media_type='image', original_text='scene',
                   hybrid_score=.2, hybrid_qualified=False, matches=[])
        result = rerank('unrelated', {'sources': [row], 'answer': 'candidates'})
        self.assertEqual(result['sources'], [])
        self.assertTrue(result['no_match'])
