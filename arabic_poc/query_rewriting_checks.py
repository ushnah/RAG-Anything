import json
import unittest
from unittest.mock import patch
from .query_rewriting import rewrite_person_query, candidates


class QueryRewritingChecks(unittest.TestCase):
    @patch('arabic_poc.query_rewriting.enabled', return_value=True)
    @patch('arabic_poc.query_rewriting.chat')
    def test_unlisted_nickname_preserves_constraints(self, chat, enabled):
        chat.return_value = json.dumps({'resolutions': [{'mention': 'Mb Salman', 'person_id': 'mohammed_bin_salman', 'status': 'resolved'}]})
        self.assertEqual(rewrite_person_query('Show photos of Mb Salman in 2020, not posters'),
                         'Show photos of Mohammed bin Salman in 2020, not posters')
        supplied = json.loads(chat.call_args.args[0][1]['content'])
        self.assertNotIn('aliases', supplied['people'][0])

    @patch('arabic_poc.query_rewriting.enabled', return_value=True)
    @patch('arabic_poc.query_rewriting.chat')
    def test_arabic_and_multiple_people(self, chat, enabled):
        chat.return_value = json.dumps({'resolutions': [
            {'mention':'بن سلمان','person_id':'mohammed_bin_salman','status':'resolved'},
            {'mention':'سوديس','person_id':'abdul_rahman_al_sudais','status':'resolved'}]})
        self.assertEqual(rewrite_person_query('أرني بن سلمان مع سوديس'),
                         'أرني Mohammed bin Salman مع Abdul Rahman Al-Sudais')

    @patch('arabic_poc.query_rewriting.enabled', return_value=True)
    @patch('arabic_poc.query_rewriting.chat')
    def test_invalid_ambiguous_and_failed_responses(self, chat, enabled):
        for value in [[], [{'mention':'MBS','person_id':'invented','status':'resolved'}],
                      [{'mention':'absent','person_id':'mohammed_bin_salman','status':'resolved'}],
                      [{'mention':'MBS','person_id':'mohammed_bin_salman','status':'ambiguous'}]]:
            chat.return_value = json.dumps({'resolutions':value})
            self.assertEqual(rewrite_person_query('Show MBS'), 'Show MBS')
        chat.side_effect = RuntimeError('offline')
        self.assertEqual(rewrite_person_query('green dome'), 'green dome')

    def test_shortlist_bounded(self):
        self.assertLessEqual(len(candidates('MBS', limit=2)), 2)
        self.assertEqual(candidates('MBS', limit=1)[0]['id'], 'mohammed_bin_salman')

    @patch('arabic_poc.query_rewriting.enabled', return_value=True)
    @patch('arabic_poc.query_rewriting.chat')
    def test_no_alias_query_is_unchanged(self, chat, enabled):
        chat.return_value = '{"resolutions": []}'
        self.assertEqual(rewrite_person_query('Show the green dome'), 'Show the green dome')
