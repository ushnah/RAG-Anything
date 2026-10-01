import unittest
from .person_aliases import resolve_people
from .video_search import hybrid_search


class PersonAliasChecks(unittest.TestCase):
    def test_known_and_unknown_names(self):
        for name in ['عبد الرحمن السديس', 'عبد الرحمن بن عبد العزيز السديس', 'Abdul Rahman Al-Sudais', 'abdul_rahman_al_sudais']:
            self.assertEqual(resolve_people({'people': [name]})[0]['id'], 'abdul_rahman_al_sudais')
        for name in ['عبد الرحمن', 'عبد العزيز', 'محمد', 'Unknown Person']:
            self.assertEqual(resolve_people({'people': [name]}), [])

    def test_alias_queries_retrieve_all_name_forms(self):
        names = ['عبد الرحمن السديس', 'عبد الرحمن بن عبد العزيز السديس', 'abdul_rahman_al_sudais']
        rows = [dict(id=str(i), video_id=str(i), media_type='image', original_text='Portrait',
                     entities={'people': [name]}, evidence_type='visual_description', anchor={}) for i, name in enumerate(names)]
        rows.append(dict(rows[0], id='other', video_id='other', entities={'people': ['محمد بن سلمان']}))
        index = {'records': rows, 'vectors': [[0, 1]] * len(rows)}
        for query in ['عبد الرحمن بن عبد العزيز السديس', 'عبد الرحمن السديس', 'Abdul Rahman Al-Sudais', 'Abdur Rahman Al Sudais']:
            result = hybrid_search(index, query, [1, 0], media_type='image')
            self.assertEqual({r['id'] for r in result['sources']}, {'0', '1', '2'})
            self.assertTrue(all(r['person_alias_match'] for r in result['sources']))
        self.assertTrue(hybrid_search(index, 'السديس', [1, 0], media_type='video')['no_match'])


if __name__ == '__main__':
    unittest.main()
