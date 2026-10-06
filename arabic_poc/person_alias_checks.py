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

    def test_embedded_aliases_and_boundaries(self):
        from .person_aliases import MATCHER, canonical_query_variants
        cases = [
            ('show me a picture of MBS', 'show me a picture of Mohammed bin Salman'),
            ('Show MbS at an event', 'Show Mohammed bin Salman at an event'),
            ('Mohammad bin Salman', 'Mohammed bin Salman'),
            ('أرني صورة محمد بن سلمان', 'أرني صورة Mohammed bin Salman'),
            ('فيديو ابن سلمان في مكة', 'فيديو Mohammed bin Salman في مكة'),
            ('Show Sheikh Sudais speaking', 'Show Abdul Rahman Al-Sudais speaking'),
            ('Sudais', 'Abdul Rahman Al-Sudais'),
            ('عبد الرحمن السديس', 'Abdul Rahman Al-Sudais'),
            ('Photos of Ibn Saud', 'Photos of Abdulaziz Ibn Saud'),
            ('King Abdulaziz', 'Abdulaziz Ibn Saud'),
            ('الملك عبدالعزيز', 'Abdulaziz Ibn Saud'),
            ('أرني مُحَمَّد بن سَلْمان!', 'أرني Mohammed bin Salman!'),
            ('MBS and Sudais', 'Mohammed bin Salman and Abdul Rahman Al-Sudais'),
            ('not MBS, only Sudais', 'not Mohammed bin Salman, only Abdul Rahman Al-Sudais'),
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(MATCHER.rewrite(query), expected)
        for query in ['a green dome', 'LAMBS', 'MBSomething', 'محمد', 'عبد الرحمن']:
            self.assertEqual(canonical_query_variants(query), [query])

    def test_ambiguous_alias_blocks_shorter_fallback(self):
        from .person_aliases import AliasMatcher
        registry = {
            'a': {'name_ar': 'ألف', 'name_en': 'Person A', 'aliases': ['Shared Name', 'Shared']},
            'b': {'name_ar': 'باء', 'name_en': 'Person B', 'aliases': ['Shared Name']},
        }
        matcher = AliasMatcher(registry)
        self.assertEqual(matcher.rewrite('Photo of Shared Name'), 'Photo of Shared Name')
        self.assertTrue(matcher.find('Shared Name')[0]['ambiguous'])
        self.assertEqual(matcher.rewrite('Photo of Person A'), 'Photo of Person A')

    def test_description_only_gallery_matches_embedded_alias(self):
        import copy
        examples = [
            ('show me a picture of MBS', 'Mohammed bin Salman at a Saudi government event'),
            ('show Sheikh Sudais speaking', 'Abdul Rahman Al-Sudais at a microphone'),
            ('أرني الملك عبدالعزيز', 'Abdulaziz Ibn Saud in a portrait'),
        ]
        for query, description in examples:
            row = dict(id='photo', video_id='photo', media_type='image', original_text=description,
                       evidence_type='visual_description', anchor={})
            index = {'records': [row], 'vectors': [[0, 1]]}
            original = copy.deepcopy(index)
            result = hybrid_search(index, query, [1, 0], media_type='image')
            self.assertEqual(result['sources'][0]['id'], 'photo')
            self.assertTrue(result['sources'][0]['person_alias_match'])
            self.assertEqual(index, original)
            self.assertTrue(hybrid_search(index, query, [1, 0], media_type='video')['no_match'])


if __name__ == '__main__':
    unittest.main()
