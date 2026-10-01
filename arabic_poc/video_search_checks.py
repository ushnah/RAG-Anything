import unittest
from .video_search import group_matches


class VideoSearchChecks(unittest.TestCase):
    def test_group_preserves_provenance_and_timestamp(self):
        def row(video, identity, kind, score, start=None):
            anchor = {'kind': kind}
            if start is not None:
                anchor['start'] = start
            return dict(video_id=video, id=identity, anchor=anchor, evidence_type=kind,
                        original_text=identity, similarity=score)
        matches = [row('a', 'label', 'metadata', .8), row('a', 'speech', 'speech', .7, 12),
                   row('b', 'frame', 'visual_description', .6, 20)]
        grouped = group_matches(matches)
        self.assertEqual(len(grouped), 2)
        self.assertNotIn('start', grouped[0]['anchor'])
        self.assertEqual(grouped[0]['matches'][1]['anchor']['start'], 12)
        self.assertEqual(grouped[1]['matches'][0]['evidence_type'], 'visual_description')
        self.assertEqual(group_matches([]), [])

    def test_hybrid_and_abstention(self):
        from .video_search import hybrid_search, tokens
        self.assertEqual(tokens('أَرِنِي فيديو عن إِبراهيم'), ['ابراهيم'])
        row = dict(id='a', video_id='v', media_type='video', original_text='القبة الخضراء',
                   evidence_type='visual_description', anchor={'start':20})
        index = {'records':[row], 'vectors':[[1,0]]}
        self.assertTrue(hybrid_search(index, 'طائرة', [0,1])['no_match'])
        result = hybrid_search(index, 'القبة الخضراء', [.4,.9165])
        self.assertFalse(result['no_match'])
        self.assertGreater(result['sources'][0]['bm25'], 0)

    def test_short_speech_does_not_override_no_match(self):
        from .video_search import hybrid_search, tokens
        self.assertEqual(tokens('القبة الخضراء'), tokens('قبة خضراء'))
        row = dict(id='s', video_id='v', media_type='video',
                   original_text='اشتركوا في القناة', evidence_type='speech',
                   anchor={'start':0,'end':2})
        index = {'records':[row], 'vectors':[[1,0]]}
        self.assertTrue(hybrid_search(index, 'submarine underwater', [.6,.8])['no_match'])
        self.assertFalse(hybrid_search(index, 'اشتركوا في القناة', [.6,.8])['no_match'])

    def test_landmark_word_variants(self):
        from .video_search import tokens
        self.assertEqual(tokens('بوابة الملك'), tokens('باب الملك'))
        self.assertEqual(tokens('مئذنتان'), tokens('منارات'))

    def test_precise_landmark_words_outrank_broad_similarity(self):
        from .video_search import hybrid_search
        exact = dict(id='exact', video_id='exact', media_type='image',
                     original_text='باب الملك عبد العزيز مئذنتان', evidence_type='visual_description', anchor={})
        broad = dict(id='broad', video_id='broad', media_type='image',
                     original_text='بوابة الملك فهد مآذن', evidence_type='visual_description', anchor={})
        result = hybrid_search({'records': [exact, broad], 'vectors': [[.8, .6], [.9, .44]]},
                               'كم عدد مآذن بوابة الملك عبد العزيز؟', [.8, .6], media_type='image')
        self.assertEqual(result['sources'][0]['id'], 'exact')

    def test_arabic_entity_matches_english_description_with_low_similarity(self):
        from .video_search import hybrid_search
        row = dict(id='poster', video_id='poster', media_type='image',
                   original_text='Speech by Abdul Rahman Al-Sudais',
                   entities={'people': ['عبد الرحمن بن عبد العزيز السديس']},
                   evidence_type='visual_description', anchor={})
        index = {'records': [row], 'vectors': [[0, 1]]}
        result = hybrid_search(index, 'السديس', [1, 0], media_type='image')
        self.assertEqual(result['sources'][0]['id'], 'poster')
        self.assertTrue(result['sources'][0]['entity_match'])
        self.assertGreater(result['sources'][0]['bm25'], 0)
        self.assertTrue(hybrid_search(index, 'سد', [1, 0], media_type='image')['no_match'])
        self.assertTrue(hybrid_search(index, 'السديس', [1, 0], media_type='video')['no_match'])

    def test_video_entity_match_preserves_scene_and_entities(self):
        from .video_search import hybrid_search
        person = {'people': ['محمد بن سلمان']}
        row = dict(id='scene', video_id='video', media_type='video',
                   original_text='A group standing indoors', entities=person,
                   evidence_type='visual_description', anchor={'kind': 'visual_description', 'start': 40})
        result = hybrid_search({'records': [row], 'vectors': [[0, 1]]},
                               'محمد بن سلمان', [1, 0], media_type='video')
        match = result['sources'][0]['matches'][0]
        self.assertEqual(match['anchor']['start'], 40)
        self.assertEqual(match['entities'], person)

    def test_temporal_boundaries(self):
        from .video_search import aligned_evidence
        speech = dict(id='s',video_id='v',evidence_type='speech',anchor={'start':10,'end':15},original_text='speech')
        def frame(identity, time, video='v'):
            return dict(id=identity,video_id=video,evidence_type='visual_description',anchor={'start':time},original_text='frame')
        result=aligned_evidence(speech,[frame('a',12),frame('b',17),frame('c',18),frame('d',12,'other')])
        self.assertEqual([(r['id'],r['relation']) for r in result],[('a','overlap'),('b','nearby')])
        self.assertEqual(aligned_evidence(dict(speech,evidence_type='metadata'),[frame('a',12)]),[])


if __name__ == '__main__':
    unittest.main()
