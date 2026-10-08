"""Checks for retrieval routing, provenance and media filtering."""
import unittest
from .visual import rank, search_text


class VisualChecks(unittest.TestCase):
    def test_video_request_never_returns_image(self):
        index = {'records': [
            {'id': 'photo', 'media_type': 'image'},
            {'id': 'frame', 'media_type': 'video', 'anchor': {'start': 20}}],
            'vectors': [[1, 0], [0.8, 0.2]]}
        result = rank(index, [1, 0], 'video', 3)
        self.assertEqual([r['id'] for r in result], ['frame'])
        self.assertEqual(result[0]['anchor']['start'], 20)
        self.assertEqual(rank({'records': [], 'vectors': []}, [1, 0], 'all', 3), [])

    def test_visual_job_uses_search_and_video_filter(self):
        from unittest.mock import patch
        from . import web
        web.JOBS['visual-test'] = {'status': 'running'}
        web.BUSY.acquire()
        response = {'answer': 'Candidates', 'sources': [], 'no_match': True}
        with patch('arabic_poc.search_service.search', return_value=response) as search, \
                patch.object(web, 'catalog', return_value=[]), \
                patch.object(web.subprocess, 'run') as run:
            web.run_question('visual-test', 'visual', 'show gate video', 'video')
        search.assert_called_once_with('show gate video',
                                      web.ROOT / web.DATASETS['visual']['storage'], 'video')
        run.assert_not_called()
        self.assertEqual(web.JOBS.pop('visual-test')['status'], 'done')
        self.assertFalse(web.BUSY.locked())

    def test_source_label_distinct_from_model_description(self):
        row = {'original_text': 'A large arch', 'source_label': 'King Abdul Aziz Gate'}
        self.assertIn('Source label:', search_text(row))
        self.assertEqual(row['original_text'], 'A large arch')


if __name__ == '__main__':
    unittest.main()
