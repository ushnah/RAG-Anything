import tempfile
import json
import unittest
from pathlib import Path
from .__main__ import Extractor, checked_answer, records, resolve_evidence, save_json


class Checks(unittest.TestCase):
    def test_citation_validation(self):
        source_id = 'a' * 24 + '-000001'
        evidence = [{'id': source_id, 'original_text': 'إِنَّ هَذَا نَصٌّ'}]
        answer = {'answer': f'الاقتباس [{source_id}]',
                  'citations': [{'id': source_id, 'quote': 'إِنَّ'}]}
        self.assertEqual(checked_answer(json.dumps(answer), evidence), answer)
        unmarked = dict(answer, answer='الاقتباس')
        self.assertEqual(checked_answer(json.dumps(unmarked), evidence)['answer'], f'الاقتباس [{source_id}]')
        answer['citations'][0]['quote'] = 'ان'
        self.assertEqual(checked_answer(json.dumps(answer), evidence)['citations'], [])
        for raw in ['not JSON', '[]', '{"answer": "x", "citations": [null]}']:
            self.assertEqual(checked_answer(raw, evidence)['citations'], [])

    def test_page_grouping(self):
        class FakeOCR:
            def parse_pdf(self, path, lang):
                return [{'text': 'أ', 'page_idx': 0}, {'text': 'ب', 'page_idx': 0},
                        {'text': 'ج', 'page_idx': 1}]
        extractor = Extractor()
        extractor.ocr = FakeOCR()
        rows = list(extractor.extract(Path('example.pdf')))
        self.assertEqual(rows[0][0], 'أ\nب')
        self.assertEqual(rows[1][1]['page'], 2)

    def test_original_and_normalized(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'book.txt'
            original = 'إِنَّ هَذَا نَصٌّ عَرَبِيٌّ'
            path.write_text(original)
            row = records(path, Extractor())[0]
            self.assertEqual(row['original_text'], original)
            self.assertEqual(row['normalized_text'], 'ان هذا نص عربي')
            self.assertEqual(records(path, Extractor())[0]['id'], row['id'])
            path.write_text('نص جديد')
            self.assertNotEqual(records(path, Extractor())[0]['id'], row['id'])

    def test_resolve_only_known_sources(self):
        source_id = 'a' * 24 + '-000001'
        row = {'id': source_id, 'original_text': 'نص'}
        chunks = [{'content': f'SOURCE_ID={source_id}'},
                  {'file_path': f'anchor:{source_id}'},
                  {'content': 'SOURCE_ID=' + 'b' * 24 + '-000001'}]
        self.assertEqual(resolve_evidence(chunks, {source_id: row}), [row])

    def test_empty_source_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'empty.txt'
            path.write_text('  ')
            with self.assertRaisesRegex(ValueError, 'No text'):
                records(path, Extractor())

    def test_anchors_and_metadata(self):
        class FakeExtractor:
            def extract(self, path):
                yield 'نص صفحة', {'kind': 'page', 'page': 3}, 'paddleocr:ar'
                yield 'كلام', {'kind': 'speech', 'start': 2.5, 'end': 4.0}, 'whisper:small'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'book.pdf'
            path.write_bytes(b'dummy')
            save_json(path.with_name('book.pdf.metadata.json'), {'author': 'المؤلف'})
            rows = records(path, FakeExtractor())
            self.assertEqual(rows[0]['anchor']['page'], 3)
            self.assertEqual(rows[1]['anchor']['start'], 2.5)
            self.assertEqual(rows[0]['metadata']['author'], 'المؤلف')



class EvaluationChecks(unittest.TestCase):
    def test_metrics(self):
        from .evaluate import extraction_metrics, retrieval_metrics
        self.assertEqual(extraction_metrics('نص عربي', 'نص عربي')['cer'], 0)
        self.assertEqual(extraction_metrics('a b', 'a c')['wer'], 0.5)
        scores = retrieval_metrics(['a', 'b'], ['x', 'a', 'a'])
        self.assertEqual(scores['recall@5'], 0.5)
        self.assertEqual(scores['mrr'], 0.5)

    def test_long_original_passages(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'long.txt'
            text = 'أَبْجَد ' * 1000
            path.write_text(text)
            rows = records(path, Extractor())
            self.assertGreater(len(rows), 1)
            for row in rows:
                anchor = row['anchor']
                self.assertEqual(row['original_text'], text[anchor['char_start']:anchor['char_end']])
                self.assertLessEqual(len(row['original_text']), 1800)

    def test_qwen_selection_and_visual_provenance(self):
        from unittest.mock import patch
        class FakeQwen:
            def __init__(self, *args, **kwargs):
                self.describe = kwargs.get('describe', False)
            def parse_image(self, path, lang='ar'):
                return [{'text': 'وصف آلي' if self.describe else 'النص الأصلي', 'page_idx': 0}]
        with patch('arabic_poc.models.QwenParser', FakeQwen):
            rows = list(Extractor(ocr_engine='qwen', describe_images=True).extract(Path('image.png')))
            self.assertEqual(rows[0][2], 'qwen:ar')
            self.assertEqual(rows[1][1]['kind'], 'visual_description')
            self.assertEqual(rows[1][2], 'qwen-vl:generated')


class InsertionChecks(unittest.IsolatedAsyncioTestCase):
    async def test_failed_indexing_is_not_success(self):
        from unittest.mock import AsyncMock
        from types import SimpleNamespace
        from raganything.utils import insert_text_content
        rag = SimpleNamespace(ainsert=AsyncMock(), doc_status=SimpleNamespace(
            get_by_id=AsyncMock(return_value={'status': 'failed', 'error_msg': 'endpoint unavailable'})))
        with self.assertRaisesRegex(RuntimeError, 'endpoint unavailable'):
            await insert_text_content(rag, 'نص', ids='doc-1')



class MediaChecks(unittest.TestCase):
    def test_asr_timestamps_do_not_exceed_media(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        extractor = Extractor()
        extractor.asr = SimpleNamespace(transcribe=lambda *args, **kwargs: {'segments': [
            {'text': 'السلام عليكم', 'start': 0, 'end': 2},
            {'text': 'خارج الملف', 'start': 3, 'end': 4}]})
        probe = json.dumps({'streams': [{'codec_type': 'audio'}], 'format': {'duration': '1.3'}})
        with patch('arabic_poc.__main__.shutil.which', return_value='/bin/tool'), patch('arabic_poc.__main__.command', return_value=probe):
            rows = list(extractor.extract(Path('short.ogg')))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1]['end'], 1.3)

if __name__ == '__main__':
    unittest.main()
