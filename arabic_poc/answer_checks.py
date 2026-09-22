import json
import unittest
from unittest.mock import patch
from .answers import answer_from_evidence

class AnswerChecks(unittest.TestCase):
    def setUp(self):
        self.row = dict(id='a'*24,document='clip.mp4',original_text='قبة خضراء',anchor={'kind':'visual_description','start':20},engine='generated')
        self.group = dict(self.row,id='b'*24,matches=[self.row])
        self.result = {'sources':[self.group],'citations':[]}

    def test_nested_citation_maps_to_card_and_real_timestamp(self):
        response=json.dumps({'answer':'بحسب الوصف الآلي، قبة خضراء ['+self.row['id']+']','citations':[{'id':self.row['id'],'quote':'قبة خضراء','anchor':{'start':999}}]})
        with patch('arabic_poc.answers.generate',return_value=response):
            result=answer_from_evidence('ماذا يظهر؟',self.result,[self.row])
        self.assertEqual(result['generation'],'grounded')
        self.assertEqual(result['citations'][0]['source_id'],self.group['id'])
        self.assertEqual(result['citations'][0]['anchor']['start'],20)

    def test_fabricated_quote_rejected_but_sources_retained(self):
        response=json.dumps({'answer':'ادعاء','citations':[{'id':self.row['id'],'quote':'نص غير موجود'}]})
        with patch('arabic_poc.answers.generate',return_value=response):
            result=answer_from_evidence('سؤال',self.result,[self.row])
        self.assertEqual(result['generation'],'invalid_citations')
        self.assertEqual(result['sources'],self.result['sources'])

    def test_no_match_skips_generation(self):
        with patch('arabic_poc.answers.generate') as generate:
            result=answer_from_evidence('سؤال',{'no_match':True,'sources':[]},[])
        generate.assert_not_called()
        self.assertEqual(result['generation'],'no_match')

    def test_insufficient_and_service_failure_preserve_evidence(self):
        with patch('arabic_poc.answers.generate',return_value='{"insufficient_evidence":true,"citations":[]}'):
            self.assertEqual(answer_from_evidence('سؤال',self.result,[self.row])['generation'],'insufficient_evidence')
        with patch('arabic_poc.answers.generate',side_effect=RuntimeError('offline')):
            self.assertEqual(answer_from_evidence('سؤال',self.result,[self.row])['sources'],self.result['sources'])
