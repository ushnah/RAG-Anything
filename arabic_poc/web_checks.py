import http.client
import json
import threading
import unittest
from unittest.mock import patch
from . import web


class WebChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = web.ThreadingHTTPServer(('127.0.0.1', 0), web.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, method='GET', body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        status, data, response_headers = response.status, response.read(), dict(response.getheaders())
        conn.close()
        return status, data, response_headers

    def test_catalog_and_private_paths(self):
        code, raw, _ = self.request('/api/catalog')
        self.assertEqual(code, 200)
        data = json.loads(raw)
        self.assertEqual([d['id'] for d in data['datasets']], ['text', 'media'])
        self.assertNotIn('"source":', raw.decode())
        self.assertNotIn('normalized_text', raw.decode())
        self.assertEqual(data['datasets'][1]['count'], 14)

    def test_media_seek_range(self):
        row = next(r for r in web.catalog('media') if r['document']=='arabic_greeting.ogg')
        path = f'/api/media/media/{row["id"]}'
        code, data, headers = self.request(path, headers={'Range': 'bytes=0-31'})
        self.assertEqual(code, 206)
        self.assertEqual(len(data), 32)
        self.assertEqual(data, web.Path(row['source']).read_bytes()[:32])
        self.assertIn('bytes 0-31/', headers['Content-Range'])
        self.assertEqual(self.request(path, headers={'Range': 'bytes=999999999-'})[0], 416)

    def test_reject_unknown_and_cross_origin(self):
        self.assertEqual(self.request('/api/media/text/../../.env')[0], 404)
        self.assertEqual(self.request('/.env')[0], 404)
        body = json.dumps({'dataset':'text','question':'مرحبا'})
        self.assertEqual(self.request('/api/ask','POST',body,{'Origin':'https://example.com'})[0],403)
        self.assertEqual(self.request('/api/catalog',headers={'Host':'example.com'})[0],403)
        self.assertEqual(self.request('/api/ask','POST','{"dataset":"unknown"}')[0],400)

    def test_saved_examples_labeled(self):
        code, data, _ = self.request('/api/examples/audio')
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(data)['mode'], 'saved')

    def test_single_live_job(self):
        body = json.dumps({'dataset':'text','question':'من يشرف على الفهرسة؟'})
        web.BUSY.acquire()
        try:
            self.assertEqual(self.request('/api/ask','POST',body)[0],409)
        finally:
            web.BUSY.release()
        def fake_run(job_id,dataset,question):
            with web.LOCK:
                web.JOBS[job_id]={'status':'done','result':{'answer':'اختبار','sources':[]}}
            web.BUSY.release()
        with patch.object(web,'run_question',fake_run):
            code,data,_=self.request('/api/ask','POST',body)
            self.assertEqual(code,202)
            job_id=json.loads(data)['job_id']
            status,raw,_=self.request('/api/jobs/'+job_id)
            self.assertEqual(status,200)
            self.assertEqual(json.loads(raw)['status'],'done')


if __name__=='__main__':
    unittest.main()
