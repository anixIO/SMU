# Проверки Django-интерфейса без доступа к настоящей камере.
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from PIL import Image

import web
from src.metrics import FrameMetrics


class WebTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base = patch.object(web, 'BASE_DIR', Path(directory.name))
        base.start()
        self.addCleanup(base.stop)
        monitor = patch.object(web, 'monitor', web.Monitor())
        monitor.start()
        self.addCleanup(monitor.stop)
        self.addCleanup(web.monitor.close)
        self.client = Client(enforce_csrf_checks=True)
        self.client.get('/')
        self.token = self.client.cookies['csrftoken'].value

    def post(self, url, data=None):
        return self.client.post(url, data or {}, HTTP_X_CSRFTOKEN=self.token)

    def jpeg(self, size=(64, 48), format='JPEG'):
        image = io.BytesIO()
        Image.new('RGB', size).save(image, format=format)
        return SimpleUploadedFile('frame.jpg', image.getvalue(), content_type='image/jpeg')

    def test_dashboard_and_request_contract(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        self.assertContainsText(page, 'Начать наблюдение')
        self.assertContainsText(page, 'id="perclos"')
        self.assertContainsText(page, 'aria-disabled="true"')
        self.assertEqual(self.client.get('/report/').status_code, 404)
        for url in ('/api/start/', '/api/frame/', '/api/stop/'):
            self.assertEqual(self.client.get(url).status_code, 405)
            self.assertEqual(self.client.post(url).status_code, 403)
        self.assertEqual(self.post('/api/frame/').status_code, 409)

    def test_frontend_assets(self):
        for filename, content_type in [('monitor.js', 'javascript'), ('monitor.css', 'text/css')]:
            response = self.client.get('/assets/' + filename)
            try:
                self.assertEqual(response.status_code, 200)
                self.assertIn(content_type, response['Content-Type'])
                self.assertTrue(b''.join(response.streaming_content))
            finally:
                response.close()
        for path in ['/assets/config.yaml', '/assets/../config.yaml']:
            self.assertEqual(self.client.get(path).status_code, 404)

    def assertContainsText(self, response, text):
        self.assertIn(text, response.content.decode())

    @patch('src.detector.FatigueDetector')
    def test_session_frame_event_and_csv_download(self, detector):
        detector.return_value.process.return_value = FrameMetrics(
            detected=True, ear=.2, mar=.1, perclos=.6, perclos_ready=True,
            fatigue=True, reasons=['глаза закрыты 0.5 с'],
        )
        detector.return_value.should_alert.return_value = True
        self.assertEqual(self.post('/api/start/').status_code, 200)
        frame = self.post('/api/frame/', {'frame': self.jpeg()})
        self.assertEqual(frame.status_code, 200)
        self.assertEqual(frame.json()['frames'], 1)
        self.assertEqual(frame.json()['ear'], .2)
        self.assertEqual(frame.json()['event']['text'], 'глаза закрыты 0.5 с')
        self.assertNotIn('landmarks', frame.json())
        detector.return_value.should_alert.return_value = False
        self.assertIsNone(self.post('/api/frame/', {'frame': self.jpeg()}).json()['event'])
        self.assertEqual(self.post('/api/stop/').status_code, 200)
        detector.return_value.close.assert_called_once()
        self.assertEqual(self.post('/api/stop/').status_code, 200)
        self.assertEqual(self.post('/api/frame/', {'frame': self.jpeg()}).status_code, 409)
        report = self.client.get('/report/')
        try:
            self.assertEqual(report.status_code, 200)
            rows = list(csv.DictReader(io.StringIO(b''.join(report.streaming_content).decode())))
        finally:
            report.close()
        self.assertEqual([row['Frame'] for row in rows], ['1', '2'])
        self.assertEqual(rows[0]['Fatigue'], '1')
        self.assertIn('глаза закрыты', (web.BASE_DIR / 'output/fatigue_log.txt').read_text())
        self.assertContainsText(self.client.get('/'), 'aria-disabled="false"')

    @patch('src.detector.FatigueDetector')
    def test_invalid_frames_do_not_reach_detector(self, detector):
        self.post('/api/start/')
        invalid = [None, SimpleUploadedFile('frame.jpg', b'broken'),
                   self.jpeg((641, 480)), self.jpeg((640, 481)),
                   self.jpeg(format='PNG'), SimpleUploadedFile('frame.jpg', b'x' * 800001)]
        for frame in invalid:
            with self.subTest(frame=frame):
                response = self.post('/api/frame/', {'frame': frame} if frame else {})
                self.assertEqual(response.status_code, 400)
        detector.return_value.process.assert_not_called()
        self.assertEqual(web.monitor.frames, 0)

    @patch('src.detector.FatigueDetector')
    def test_restarting_closes_previous_session_and_keeps_reports(self, detector):
        self.post('/api/start/')
        first_report, first_log = web.monitor.csv_path, web.monitor.video_log
        self.post('/api/start/')
        self.assertTrue(first_log.file.closed)
        detector.return_value.close.assert_called_once()
        self.assertNotEqual(first_report, web.monitor.csv_path)
        self.assertTrue(first_report.is_file())

    @patch('src.detector.FatigueDetector', side_effect=RuntimeError('test failure'))
    def test_failed_start_leaves_no_active_detector(self, detector):
        with self.assertLogs('web', level='ERROR'):
            response = self.post('/api/start/')
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(web.monitor.detector)
        self.assertIsNone(web.monitor.video_log)
        self.assertEqual(self.post('/api/frame/').status_code, 409)


if __name__ == '__main__':
    unittest.main()
