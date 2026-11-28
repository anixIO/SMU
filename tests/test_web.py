"""HTTP, хранение сеансов и доступ администратора без реальной камеры."""

import csv
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import connection
from django.test import Client, TestCase, override_settings
from django.test.utils import setup_databases, teardown_databases
from PIL import Image

import web
from src.config import Config
from src.metrics import FrameMetrics
from src.models import Observation, WarningEvent
from src.sessions import close_monitors

TUNNEL = "smu-test.devtunnels.ms"
_test_database = None


def setUpModule():
    # Оба способа запуска тестов используют изолированную базу.
    global _test_database
    if connection.settings_dict["NAME"] != ":memory:":
        _test_database = setup_databases(verbosity=0, interactive=False)


def tearDownModule():
    if _test_database is not None:
        teardown_databases(_test_database, verbosity=0)


@override_settings(ALLOWED_HOSTS=["localhost", TUNNEL], CSRF_TRUSTED_ORIGINS=["https://" + TUNNEL],
                   PASSWORD_HASHERS=["django.contrib.auth.hashers.MD5PasswordHasher"])
class WebTests(TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.enterContext(override_settings(SMU_CONFIG_DIR=self.path, SMU_DATA_DIR=self.path))
        self.client = self.new_client()

    def tearDown(self):
        close_monitors()

    def new_client(self, host="localhost"):
        client = Client(enforce_csrf_checks=True, HTTP_HOST=host)
        client.get("/")
        return client

    def post(self, url, data=None, client=None, **extra):
        client = client or self.client
        return client.post(url, data or {}, HTTP_X_CSRFTOKEN=client.cookies["csrftoken"].value, **extra)

    def jpeg(self, size=(64, 48), format="JPEG"):
        image = io.BytesIO()
        Image.new("RGB", size).save(image, format=format)
        return SimpleUploadedFile("frame.jpg", image.getvalue(), content_type="image/jpeg")

    def test_dashboard_and_request_contract(self):
        page = self.client.get("/")
        self.assertContains(page, "Начать наблюдение")
        self.assertContains(page, 'id="perclos"')
        self.assertContains(page, 'aria-disabled="true"')
        self.assertEqual(self.client.head("/").status_code, 200)
        self.assertEqual(self.client.get("/report/").status_code, 404)
        for url in ("/api/start/", "/api/frame/", "/api/stop/"):
            self.assertEqual(self.client.get(url).status_code, 405)
            self.assertEqual(self.client.post(url).status_code, 403)
        self.assertEqual(self.post("/api/frame/").status_code, 409)

    def test_frontend_assets(self):
        for filename, content_type in [("monitor.js", "javascript"), ("monitor.css", "text/css")]:
            response = self.client.get("/assets/" + filename)
            try:
                self.assertEqual(response.status_code, 200)
                self.assertIn(content_type, response["Content-Type"])
                self.assertTrue(b"".join(response.streaming_content))
            finally:
                response.close()
        for path in ["/assets/config.yaml", "/assets/../config.yaml"]:
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_tunnel_response_length_and_csrf(self):
        self.assertEqual(web.tunnel_origin(f"https://{TUNNEL}/"), (f"https://{TUNNEL}", TUNNEL))
        phone = self.new_client(host=TUNNEL)
        page = phone.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(int(page["Content-Length"]), len(page.content))
        response = self.post("/api/frame/", client=phone, HTTP_ORIGIN=f"https://{TUNNEL}")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(int(response["Content-Length"]), len(response.content))
        self.assertEqual(self.post("/api/frame/", client=phone, HTTP_ORIGIN="https://untrusted.example").status_code, 403)
        self.assertEqual(phone.get("/admin/").status_code, 404)
        self.assertEqual(phone.get("/admin/login/").status_code, 404)

    @patch("src.detector.FatigueDetector")
    def test_session_frame_event_and_csv_download(self, detector):
        detector.return_value.process.return_value = FrameMetrics(
            detected=True, ear=.2, mar=.1, perclos=.6, perclos_ready=True, status="fatigue",
            fatigue=True, reasons=["Глаза закрыты 0.5 с"],
        )
        detector.return_value.should_alert.return_value = True
        response = self.post("/api/start/")
        self.assertEqual(response.status_code, 200)
        observation = response.json()["observation"]
        frame = self.post("/api/frame/", {"frame": self.jpeg()}, HTTP_X_OBSERVATION=observation)
        self.assertEqual(frame.status_code, 200)
        self.assertEqual(frame.json()["frames"], 1)
        self.assertEqual(frame.json()["ear"], .2)
        self.assertIsNone(frame.json()["pitch"])
        self.assertEqual(frame.json()["event"]["text"], "Глаза закрыты 0.5 с")
        self.assertNotIn("landmarks", frame.json())
        self.assertEqual(self.client.get("/report/").status_code, 404)
        detector.return_value.should_alert.return_value = False
        self.assertIsNone(self.post("/api/frame/", {"frame": self.jpeg()}).json()["event"])
        self.assertTrue(self.post("/api/stop/", {"observation": observation}).json()["has_report"])
        detector.return_value.close.assert_called_once()
        self.assertEqual(self.post("/api/stop/").status_code, 200)
        self.assertEqual(self.post("/api/frame/", {"frame": self.jpeg()}).status_code, 409)
        report = self.client.get("/report/")
        try:
            self.assertEqual(report.status_code, 200)
            rows = list(csv.DictReader(io.StringIO(b"".join(report.streaming_content).decode())))
        finally:
            report.close()
        self.assertEqual([row["Frame"] for row in rows], ["1", "2"])
        self.assertEqual(rows[0]["Fatigue"], "1")
        self.assertEqual(rows[0]["Pitch"], "")
        self.assertEqual(rows[0]["FaceVisible"], "1")
        self.assertEqual(WarningEvent.objects.get().observation_id.hex, observation.replace("-", ""))
        self.assertContains(self.client.get("/"), 'aria-disabled="false"')

    @patch("src.detector.FatigueDetector")
    def test_invalid_frames_do_not_reach_detector(self, detector):
        self.assertEqual(self.post("/api/start/").status_code, 200)
        invalid = [None, SimpleUploadedFile("frame.jpg", b"broken"),
                   self.jpeg((641, 480)), self.jpeg((640, 481)),
                   self.jpeg(format="PNG"), SimpleUploadedFile("frame.jpg", b"x" * 800001),
                   SimpleUploadedFile("frame.jpg", b"x" * 1000001)]
        for frame in invalid:
            with self.subTest(frame=frame):
                response = self.post("/api/frame/", {"frame": frame} if frame else {})
                self.assertEqual(response.status_code, 400)
        detector.return_value.process.assert_not_called()
        self.assertEqual(Observation.objects.get().frames, 0)

    @patch("src.detector.FatigueDetector")
    def test_separate_browsers_and_stale_stop(self, detector):
        start = self.post("/api/start/").json()["observation"]
        another = self.new_client()
        self.assertEqual(self.post("/api/frame/", client=another, HTTP_X_OBSERVATION=start).status_code, 409)
        second = self.post("/api/start/", client=another).json()["observation"]
        self.assertNotEqual(start, second)
        self.assertEqual(len(web.monitors), 2)
        newer = self.post("/api/start/").json()["observation"]
        self.assertEqual(len(web.monitors), 2)
        self.post("/api/stop/", {"observation": start})
        self.assertIsNone(Observation.objects.get(pk=newer).ended_at)
        self.assertIsNone(Observation.objects.get(pk=second).ended_at)
        self.assertIsNotNone(Observation.objects.get(pk=start).ended_at)

    @patch("src.detector.FatigueDetector", side_effect=RuntimeError("test failure"))
    def test_failed_start_leaves_no_active_detector(self, detector):
        with self.assertLogs("web", level="ERROR"):
            response = self.post("/api/start/")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(web.monitors)
        self.assertEqual(Observation.objects.count(), 0)
        self.assertEqual(self.post("/api/frame/").status_code, 409)

    def test_admin_signs_in_and_can_manage_own_installation(self):
        self.assertRedirects(self.client.get("/admin/"), "/admin/login/")
        admin = get_user_model().objects.create_user("test-staff", password="local-test-only", is_staff=True)
        ordinary = get_user_model().objects.create_user("reader", password="local-test-only")
        self.client.force_login(ordinary)
        self.assertRedirects(self.client.get("/admin/"), "/admin/login/")
        self.assertEqual(self.post("/admin/login/", {"username": admin.username, "password": "local-test-only"}).status_code, 302)
        self.assertContains(self.client.get("/admin/"), "Настройки детектора")

        bad = {**Config.DEFAULTS, "PERCLOS_MIN_SECONDS": 120}
        self.assertEqual(self.post("/admin/config/", bad).status_code, 400)
        self.assertFalse((self.path / "config.local.yaml").exists())
        good = {**Config.DEFAULTS, "EYE_AR_THRESH": .24}
        self.assertEqual(self.post("/admin/config/", good).status_code, 302)
        self.assertEqual(Config(self.path / "config.local.yaml")["EYE_AR_THRESH"], .24)
        self.assertEqual(self.post("/admin/config/", {"reset": 1}).status_code, 302)
        self.assertFalse((self.path / "config.local.yaml").exists())

        warning = WarningEvent.objects.create(observation=Observation.objects.create(), reason="Глаза закрыты")
        self.assertContains(self.client.get("/admin/"), warning.reason)
        self.assertEqual(self.post(f"/admin/warnings/{warning.pk}/", {"false_alarm": 1}).status_code, 302)
        warning.refresh_from_db()
        self.assertTrue(warning.false_alarm)
        self.assertEqual(self.post("/admin/logout/").status_code, 302)
        self.assertRedirects(self.client.get("/admin/"), "/admin/login/")

    def test_legacy_csv_needs_staff(self):
        old = self.path / "web_20250102_030405_123456.csv"
        old.write_text("Frame,Timestamp\n1,0.01\n")
        self.assertEqual(self.client.get(f"/admin/archive/{old.name}/").status_code, 302)
        admin = get_user_model().objects.create_user("archivist", is_staff=True)
        self.client.force_login(admin)
        self.assertContains(self.client.get("/admin/"), "Архив CSV")
        report = self.client.get(f"/admin/archive/{old.name}/")
        try:
            self.assertEqual(report.status_code, 200)
            self.assertEqual(b"".join(report.streaming_content), old.read_bytes())
        finally:
            report.close()
