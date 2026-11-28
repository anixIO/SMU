# Проверки расчётов и работы компонентов без включения камеры.
# Запуск: python -m unittest discover -s tests -v

import unittest
import csv
import tempfile
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.app import App, Config
from src.detector import FatigueDetector, FatigueRules
from src.interface import FontManager, HUDRenderer
from src.metrics import FrameMetrics, MetricsCalculator, TemporalMetrics
from src.reporting import AlertSystem, VideoLogger, plot_metrics


class TemporalTests(unittest.TestCase):
    def test_durations_at_different_fps(self):
        for fps in (10, 15, 30, 60):
            with self.subTest(fps=fps):
                state = TemporalMetrics(5, 5)
                for i in range(fps * 5 + 1):
                    state.update(i / fps, True, True)
                    if i / fps < 5:
                        self.assertFalse(state.ready)
                self.assertAlmostEqual(state.eye_seconds, 5)
                self.assertAlmostEqual(state.yawn_seconds, 5)
                self.assertTrue(state.ready)
                self.assertEqual(state.perclos, 1)

    def test_time_weighting_and_clipping(self):
        state = TemporalMetrics(2, 2)
        for t, closed in [(0, True), (.2, True), (.5, False), (1.5, False), (2, False)]:
            state.update(t, closed, False)
        self.assertAlmostEqual(state.perclos, .25)
        state.update(2.25, False, False)
        self.assertAlmostEqual(state.perclos, .125)

    def test_reset_does_not_join_events(self):
        state = TemporalMetrics(5, 5)
        state.update(0, True, True)
        state.update(.3, True, True)
        state.reset()
        state.update(1, True, True)
        self.assertEqual(state.eye_seconds, 0)
        self.assertEqual(state.yawn_seconds, 0)
        self.assertFalse(state.ready)
        self.assertEqual(state.perclos, 0)

    def test_long_gap_resets_history(self):
        state = TemporalMetrics(5, 5)
        state.update(0, True, True)
        state.update(.5, True, True)
        state.update(3, True, True)
        self.assertEqual(state.eye_seconds, 0)
        self.assertFalse(state.ready)
        self.assertEqual(state.perclos, 0)

    def test_opening_resets_continuous_events(self):
        state = TemporalMetrics(5, 5)
        state.update(0, True, True)
        state.update(.5, False, False)
        state.update(.6, True, True)
        self.assertEqual(state.eye_seconds, 0)
        self.assertEqual(state.yawn_seconds, 0)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            TemporalMetrics(2, 5)


class ComponentTests(unittest.TestCase):
    # Проверяем, что объединение файлов и оптимизации не меняют результаты.

    def test_reporting_direct_launch_explains_entry_point(self):
        script = Path(__file__).resolve().parent.parent / 'src' / 'reporting.py'
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(script)], cwd=directory,
                                    capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Для запуска приложения', result.stdout)
        self.assertIn(str(script.parent.parent / 'main.py'), result.stdout)
        self.assertNotIn('Traceback', result.stderr)

    def test_eye_and_mouth_geometry(self):
        eye = [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)]
        self.assertAlmostEqual(MetricsCalculator.eye_aspect_ratio(eye), 0.5)
        mouth = [(0, 0), (4, 0), (2, 1), (2, -1)]
        self.assertAlmostEqual(MetricsCalculator.mouth_aspect_ratio(mouth), 0.5)
        self.assertEqual(MetricsCalculator.eye_aspect_ratio([(0, 0)] * 6), 0)
        self.assertEqual(MetricsCalculator.mouth_aspect_ratio([(0, 0)] * 4), 0)

    def test_head_pose_recovers_projected_angles(self):
        camera = np.array([[640., 0, 320], [0, 640, 240], [0, 0, 1]])
        for angles in [(0, 0, 0), (40, 0, 0), (-40, 0, 0), (0, 35, 0),
                       (0, -35, 0), (0, 0, 20), (30, -15, 10)]:
            with self.subTest(angles=angles):
                pitch, yaw, roll = np.deg2rad(angles)
                rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                               [0, np.sin(pitch), np.cos(pitch)]])
                ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                               [-np.sin(yaw), 0, np.cos(yaw)]])
                rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                               [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
                rotation, _ = cv2.Rodrigues(rz @ ry @ rx)
                projected, _ = cv2.projectPoints(
                    MetricsCalculator.HEAD_MODEL, rotation, np.array([0., 0., 60.]),
                    camera, np.zeros(4),
                )
                points = np.zeros((468, 2))
                points[MetricsCalculator.HEAD_POSE_IDX] = projected[:, 0, :]
                np.testing.assert_allclose(
                    MetricsCalculator.estimate_head_pose(points, 640, 480), angles, atol=.1,
                )

    def test_head_tilt_is_a_sustained_attention_signal(self):
        for pitch, direction in [(40, 'вперёд'), (-40, 'назад')]:
            rules = FatigueRules(Config(overrides={'CALIBRATION_SECONDS': 0}))
            first = rules.update(FrameMetrics(detected=True, ear=.3, pitch=pitch, yaw=0), 0)
            self.assertEqual(first.reasons, [])
            sustained = rules.update(FrameMetrics(detected=True, ear=.3, pitch=pitch, yaw=0), .8)
            self.assertEqual(sustained.status, 'attention')
            self.assertTrue(any(direction in reason for reason in sustained.reasons))
            self.assertFalse(sustained.fatigue)

    def test_calibration_keeps_naturally_narrow_eyes_from_triggering_warning(self):
        rules = FatigueRules(Config(overrides={'CALIBRATION_SECONDS': 1}))
        for i in range(13):
            result = rules.update(FrameMetrics(detected=True, ear=.18, mar=.1, pitch=12, yaw=6, roll=0), i / 10)
        self.assertLess(rules.ear_threshold, .18)
        self.assertEqual(result.pitch, 0)
        self.assertEqual(result.status, 'tracking')
        for i in range(13, 19):
            result = rules.update(FrameMetrics(detected=True, ear=.07, mar=.1, pitch=12, yaw=6, roll=0), i / 10)
        self.assertEqual(result.status, 'fatigue')
        self.assertTrue(result.fatigue)

    def test_missing_pose_is_unknown(self):
        self.assertIsNone(MetricsCalculator.estimate_head_pose([(0, 0)] * 468, 640, 480))

    def test_head_pose_remains_upright_with_noisy_landmarks(self):
        camera = np.array([[640., 0, 320], [0, 640, 240], [0, 0, 1]])
        projected, _ = cv2.projectPoints(
            MetricsCalculator.HEAD_MODEL, np.zeros(3), np.array([5., -3., 60.]),
            camera, np.zeros(4),
        )
        rng = np.random.default_rng(42)
        for _ in range(30):
            points = np.zeros((468, 2))
            points[MetricsCalculator.HEAD_POSE_IDX] = projected[:, 0, :] + rng.normal(0, .8, (6, 2))
            pitch, yaw, roll = MetricsCalculator.estimate_head_pose(points, 640, 480)
            self.assertLess(max(abs(pitch), abs(yaw), abs(roll)), 10)

    def test_font_cache_and_hud(self):
        font = FontManager()
        self.assertIs(font._get_font(18), font._get_font(18))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hud = HUDRenderer(font)
        for detected in (False, True):
            rendered = hud.render(frame.copy(), FrameMetrics(detected=detected), 1, 30)
            self.assertEqual(rendered.shape, frame.shape)
            self.assertTrue(rendered.any())

    def test_preprocessing_matches_original(self):
        app = App.__new__(App)
        app.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame = np.random.default_rng(42).integers(0, 256, (80, 120, 3), dtype=np.uint8)
        light, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
        light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
        expected = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)
        expected = cv2.convertScaleAbs(expected, alpha=1.1, beta=10)
        np.testing.assert_array_equal(app.preprocess(frame), expected)
        np.testing.assert_array_equal(app.preprocess(frame), expected)

    def test_logs_and_charts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'output'
            csv_path = folder / 'metrics.csv'
            video = VideoLogger(str(csv_path))
            try:
                video.write(1, FrameMetrics(ear=0.3, mar=0.1))
                video.write(2, FrameMetrics(ear=0.2, fatigue=True))
            finally:
                video.close()
            with csv_path.open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row['Fatigue'] for row in rows], ['0', '1'])
            event_path = folder / 'events.txt'
            AlertSystem.log_event('проверка', str(event_path))
            self.assertIn('проверка', event_path.read_text())
            import matplotlib
            matplotlib.use('Agg')
            plot_metrics(str(csv_path), str(folder / 'charts'))
            self.assertEqual({p.name for p in (folder / 'charts').glob('*.png')},
                             {'EAR.png', 'MAR.png', 'PERCLOS.png', 'Fatigue.png'})

    def test_camera_loop_without_real_camera(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory, \
                patch('src.app.cv2.VideoCapture') as capture, \
                patch('src.app.FatigueDetector') as detector, \
                patch('src.app.VideoLogger') as video, \
                patch('src.app.plot_metrics') as plot, \
                patch('src.app.cv2.imshow'), patch('src.app.cv2.waitKey', return_value=27), \
                patch('src.app.cv2.destroyAllWindows'):
            capture.return_value.read.return_value = (True, frame)
            detector.return_value.process.return_value = FrameMetrics(detected=True)
            detector.return_value.should_alert.return_value = False
            app = App(str(Path(directory) / 'missing.yaml'))
            app.run()
            capture.return_value.release.assert_called_once()
            detector.return_value.close.assert_called_once()
            detector.return_value.drawing.draw_landmarks.assert_called_once()
            video.return_value.write.assert_called_once()
            video.return_value.close.assert_called_once()
            plot.assert_called_once()

    def test_detector_resets_when_face_disappears(self):
        with patch('src.detector.mp.solutions.face_mesh.FaceMesh') as mesh:
            mesh.return_value.process.return_value.multi_face_landmarks = None
            detector = FatigueDetector(Config(overrides={'CALIBRATION_SECONDS': 0}))
            try:
                detector.rules.temporal.update(0, True, True)
                detector.rules.temporal.update(0.5, True, True)
                result = detector.process(np.zeros((80, 120, 3), dtype=np.uint8))
                self.assertFalse(result.detected)
                self.assertEqual(detector.rules.temporal.eye_seconds, 0)
                self.assertFalse(detector.rules.temporal.ready)
            finally:
                detector.close()


if __name__ == '__main__':
    unittest.main()
