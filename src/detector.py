"""Оценка признаков сонливости по движению лица и длительности событий."""

from __future__ import annotations

import time
from statistics import median

import cv2
import mediapipe as mp
import numpy as np

from src.config import Config
from src.metrics import FrameMetrics, MetricsCalculator, TemporalMetrics


class FatigueRules:
    """Правила работают и в реальном времени, и при повторе записанного видео."""

    def __init__(self, config: Config):
        self.cfg = config
        self.temporal = TemporalMetrics(config["PERCLOS_WINDOW_SECONDS"],
                                        config["PERCLOS_MIN_SECONDS"],
                                        config["MAX_FRAME_GAP_SECONDS"])
        self.calibration_start = None
        self.calibration_duration = config["CALIBRATION_SECONDS"]
        self.calibration_done = not self.calibration_duration
        self.samples = []
        self.ear_threshold = config["EYE_AR_THRESH"]
        self.neutral_pitch = self.neutral_yaw = 0.0
        self.last_frame = None
        self.head_since = None
        self.closed = False

    def reset_events(self):
        self.temporal.reset()
        self.head_since = None
        self.closed = False

    def calibrate(self, m: FrameMetrics, now: float):
        if self.calibration_start is None:
            self.calibration_start = now
        if now - self.calibration_start >= self.calibration_duration + 5:
            self.calibration_done = True
            self.samples.clear()
            return
        valid_pose = (m.pitch is not None and m.yaw is not None and m.roll is not None
                      and abs(m.pitch) < 45 and abs(m.yaw) < 28 and abs(m.roll) < 25)
        if not (m.detected and valid_pose and 0.15 <= m.ear <= 0.55 and m.mar < self.cfg["MAR_THRESH"]):
            # Моргания не портят профиль. При смене положения ждём нового ряда отсчётов.
            if not m.detected or not valid_pose:
                self.samples.clear()
            return
        if self.samples and now - self.samples[-1][0] > self.cfg["MAX_FRAME_GAP_SECONDS"]:
            self.samples.clear()
        self.samples.append((now, m.ear, m.pitch, m.yaw))
        elapsed = now - self.samples[0][0]
        m.calibration = min(elapsed / self.calibration_duration, 1.0)
        if elapsed >= self.calibration_duration and len(self.samples) >= 6:
            _, eyes, pitches, yaws = zip(*self.samples)
            eye_baseline = median(eyes)
            if median(abs(ear - eye_baseline) for ear in eyes) > 0.045:
                self.samples.clear()
                m.calibration = 0.0
                return
            # Границы ограничивают влияние ошибок трекинга при калибровке.
            calibrated = eye_baseline * self.cfg["EAR_CLOSED_RATIO"]
            self.ear_threshold = min(0.32, max(0.11, calibrated))
            self.neutral_pitch = median(pitches)
            self.neutral_yaw = median(yaws)
            self.calibration_done = True
            self.samples.clear()

    def update(self, m: FrameMetrics, now: float) -> FrameMetrics:
        if not self.calibration_done:
            m.calibration = 0.0
            self.calibrate(m, now)
            if not self.calibration_done:
                m.status = "calibrating" if m.detected else "no_face"
                return m
            m.calibration = None
        if not m.detected:
            self.reset_events()
            self.last_frame = None
            m.status = "no_face"
            return m

        if self.last_frame is not None and now - self.last_frame > self.cfg["MAX_FRAME_GAP_SECONDS"]:
            self.reset_events()
        self.last_frame = now
        # При повторном открытии глаз требуем чуть больший EAR, чем при закрытии.
        self.closed = m.ear < self.ear_threshold * (1.08 if self.closed else 1)
        self.temporal.update(now, self.closed, m.mar > self.cfg["MAR_THRESH"])
        m.perclos = self.temporal.perclos
        m.perclos_ready = self.temporal.ready

        if self.temporal.eye_seconds + 1e-9 >= self.cfg["EYE_CLOSED_SECONDS"]:
            m.reasons.append(f"Глаза закрыты {self.temporal.eye_seconds:.1f} с")
        if m.perclos_ready and m.perclos > self.cfg["PERCLOS_THRESH"]:
            m.reasons.append(f"Закрытые глаза: {m.perclos:.0%} времени")
        m.fatigue = bool(m.reasons)
        if m.fatigue:
            m.status = "fatigue"

        direction = None
        if m.pitch is not None and m.yaw is not None:
            m.pitch -= self.neutral_pitch
            m.yaw -= self.neutral_yaw
            if abs(m.yaw) > self.cfg["HEAD_YAW_THRESH"]:
                direction = "Голова повёрнута в сторону"
            elif abs(m.pitch) > self.cfg["HEAD_PITCH_THRESH"]:
                direction = "Голова наклонена вперёд" if m.pitch > 0 else "Голова наклонена назад"
        if direction:
            if self.head_since is None:
                self.head_since = now
            if now - self.head_since + 1e-9 >= self.cfg["HEAD_AWAY_SECONDS"]:
                m.reasons.append(direction)
        else:
            self.head_since = None
        if self.temporal.yawn_seconds + 1e-9 >= self.cfg["YAWN_SECONDS"]:
            m.reasons.append("Возможный зевок")
        if m.reasons and not m.fatigue:
            m.status = "attention"
        return m


class FatigueDetector:
    def __init__(self, config: Config):
        self.cfg = config
        self.rules = FatigueRules(config)
        self.metrics = MetricsCalculator()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawing = mp.solutions.drawing_utils
        self.connections = mp.solutions.face_mesh.FACEMESH_CONTOURS
        self.last_alarm = float("-inf")

    def measure(self, image: np.ndarray, *, rgb: bool = False) -> FrameMetrics:
        h, w = image.shape[:2]
        result = self.face_mesh.process(image if rgb else cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return FrameMetrics()
        landmarks = result.multi_face_landmarks[0]
        points = [(point.x * w, point.y * h) for point in landmarks.landmark]
        i, j = self.metrics.HEAD_POSE_IDX[2:4]
        if np.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]) < min(w, h) * 0.075:
            return FrameMetrics()
        face_points = (self.metrics.LEFT_EYE_IDX + self.metrics.RIGHT_EYE_IDX + self.metrics.MOUTH_IDX)
        if any(not (0 <= points[i][0] <= w and 0 <= points[i][1] <= h) for i in face_points):
            return FrameMetrics()

        m = FrameMetrics(detected=True, landmarks=landmarks,
                         ear=self.metrics.compute_ear(points), mar=self.metrics.compute_mar(points))
        pose = self.metrics.estimate_head_pose(points, w, h)
        if pose is not None:
            m.pitch, m.yaw, m.roll = pose
        return m

    def process(self, image: np.ndarray, *, rgb: bool = False, now: float | None = None) -> FrameMetrics:
        timestamp = time.monotonic() if now is None else now
        return self.rules.update(self.measure(image, rgb=rgb), timestamp)

    def should_alert(self) -> bool:
        now = time.monotonic()
        if now - self.last_alarm >= self.cfg["ALERT_COOLDOWN"]:
            self.last_alarm = now
            return True
        return False

    def close(self):
        self.face_mesh.close()
