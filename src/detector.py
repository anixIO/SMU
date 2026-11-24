# Детектор: поиск лица через MediaPipe и проверка признаков усталости по метрикам.
# Пороговые значения берутся из config.yaml; формулы находятся в metrics.py.

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import cv2
import mediapipe as mp
import numpy as np

from src.metrics import FrameMetrics, MetricsCalculator, TemporalMetrics

if TYPE_CHECKING:
    from src.app import Config

logger = logging.getLogger(__name__)


class FatigueDetector:
    # Находит лицо, считает метрики и решает, есть ли признаки усталости.
    #
    # Решения по глазам и зевоте используют секунды непрерывного наблюдения.

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.metrics = MetricsCalculator()
        self.temporal = TemporalMetrics(config["PERCLOS_WINDOW_SECONDS"],
                                        config["PERCLOS_MIN_SECONDS"],
                                        config["MAX_FRAME_GAP_SECONDS"])

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawing = mp.solutions.drawing_utils
        self.connections = mp.solutions.face_mesh.FACEMESH_CONTOURS

        self.last_alarm = 0.0
        logger.info("Детектор инициализирован")

    def process(self, frame: np.ndarray) -> FrameMetrics:
        # Обрабатывает один кадр.
        #
        # Параметры:
        #     frame: кадр BGR.
        #
        # Результат:
        #     Метрики кадра; ``detected=False``, если лицо не найдено.
        now = time.monotonic()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        m = FrameMetrics()
        if not result.multi_face_landmarks:
            self.reset()
            return m

        landmarks = result.multi_face_landmarks[0]
        m.detected = True
        m.landmarks = landmarks
        # Дробные координаты сохраняют точность углов на уменьшенном кадре.
        pts = [(p.x * w, p.y * h) for p in landmarks.landmark]

        m.ear = self.metrics.compute_ear(pts)
        m.mar = self.metrics.compute_mar(pts)

        m.pitch, m.yaw, m.roll = MetricsCalculator.estimate_head_pose(pts, w, h)

        self.temporal.update(now, m.ear < self.cfg["EYE_AR_THRESH"],
                             m.mar > self.cfg["MAR_THRESH"])
        m.perclos = self.temporal.perclos
        m.perclos_ready = self.temporal.ready

        m.reasons = self.evaluate(m)
        m.fatigue = len(m.reasons) > 0
        return m

    def evaluate(self, m: FrameMetrics) -> list[str]:
        # Проверяет пороги и возвращает список сработавших причин.
        #
        # Использует длительность непрерывных событий и прогретый PERCLOS.
        reasons: list[str] = []
        if self.temporal.eye_seconds + 1e-9 >= self.cfg["EYE_CLOSED_SECONDS"]:
            reasons.append(f"глаза закрыты {self.temporal.eye_seconds:.1f} с")
        if self.temporal.yawn_seconds + 1e-9 >= self.cfg["YAWN_SECONDS"]:
            reasons.append(f"зевота {self.temporal.yawn_seconds:.1f} с")
        if m.perclos_ready and m.perclos > self.cfg["PERCLOS_THRESH"]:
            reasons.append(f"perclos {m.perclos:.2f}")
        if abs(m.pitch) > self.cfg["HEAD_PITCH_THRESH"]:
            direction = "вперёд" if m.pitch > 0 else "назад"
            reasons.append(f"наклон головы {direction} {abs(m.pitch):.0f}°")
        if abs(m.yaw) > self.cfg["HEAD_YAW_THRESH"]:
            reasons.append(f"поворот {m.yaw:.0f}")
        return reasons

    def should_alert(self) -> bool:
        # True, если с последнего оповещения прошло больше ALERT_COOLDOWN секунд.
        now = time.monotonic()
        if now - self.last_alarm > self.cfg["ALERT_COOLDOWN"]:
            self.last_alarm = now
            return True
        return False

    def reset(self) -> None:
        # Сбрасывает события и историю при потере лица.
        self.temporal.reset()

    def close(self) -> None:
        # Освобождает ресурсы MediaPipe.
        self.face_mesh.close()
