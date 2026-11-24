# Расчёты: данные одного кадра, длительность закрытия глаз и зевоты, EAR/MAR и углы головы.
# Здесь находятся формулы; камера и интерфейс в расчётах не участвуют.

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import dist
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)
Point = tuple[float, float]


@dataclass
class FrameMetrics:
    # Метрики, посчитанные для одного кадра.
    #
    # Поля:
    #     detected: найдено ли лицо в кадре.
    #     ear: eye aspect ratio, среднее по двум глазам.
    #     mar: mouth aspect ratio.
    #     perclos: доля времени с закрытыми глазами в скользящем окне.
    #     pitch: наклон головы вперёд/назад, градусы.
    #     yaw: поворот головы влево/вправо, градусы.
    #     roll: крен головы, градусы.
    #     fatigue: сработал ли хотя бы один признак усталости.
    #     reasons: текстовые причины срабатывания.
    #     landmarks: сырые landmarks MediaPipe (для отрисовки сетки).

    detected: bool = False
    ear: float = 0.0
    mar: float = 0.0
    perclos: float = 0.0
    perclos_ready: bool = False
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    fatigue: bool = False
    reasons: list[str] = field(default_factory=list)
    landmarks: Any = None


class TemporalMetrics:
    def __init__(self, window: float, min_observation: float, max_gap: float = 1.0):
        if not 0 < min_observation <= window or max_gap <= 0:
            raise ValueError("Нужно 0 < PERCLOS_MIN_SECONDS <= PERCLOS_WINDOW_SECONDS и MAX_FRAME_GAP_SECONDS > 0")
        self.window = window
        self.min_observation = min_observation
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.intervals = deque()
        self.previous = None
        self.eye_since = None
        self.yawn_since = None
        self.eye_seconds = self.yawn_seconds = self.perclos = 0.0
        self.ready = False

    def update(self, now: float, closed: bool, yawning: bool):
        if self.previous is not None:
            last, was_closed = self.previous
            if now < last:
                raise ValueError("Время наблюдений должно возрастать")
            if now - last > self.max_gap:
                self.reset()
            elif now > last:
                self.intervals.append((last, now, was_closed))
        # Запоминаем начало непрерывного закрытия глаз и зевоты.
        if not closed:
            self.eye_since = None
        elif self.eye_since is None:
            self.eye_since = now
        if not yawning:
            self.yawn_since = None
        elif self.yawn_since is None:
            self.yawn_since = now
        self.eye_seconds = now - self.eye_since if self.eye_since is not None else 0.0
        self.yawn_seconds = now - self.yawn_since if self.yawn_since is not None else 0.0
        self.previous = (now, closed)
        cutoff = now - self.window
        # Удаляем старые интервалы; оставшиеся взвешиваем по времени, а не по FPS.
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        total = shut = 0.0
        for start, end, state in self.intervals:
            duration = end - max(start, cutoff)
            total += duration
            shut += duration * state
        self.perclos = shut / total if total else 0.0
        self.ready = total + 1e-9 >= self.min_observation


class MetricsCalculator:
    # Считает метрики по 468/478 landmarks MediaPipe FaceMesh.
    #
    # Индексы точек соответствуют топологии MediaPipe FaceMesh.

    LEFT_EYE_IDX: list[int] = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX: list[int] = [362, 385, 387, 263, 373, 380]
    MOUTH_IDX: list[int] = [61, 291, 81, 178]
    HEAD_POSE_IDX = [1, 152, 33, 263, 61, 291]
    # MediaPipe canonical_face_model.obj: нос, подбородок, уголки глаз и рта.
    # https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
    # Оси приведены к OpenCV: Y вниз, Z от камеры; анфас соответствует нулевым углам.
    HEAD_MODEL = np.array([
        (0.000000, 1.126865, -7.475604),
        (0.000000, 9.403378, -4.264492),
        (-4.445859, -2.663991, -3.173422),
        (4.445859, -2.663991, -3.173422),
        (-2.456206, 4.342621, -4.283884),
        (2.456206, 4.342621, -4.283884),
    ], dtype=np.float64)

    @staticmethod
    def eye_aspect_ratio(eye: Sequence[Point]) -> float:
        # EAR по 6 точкам глаза (Soukupová & Čech, 2016).
        #
        # Параметры:
        #     eye: точки p1..p6, где p1/p4 — уголки глаза.
        #
        # Результат:
        #     (|p2-p6| + |p3-p5|) / (2·|p1-p4|); 0.0 при вырожденном глазе.
        a = dist(eye[1], eye[5])
        b = dist(eye[2], eye[4])
        c = dist(eye[0], eye[3])
        return (a + b) / (2.0 * c) if c > 0 else 0.0

    @staticmethod
    def mouth_aspect_ratio(mouth: Sequence[Point]) -> float:
        # MAR по 4 точкам: [левый угол, правый угол, верх, низ].
        #
        # Результат:
        #     высота / ширина рта; 0.0 при нулевой ширине.
        width = dist(mouth[0], mouth[1])
        height = dist(mouth[2], mouth[3])
        return height / width if width > 0 else 0.0

    def compute_ear(self, pts: Sequence[Point]) -> float:
        # Средний EAR двух глаз по полному списку landmarks.
        left = self.eye_aspect_ratio([pts[i] for i in self.LEFT_EYE_IDX])
        right = self.eye_aspect_ratio([pts[i] for i in self.RIGHT_EYE_IDX])
        return (left + right) / 2.0

    def compute_mar(self, pts: Sequence[Point]) -> float:
        # MAR по полному списку landmarks.
        return self.mouth_aspect_ratio([pts[i] for i in self.MOUTH_IDX])

    @staticmethod
    def estimate_head_pose(pts: Sequence[Point], w: int, h: int) -> tuple[float, float, float]:
        """Возвращает (pitch, yaw, roll) в градусах; pitch > 0 при наклоне вперёд."""
        pts_2d = np.array(
            [(pts[i][0], pts[i][1]) for i in MetricsCalculator.HEAD_POSE_IDX],
            dtype=np.float64,
        )
        focal = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        try:
            eye_distance = np.linalg.norm(pts_2d[2] - pts_2d[3])
            if eye_distance < 1:
                return 0.0, 0.0, 0.0
            depth = focal * np.linalg.norm(
                MetricsCalculator.HEAD_MODEL[2] - MetricsCalculator.HEAD_MODEL[3]
            ) / eye_distance
            image_center = pts_2d.mean(axis=0)
            model_center = MetricsCalculator.HEAD_MODEL.mean(axis=0)
            translation = np.array([
                (image_center[0] - center[0]) * depth / focal,
                (image_center[1] - center[1]) * depth / focal,
                depth,
            ]) - model_center
            # Начальное положение перед камерой исключает зеркальное решение за ней.
            success, rvec, translation = cv2.solvePnP(
                MetricsCalculator.HEAD_MODEL, pts_2d, camera_matrix, dist_coeffs,
                rvec=np.zeros((3, 1)), tvec=translation.reshape(3, 1),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success or translation[2, 0] <= 0:
                return 0.0, 0.0, 0.0
            rmat, _ = cv2.Rodrigues(rvec)
            angles = cv2.RQDecomp3x3(rmat)[0]
            return tuple(float(angle) for angle in angles)
        except cv2.error as e:
            logger.debug(f"solvepnp не сошёлся: {e}")
            return 0.0, 0.0, 0.0
