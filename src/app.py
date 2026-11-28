# Приложение: чтение config.yaml, захват камеры и основной цикл обработки кадров.
# Порядок работы: камера -> детектор -> интерфейс -> журнал и оповещение.

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import Config, profile_path
from src.detector import FatigueDetector
from src.interface import FontManager, HUDRenderer
from src.reporting import AlertSystem, VideoLogger, plot_metrics

logger = logging.getLogger(__name__)


class App:
    # Связывает все компоненты и крутит цикл обработки видео.

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg = Config(profile_path(Path.cwd()) if config_path == "config.yaml" else config_path)
        self.detector = FatigueDetector(self.cfg)
        self.video_log = VideoLogger()
        self.hud = HUDRenderer(FontManager())
        self.alert = AlertSystem()
        # Эти объекты одинаковы для всех кадров — создаём их один раз.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.landmark_style = self.detector.drawing.DrawingSpec((160, 211, 187), 1, 1)
        self.connection_style = self.detector.drawing.DrawingSpec((111, 166, 141), 1, 1)

        self.cap = cv2.VideoCapture(self.cfg["CAMERA_INDEX"])
        if not self.cap.isOpened():
            logger.error("не удалось открыть камеру")
            sys.exit(1)

        self.frame_id = 0
        self.fps = 30.0
        self._fps_count = 0
        self._fps_timer = time.monotonic()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Выравнивает освещённость: CLAHE по L-каналу LAB + небольшой контраст.
        #
        # Параметры:
        #     frame: кадр BGR.
        #
        # Результат:
        #     Обработанный кадр; при ошибке — исходный.
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            return cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        except Exception:
            return frame

    def _update_fps(self) -> None:
        # Пересчитывает FPS раз в секунду.
        self._fps_count += 1
        if time.monotonic() - self._fps_timer >= 1.0:
            self.fps = self._fps_count
            self._fps_count = 0
            self._fps_timer = time.monotonic()

    def run(self) -> None:
        # Основной цикл; выход по Esc или Ctrl+C.
        logger.info("система запущена, нажми esc для выхода")
        try:
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    logger.error("камера перестала отдавать кадры")
                    break

                frame = cv2.flip(frame, 1)
                self.frame_id += 1
                self._update_fps()

                frame = self.preprocess(frame)
                m = self.detector.process(frame)
                self.video_log.write(self.frame_id, m)

                if m.fatigue and self.detector.should_alert():
                    reason = "; ".join(m.reasons)
                    logger.warning(f"усталость: {reason}")
                    self.alert.log_event(reason)
                    self.alert.play_sound()

                if m.detected:
                    self.detector.drawing.draw_landmarks(
                        frame, m.landmarks, self.detector.connections,
                        self.landmark_style,
                        self.connection_style,
                    )

                display = self.hud.render(frame, m, self.frame_id, self.fps)
                cv2.imshow("СМУ", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        except KeyboardInterrupt:
            logger.info("остановлено пользователем")
        except Exception as e:
            logger.error(f"ошибка в основном цикле: {e}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        # Освобождает камеру, закрывает логи и строит графики.
        logger.info("завершаю работу")
        self.cap.release()
        self.video_log.close()
        self.detector.close()
        cv2.destroyAllWindows()
        plot_metrics()
        logger.info("готово")
