# Приложение: чтение config.yaml, захват камеры и основной цикл обработки кадров.
# Порядок работы: камера -> детектор -> интерфейс -> журнал и оповещение.

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import cv2
import numpy as np
import yaml

from src.detector import FatigueDetector
from src.interface import FontManager, HUDRenderer
from src.reporting import AlertSystem, VideoLogger, plot_metrics

logger = logging.getLogger(__name__)


class Config:
    # Настройки детектора: значения из YAML поверх DEFAULTS.
    #
    # Доступ к значениям через индексацию: ``cfg['EYE_AR_THRESH']``.

    DEFAULTS: dict[str, Any] = {
        "EYE_AR_THRESH": 0.25,
        "EYE_CLOSED_SECONDS": 0.4,
        "MAR_THRESH": 0.7,
        "YAWN_SECONDS": 0.5,
        "PERCLOS_WINDOW_SECONDS": 5.0,
        "PERCLOS_MIN_SECONDS": 5.0,
        "MAX_FRAME_GAP_SECONDS": 1.0,
        "PERCLOS_THRESH": 0.4,
        "ALERT_COOLDOWN": 5,
        "HEAD_PITCH_THRESH": 25,
        "HEAD_YAW_THRESH": 20,
        "CAMERA_INDEX": 0,
    }

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.data: dict[str, Any] = self._load(config_path)

    def _load(self, path: str) -> dict[str, Any]:
        # Читает YAML; при отсутствии файла или ошибке возвращает DEFAULTS.
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if loaded:
                    # Старые конфиги переводим при допущении исходных 30 FPS.
                    for old, new in [("EYE_AR_CONSEC_FRAMES", "EYE_CLOSED_SECONDS"),
                                     ("YAWN_CONSEC_FRAMES", "YAWN_SECONDS"),
                                     ("PERCLOS_WINDOW", "PERCLOS_WINDOW_SECONDS")]:
                        if old in loaded and new not in loaded:
                            loaded[new] = loaded[old] / 30.0
                            logger.warning("%s устарел; используем %s при 30 FPS", old, new)
                    if "PERCLOS_WINDOW" in loaded and "PERCLOS_MIN_SECONDS" not in loaded:
                        loaded["PERCLOS_MIN_SECONDS"] = loaded["PERCLOS_WINDOW_SECONDS"]
                    logger.info(f"конфиг загружен: {path}")
                    return {**self.DEFAULTS, **loaded}
            except Exception as e:
                logger.warning(f"не смог прочитать {path}: {e}")
        logger.info("используем настройки по умолчанию")
        return dict(self.DEFAULTS)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


class App:
    # Связывает все компоненты и крутит цикл обработки видео.

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg = Config(config_path)
        self.detector = FatigueDetector(self.cfg)
        self.video_log = VideoLogger()
        self.hud = HUDRenderer(FontManager())
        self.alert = AlertSystem()
        # Эти объекты одинаковы для всех кадров — создаём их один раз.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.landmark_style = self.detector.drawing.DrawingSpec((0, 255, 0), 1, 1)
        self.connection_style = self.detector.drawing.DrawingSpec((0, 0, 255), 1, 1)

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

                display = self.hud.render(frame, m, self.frame_id, self.fps)

                if m.detected:
                    self.detector.drawing.draw_landmarks(
                        display, m.landmarks, self.detector.connections,
                        self.landmark_style,
                        self.connection_style,
                    )

                cv2.imshow("детекция усталости", display)
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
