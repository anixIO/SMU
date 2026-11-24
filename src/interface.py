# Интерфейс: системный шрифт, русский текст, индикаторы и предупреждение на видеокадре.
# Для изменения надписей и внешнего вида откройте этот файл.

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.metrics import FrameMetrics

logger = logging.getLogger(__name__)


class FontManager:
    # Ищет системный TTF-шрифт и рисует им текст через PIL.
    #
    # ``cv2.putText`` не умеет кириллицу, поэтому текст рисуется через PIL.

    CANDIDATES: list[str] = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    def __init__(self) -> None:
        self.font_path: str | None = None
        self._find_font()

    def _find_font(self) -> None:
        # Выбирает первый существующий шрифт из CANDIDATES.
        for path in self.CANDIDATES:
            if os.path.exists(path):
                self.font_path = path
                logger.info(f"шрифт найден: {path}")
                return
        logger.warning("системный шрифт не найден, русский текст может отображаться криво")

    @lru_cache(maxsize=8)
    def _get_font(self, size: int):
        # Загружаем каждый размер один раз и используем на следующих кадрах.
        if self.font_path:
            return ImageFont.truetype(self.font_path, size)
        return ImageFont.load_default()

    def put_text(
        self,
        img: np.ndarray,
        text: str,
        position: tuple[int, int],
        font_size: int = 24,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        # Рисует текст на BGR-изображении.
        #
        # Параметры:
        #     img: кадр в формате BGR.
        #     text: строка (поддерживается кириллица).
        #     position: координаты левого верхнего угла текста.
        #     font_size: размер шрифта в пикселях.
        #     color: цвет в BGR.
        #
        # Результат:
        #     Новый кадр с текстом; при ошибке — исходный кадр.
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)

            draw.text(position, text, font=self._get_font(font_size), fill=color[::-1])
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"ошибка при рисовании текста: {e}")
            return img


class HUDRenderer:
    # Рисует полосы EAR/MAR/PERCLOS, углы головы, FPS и баннер усталости.

    def __init__(self, font_manager: FontManager) -> None:
        self.font = font_manager

    def _draw_bar(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        value: float,
        max_val: float,
        warn: float,
        crit: float,
    ) -> np.ndarray:
        # Рисует горизонтальную полосу-индикатор.
        #
        # Цвет: зелёный < warn ≤ жёлтый < crit ≤ красный (доли от max_val).
        ratio = min(value / max_val, 1.0) if max_val > 0 else 0
        if ratio >= crit:
            color = (0, 0, 255)
        elif ratio >= warn:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
        cv2.rectangle(img, (x, y), (x + int(w * ratio), y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
        return img

    def render(self, img: np.ndarray, m: FrameMetrics, frame_id: int, fps: float) -> np.ndarray:
        # Рисует HUD на кадре.
        #
        # Параметры:
        #     img: кадр BGR (изменяется на месте).
        #     m: метрики текущего кадра.
        #     frame_id: номер кадра.
        #     fps: текущий FPS.
        #
        # Результат:
        #     Кадр с HUD.
        h, w = img.shape[:2]

        if not m.detected:
            img = self.font.put_text(img, "Лицо не обнаружено",
                                     (w // 2 - 150, h // 2), 28, (0, 0, 255))
            cv2.putText(img, f"frame: {frame_id}  fps: {fps:.1f}",
                        (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            return img

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.rectangle(overlay, (0, 70), (280, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        y = 90
        img = self._draw_bar(img, 15, y, 240, 12, m.ear, 0.4, 0.2, 0.15)
        img = self.font.put_text(img, f"глаза (ear): {m.ear:.2f}", (15, y - 5), 18, (200, 200, 200))

        img = self._draw_bar(img, 15, y + 35, 240, 12, m.mar, 1.0, 0.5, 0.7)
        img = self.font.put_text(img, f"рот (mar): {m.mar:.2f}", (15, y + 30), 18, (200, 200, 200))

        img = self._draw_bar(img, 15, y + 70, 240, 12, m.perclos, 1.0, 0.3, 0.4)
        img = self.font.put_text(img, (f"perclos: {m.perclos:.2f}" if m.perclos_ready else "perclos: накопление данных"), (15, y + 65), 18, (200, 200, 200))

        img = self.font.put_text(img, f"наклон: {m.pitch:.0f}", (15, y + 110), 16, (200, 200, 200))
        img = self.font.put_text(img, f"поворот: {m.yaw:.0f}", (15, y + 140), 16, (200, 200, 200))
        img = self.font.put_text(img, f"крен: {m.roll:.0f}", (15, y + 170), 16, (200, 200, 200))

        cv2.putText(img, f"frame: {frame_id}", (w - 130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"fps: {fps:.1f}", (w - 130, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if m.fatigue:
            alpha = 0.5 + 0.3 * np.sin(time.time() * 10)
            cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            img = self.font.put_text(img, "Обнаружена усталость! сделайте перерыв",
                                     (w // 2 - 230, h - 48), 26, (255, 255, 255))

        cx, cy, r = w - 80, h - 80, 55
        angle = int(180 * min(m.perclos, 1.0))
        cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 180, (50, 50, 50), -1)
        if m.perclos < 0.3:
            circle_color = (0, 255, 0)
        elif m.perclos < 0.4:
            circle_color = (0, 255, 255)
        else:
            circle_color = (0, 0, 255)
        cv2.ellipse(img, (cx, cy), (r, r), 0, 0, angle, circle_color, -1)
        cv2.putText(img, f"{int(m.perclos * 100)}%", (cx - 23, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return img
