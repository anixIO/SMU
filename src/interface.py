"""Оконный интерфейс СМУ поверх живого видео."""

import logging
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.metrics import FrameMetrics

logger = logging.getLogger(__name__)


class FontManager:
    CANDIDATES = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    def __init__(self):
        self.font_path = next((path for path in self.CANDIDATES if os.path.exists(path)), None)
        if self.font_path is None:
            logger.warning("Не найден системный шрифт с поддержкой кириллицы.")

    @lru_cache(maxsize=8)
    def _get_font(self, size):
        return ImageFont.truetype(self.font_path, size) if self.font_path else ImageFont.load_default(size=size)

    @lru_cache(maxsize=128)
    def label(self, content, size, color):
        font = self._get_font(size)
        left, top, right, bottom = font.getbbox(content)
        patch = Image.new("RGBA", (max(1, right - left), max(1, bottom - top)))
        ImageDraw.Draw(patch).text((-left, -top), content, font=font, fill=color)
        return patch, (left, top)


class HUDRenderer:
    def __init__(self, font_manager: FontManager):
        self.font = font_manager

    def render(self, img: np.ndarray, m: FrameMetrics, frame_id: int, fps: float) -> np.ndarray:
        width = img.shape[1]
        scale = min(1.2, max(.55, width / 640))
        status = {
            "no_face": ("Лицо вне кадра", "Посмотрите прямо в камеру"),
            "calibrating": ("Калибровка", f"Смотрите прямо · {m.calibration or 0:.0%}"),
            "attention": ("Обратите внимание", m.reasons[0] if m.reasons else ""),
            "fatigue": ("Сделайте перерыв", m.reasons[0] if m.reasons else ""),
        }
        title, subtitle = status.get(m.status, ("Признаков сонливости нет", "Наблюдение продолжается"))
        if not m.detected:
            title, subtitle = status["no_face"]
        hue = {"fatigue": "#f5ae9c", "attention": "#f5d595"}.get(m.status, "#c6f2de")

        # Все подписи рисуются за одно преобразование кадра в RGB.
        frame = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame, "RGBA")

        def text(x, y, content, size, color):
            patch, (left, top) = self.font.label(content, max(1, round(size * scale)), color)
            frame.paste(patch, (round(x * scale) + left, round(y * scale) + top), patch)

        draw.rounded_rectangle((8, 8, max(9, width - 8), round(119 * scale)),
                               radius=10, fill=(18, 37, 30, 230))
        text(28, 21, "СМУ  /  НАБЛЮДЕНИЕ", 13, "#c3d7cd")
        text(28, 44, title, 22, hue)
        text(28, 75, subtitle, 14, "#eaf6ee")
        if width >= 360:
            text(527, 22, f"{fps:.1f} кадр/с", 13, "#c3d7cd")
        if m.detected:
            perclos = f"{m.perclos:.0%}" if m.perclos_ready else "—"
            text(28, 97, f"EAR  {m.ear:.2f}    MAR  {m.mar:.2f}    PERCLOS  {perclos}", 12, "#c3d7cd")
        return cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
