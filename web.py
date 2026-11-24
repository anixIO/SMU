# Django-интерфейс системы контроля усталости. Запуск: .venv/bin/python web.py
# Дизайн страницы находится в templates/design.html. main.py запускает прежнее окно OpenCV.

from __future__ import annotations

import atexit
import logging
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

import django
from django.conf import settings

BASE_DIR = Path(__file__).resolve().parent


if not settings.configured:
    settings.configure(
        DEBUG=False,
        SECRET_KEY=secrets.token_urlsafe(48),
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=["127.0.0.1", "localhost", "[::1]"],
        MIDDLEWARE=["django.middleware.csrf.CsrfViewMiddleware"],
        TEMPLATES=[{
            "BACKEND": "django.template.backends.django.DjangoTemplates",
            "DIRS": [BASE_DIR / "templates"],
            "APP_DIRS": False,
        }],
        DATA_UPLOAD_MAX_MEMORY_SIZE=1_000_000,
        LANGUAGE_CODE="ru",
        USE_TZ=True,
    )
    django.setup()

from django.core.management import execute_from_command_line
from django.http import FileResponse, JsonResponse
from django.shortcuts import render
from django.urls import path, re_path
from django.views.decorators.http import require_GET, require_POST
from django.views.static import serve

logger = logging.getLogger(__name__)


class Monitor:
    # ponytail: один пользователь и один поток; для нескольких пользователей нужны отдельные сеансы.

    def __init__(self):
        self.detector = None
        self.video_log = None
        self.csv_path = None
        self.frames = 0
        self.started = 0.0

    def close(self):
        """Повторный вызов безопасен, в том числе после неудачного запуска."""
        if self.video_log is not None:
            self.video_log.close()
            self.video_log = None
        if self.detector is not None:
            self.detector.close()
            self.detector = None


monitor = Monitor()
atexit.register(monitor.close)


@require_GET
def dashboard(request):
    from src.app import Config
    config = Config(str(BASE_DIR / "config.yaml"))
    cards = [
        {"id": "ear", "title": "Открытость глаз", "unit": "EAR", "hint": f'Порог закрытия: {config["EYE_AR_THRESH"]}'},
        {"id": "mar", "title": "Открытость рта", "unit": "MAR", "hint": f'Порог зевоты: {config["MAR_THRESH"]}'},
        {"id": "perclos", "title": "Закрытые глаза", "unit": "%", "hint": f'Окно наблюдения: {config["PERCLOS_WINDOW_SECONDS"]:g} с'},
    ]
    return render(request, "design.html", {
        "cards": cards,
        "pitch_hint": f'Порог наклона вперёд/назад: {config["HEAD_PITCH_THRESH"]:g}°',
        "has_report": monitor.csv_path is not None and monitor.csv_path.is_file(),
    })


@require_POST
def start_session(request):
    from src.app import Config
    from src.detector import FatigueDetector
    from src.reporting import VideoLogger
    monitor.close()
    try:
        monitor.detector = FatigueDetector(Config(str(BASE_DIR / "config.yaml")))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        monitor.csv_path = BASE_DIR / "output" / f"web_{stamp}.csv"
        monitor.video_log = VideoLogger(str(monitor.csv_path))
        if monitor.video_log.file is None:
            raise OSError("Не удалось открыть CSV-файл")
        monitor.frames = 0
        monitor.started = time.monotonic()
    except Exception:
        monitor.close()
        logger.exception("Не удалось запустить веб-сеанс")
        return JsonResponse({"error": "Не удалось запустить MediaPipe. Подробности — в терминале сервера."}, status=503)
    return JsonResponse({"active": True})


@require_POST
def analyze_frame(request):
    if monitor.detector is None:
        return JsonResponse({"error": "Сначала начните наблюдение."}, status=409)
    upload = request.FILES.get("frame")
    if upload is None or upload.size > 800_000:
        return JsonResponse({"error": "Нужен JPEG-кадр размером до 800 КБ."}, status=400)
    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(upload) as image:
            if image.format != "JPEG" or not (0 < image.width <= 640 and 0 < image.height <= 480):
                return JsonResponse({"error": "Ожидается JPEG не больше 640 × 480."}, status=400)
            frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return JsonResponse({"error": "Изображение повреждено или имеет неверный формат."}, status=400)
    try:
        metrics = monitor.detector.process(frame)
        monitor.frames += 1
        monitor.video_log.write(monitor.frames, metrics)
        event = None
        if metrics.fatigue and monitor.detector.should_alert():
            from src.reporting import AlertSystem
            event = {"time": datetime.now().strftime("%H:%M:%S"), "text": "; ".join(metrics.reasons)}
            AlertSystem.log_event(event["text"], str(BASE_DIR / "output" / "fatigue_log.txt"))
        values = {name: getattr(metrics, name) for name in (
            "detected", "ear", "mar", "perclos", "perclos_ready", "pitch", "yaw", "roll", "fatigue", "reasons"
        )}
        values.update(event=event, frames=monitor.frames,
                      fps=round(monitor.frames / max(time.monotonic() - monitor.started, 0.001), 1))
        return JsonResponse(values)
    except Exception:
        logger.exception("Ошибка анализа кадра")
        return JsonResponse({"error": "Не удалось обработать кадр. Подробности — в терминале сервера."}, status=500)


@require_POST
def stop_session(request):
    monitor.close()
    return JsonResponse({"active": False})


@require_GET
def download_report(request):
    if monitor.csv_path is None or not monitor.csv_path.is_file():
        return JsonResponse({"error": "Отчёт появится после начала наблюдения."}, status=404)
    return FileResponse(monitor.csv_path.open("rb"), as_attachment=True, filename=monitor.csv_path.name)


urlpatterns = [
    # Для локального запуска отдаём только два файла интерфейса.
    re_path(r"^assets/(?P<path>monitor\.(?:css|js))$", serve,
            {"document_root": BASE_DIR / "static"}, name="asset"),
    path("", dashboard, name="dashboard"),
    path("api/start/", start_session, name="start"),
    path("api/frame/", analyze_frame, name="frame"),
    path("api/stop/", stop_session, name="stop"),
    path("report/", download_report, name="report"),
]


if __name__ == "__main__":
    arguments = sys.argv[1:] or ["runserver", "127.0.0.1:8000"]
    if arguments[0] == "runserver":
        # Один процесс и один поток сохраняют общий сеанс и порядок кадров MediaPipe.
        arguments += ["--noreload", "--nothreading"]
    execute_from_command_line([sys.argv[0], *arguments])
