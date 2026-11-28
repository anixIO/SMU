"""Веб-приложение СМУ. Запуск: .venv/bin/python web.py."""

from __future__ import annotations

import logging
import os
import secrets
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import django
from django.conf import settings

BASE_DIR = Path(__file__).resolve().parent


def tunnel_origin(value):
    if not value:
        return "", None
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError as error:
        raise SystemExit("Укажите корректную HTTPS-ссылку в SMU_TUNNEL_URL.") from error
    if (url.scheme != "https" or not url.hostname or url.username is not None
            or url.path not in ("", "/") or url.query or url.fragment or "*" in url.netloc):
        raise SystemExit("Укажите HTTPS-ссылку без пути в SMU_TUNNEL_URL.")
    host = f"[{url.hostname}]" if ":" in url.hostname else url.hostname
    origin = f"https://{host}" + (f":{port}" if port and port != 443 else "")
    return origin, host


if not settings.configured:
    origin, host = tunnel_origin(os.environ.get("SMU_TUNNEL_URL", "").strip())
    data_dir = Path(os.environ.get("SMU_DATA_DIR", BASE_DIR / "output")).resolve()
    settings.configure(
        DEBUG=False,
        SECRET_KEY=os.environ.get("SMU_SECRET_KEY") or secrets.token_urlsafe(48),
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=["127.0.0.1", "localhost", "[::1]"] + ([host] if host else []),
        CSRF_TRUSTED_ORIGINS=[origin] if origin else [],
        SMU_CONFIG_DIR=BASE_DIR,
        SMU_DATA_DIR=data_dir,
        INSTALLED_APPS=["django.contrib.contenttypes", "django.contrib.auth",
                        "django.contrib.sessions", "src"],
        DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": data_dir / "smu.sqlite3",
                                    "TEST": {"NAME": ":memory:"}}},
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
        AUTH_PASSWORD_VALIDATORS=[
            {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
            {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
            {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
        ],
        MIDDLEWARE=[
            "django.middleware.common.CommonMiddleware",
            f"{__name__}.limit_camera_upload",
            "django.contrib.sessions.middleware.SessionMiddleware",
            "django.middleware.csrf.CsrfViewMiddleware",
            "django.contrib.auth.middleware.AuthenticationMiddleware",
        ],
        TEMPLATES=[{
            "BACKEND": "django.template.backends.django.DjangoTemplates",
            "DIRS": [BASE_DIR / "templates"],
            "APP_DIRS": False,
        }],
        DATA_UPLOAD_MAX_MEMORY_SIZE=1_000_000,
        FILE_UPLOAD_MAX_MEMORY_SIZE=1_000_000,
        LANGUAGE_CODE="ru",
        TIME_ZONE=os.environ.get("SMU_TIME_ZONE", "Asia/Yekaterinburg"),
        USE_TZ=True,
    )
    django.setup()

from django.core.management import call_command, execute_from_command_line
from django.db.models import F
from django.http import Http404, JsonResponse
from django.shortcuts import render
from django.urls import path, re_path
from django.utils import timezone
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_GET, require_POST, require_safe
from django.views.static import serve

from src.config import Config, profile_path
from src.models import Observation, WarningEvent
from src.sessions import ACTIVE_LIMIT, Monitor, close_monitor, expire_monitors, monitors
from src.staff import is_local, open_report, staff_urls

logger = logging.getLogger(__name__)


def limit_camera_upload(get_response):
    def middleware(request):
        length = request.headers.get("Content-Length", "")
        if request.path_info == "/api/frame/" and length.isascii() and length.isdigit() and (len(length) > 12 or int(length) > 900_000):
            return JsonResponse({"error": "Кадр превышает допустимый размер."}, status=400)
        return get_response(request)

    return middleware


def current_observation(request):
    value = request.session.get("observation_id")
    return UUID(value) if value else None


@never_cache
@require_safe
def dashboard(request):
    expire_monitors()
    observation_id = current_observation(request)
    has_report = bool(observation_id and
                      Observation.objects.filter(pk=observation_id, ended_at__isnull=False, frames__gt=0).exists())
    try:
        calibration_enabled = bool(Config(profile_path(settings.SMU_CONFIG_DIR))["CALIBRATION_SECONDS"])
        config_error = ""
    except ValueError as error:
        calibration_enabled = False
        config_error = str(error)
    return render(request, "design.html", {
        "has_report": has_report, "show_admin": is_local(request),
        "config_error": config_error, "calibration_enabled": calibration_enabled,
    })


@require_POST
def start_session(request):
    from src.detector import FatigueDetector
    from src.reporting import VideoLogger

    expire_monitors()
    previous_id = current_observation(request)
    if len(monitors) >= ACTIVE_LIMIT and previous_id not in monitors:
        return JsonResponse({"error": "Все сеансы заняты. Попробуйте немного позже."}, status=503)
    try:
        config = Config(profile_path(settings.SMU_CONFIG_DIR))
    except ValueError as error:
        return JsonResponse({"error": str(error)}, status=503)

    close_monitor(previous_id)
    detector = video_log = None
    try:
        detector = FatigueDetector(config)
        entry = Observation()
        video_log = VideoLogger(str(settings.SMU_DATA_DIR / entry.report_name))
        if video_log.file is None:
            raise OSError("Не удалось открыть CSV-файл")
        entry.save(force_insert=True)
    except Exception:
        if video_log:
            video_log.close()
        if detector:
            detector.close()
        logger.exception("Не удалось запустить веб-сеанс")
        return JsonResponse({"error": "Не удалось запустить анализ. Проверьте терминал сервера."}, status=503)

    monitors[entry.pk] = Monitor(entry, detector, video_log)
    request.session["observation_id"] = str(entry.pk)
    return JsonResponse({"active": True, "observation": str(entry.pk)})


def request_monitor(request):
    observation_id = current_observation(request)
    expected = request.headers.get("X-Observation") or request.POST.get("observation")
    if expected and expected != str(observation_id):
        return None
    expire_monitors()
    return monitors.get(observation_id)


@require_POST
def analyze_frame(request):
    monitor = request_monitor(request)
    if monitor is None:
        return JsonResponse({"error": "Сеанс завершился. Начните наблюдение снова."}, status=409)
    upload = request.FILES.get("frame")
    if upload is None or upload.size > 800_000:
        return JsonResponse({"error": "Нужен JPEG-кадр размером до 800 КБ."}, status=400)

    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(upload) as image:
            if image.format != "JPEG" or not (0 < image.width <= 640 and 0 < image.height <= 480):
                return JsonResponse({"error": "Ожидается JPEG не больше 640 × 480."}, status=400)
            frame = np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return JsonResponse({"error": "Изображение повреждено или имеет неверный формат."}, status=400)
    try:
        metrics = monitor.detector.process(frame, rgb=True)
        monitor.record(metrics)
        event = None
        if metrics.fatigue and monitor.detector.should_alert():
            warning = WarningEvent.objects.create(observation=monitor.observation, reason="; ".join(metrics.reasons))
            event = {"time": timezone.localtime(warning.created_at).strftime("%H:%M:%S"), "text": warning.reason}
        values = {name: getattr(metrics, name) for name in (
            "detected", "ear", "mar", "perclos", "perclos_ready", "pitch", "yaw", "roll",
            "fatigue", "reasons", "status", "calibration",
        )}
        entry = monitor.observation
        values.update(event=event, frames=entry.frames, face_frames=entry.face_frames, seconds=entry.seconds,
                      fps=round(entry.frames / max(time.monotonic() - monitor.started, 0.001), 1))
        return JsonResponse(values)
    except Exception:
        logger.exception("Ошибка анализа кадра")
        return JsonResponse({"error": "Не удалось обработать кадр. Подробности в терминале сервера."}, status=500)


@require_POST
def stop_session(request):
    monitor = request_monitor(request)
    if monitor:
        close_monitor(monitor.observation.pk)
    entry = Observation.objects.filter(pk=current_observation(request), ended_at__isnull=False).first()
    return JsonResponse({"active": False, "has_report": bool(entry and entry.frames)})


@never_cache
@require_GET
def download_report(request):
    entry = Observation.objects.filter(pk=current_observation(request), ended_at__isnull=False, frames__gt=0).first()
    if entry is None:
        raise Http404
    return open_report(entry.report_name)


urlpatterns = [
    re_path(r"^assets/(?P<path>monitor\.(?:css|js))$", serve,
            {"document_root": BASE_DIR / "static"}, name="asset"),
    path("", dashboard, name="dashboard"),
    path("api/start/", start_session, name="start"),
    path("api/frame/", analyze_frame, name="frame"),
    path("api/stop/", stop_session, name="stop"),
    path("report/", download_report, name="report"),
    *staff_urls,
]


if __name__ == "__main__":
    arguments = sys.argv[1:] or ["runserver", "127.0.0.1:8000"]
    settings.SMU_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if arguments[0] in {"runserver", "createsuperuser"}:
        call_command("migrate", interactive=False, verbosity=0)
    if arguments[0] == "runserver":
        Observation.objects.filter(ended_at__isnull=True).update(ended_at=F("last_frame_at"))
        arguments += ["--noreload", "--nothreading"]
    execute_from_command_line([sys.argv[0], *arguments])
