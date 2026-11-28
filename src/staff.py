"""Локальный дашборд с учётными записями Django и настройками детектора."""

import os
import re
from datetime import datetime, timedelta
from functools import wraps
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import yaml
from django import forms
from django.conf import settings
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.views import LoginView, LogoutView
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db.models import Count, Sum
from django.db.models.functions import TruncDate
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import path
from django.utils import timezone
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_GET, require_POST, require_safe

from src.config import Config, SETTINGS, profile_path
from src.models import Observation, WarningEvent
from src.sessions import expire_monitors


def is_local(request):
    return urlsplit("//" + request.get_host()).hostname in {"localhost", "127.0.0.1", "::1"}


def local_staff(view):
    @wraps(view)
    def guarded(request, *args, **kwargs):
        if not is_local(request):
            raise Http404
        if not request.user.is_active or not request.user.is_staff:
            return redirect("staff-login")
        return view(request, *args, **kwargs)

    return never_cache(guarded)


class StaffAuthenticationForm(AuthenticationForm):
    def confirm_login_allowed(self, user):
        if not user.is_staff:
            raise ValidationError(self.error_messages["invalid_login"], code="invalid_login")
        super().confirm_login_allowed(user)


class StaffLogin(LoginView):
    template_name = "staff_login.html"
    authentication_form = StaffAuthenticationForm
    next_page = "staff"
    redirect_authenticated_user = False
    extra_context = {"show_admin": True}

    def dispatch(self, request, *args, **kwargs):
        if not is_local(request):
            raise Http404
        return super().dispatch(request, *args, **kwargs)


class ConfigForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in SETTINGS.items():
            field_type = forms.IntegerField if name == "CAMERA_INDEX" else forms.FloatField
            self.fields[name] = field_type(
                label=field.label, min_value=field.minimum, max_value=field.maximum,
                help_text=field.hint,
            )

    def clean(self):
        values = super().clean()
        if not self.errors:
            try:
                Config.validate(values)
            except ValueError as error:
                raise ValidationError(str(error)) from error
        return values


LEGACY_REPORT = re.compile(r"web_(\d{8}_\d{6})_\d{6}\.csv")


def legacy_reports():
    reports = []
    for entry in sorted(settings.SMU_DATA_DIR.glob("web_*.csv"), reverse=True):
        match = LEGACY_REPORT.fullmatch(entry.name)
        if match:
            try:
                reports.append((entry.name, datetime.strptime(match[1], "%Y%m%d_%H%M%S")))
            except ValueError:
                continue
    return reports


def staff_context(request):
    today = timezone.localdate()
    first = today - timedelta(days=6)
    events = WarningEvent.objects.filter(created_at__date__gte=first)
    counts = dict(events.annotate(day=TruncDate("created_at")).values_list("day").annotate(total=Count("id")))
    maximum = max(counts.values(), default=0)
    days = []
    for i in range(7):
        day = first + timedelta(days=i)
        total = counts.get(day, 0)
        days.append({"day": day, "count": total, "height": round(total * 100 / maximum) if maximum else 0})
    observations = Observation.objects.filter(started_at__date__gte=first)
    summary = observations.aggregate(frames=Sum("frames"), face_frames=Sum("face_frames"))
    coverage = round(100 * summary["face_frames"] / summary["frames"]) if summary["frames"] else None
    reports = Observation.objects.annotate(warning_count=Count("warnings")).order_by("-started_at")
    return {
        "staff_page": True, "days": days, "session_count": observations.count(), "warning_count": sum(counts.values()),
        "coverage": coverage, "false_count": events.filter(false_alarm=True).count(),
        "reports": Paginator(reports, 12).get_page(request.GET.get("page")),
        "events": Paginator(WarningEvent.objects.order_by("-created_at"), 10).get_page(request.GET.get("events")),
        "archives": Paginator(legacy_reports(), 12).get_page(request.GET.get("archive")),
    }


@local_staff
@require_safe
def dashboard(request):
    expire_monitors()
    config_path = profile_path(settings.SMU_CONFIG_DIR)
    try:
        initial = Config(config_path).data
        config_error = ""
    except ValueError as error:
        initial = Config.DEFAULTS
        config_error = str(error)
    return render(request, "staff.html", {
        **staff_context(request), "form": ConfigForm(initial=initial), "config_error": config_error,
        "local_config": config_path.name, "saved": request.GET.get("saved") == "1",
    })


def save_profile(path, values):
    temp_path = None
    try:
        with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as file:
            temp_path = file.name
            yaml.safe_dump(values, file, sort_keys=False, allow_unicode=True)
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@local_staff
@require_POST
def update_config(request):
    path = settings.SMU_CONFIG_DIR / "config.local.yaml"
    form = ConfigForm(request.POST)
    reset = request.POST.get("reset") == "1"
    if reset or form.is_valid():
        try:
            if reset:
                Config(settings.SMU_CONFIG_DIR / "config.yaml")
                path.unlink(missing_ok=True)
            else:
                save_profile(path, form.cleaned_data)
            return redirect("/admin/?saved=1#settings")
        except (OSError, ValueError) as error:
            form.add_error(None, str(error))
    return render(request, "staff.html", {
        **staff_context(request), "form": form, "local_config": profile_path(settings.SMU_CONFIG_DIR).name,
        "open_settings": True,
    }, status=400)


@local_staff
@require_POST
def mark_warning(request, event_id):
    event = get_object_or_404(WarningEvent, pk=event_id)
    event.false_alarm = request.POST.get("false_alarm") == "1"
    event.save(update_fields=["false_alarm"])
    return redirect("/admin/#warnings")


def open_report(name):
    try:
        return FileResponse((settings.SMU_DATA_DIR / name).open("rb"), as_attachment=True, filename=name)
    except FileNotFoundError as error:
        raise Http404 from error


@local_staff
@require_GET
def download_report(request, observation_id):
    report = get_object_or_404(Observation, pk=observation_id, ended_at__isnull=False)
    return open_report(report.report_name)


@local_staff
@require_GET
def download_legacy(request, name):
    if not LEGACY_REPORT.fullmatch(name):
        raise Http404
    return open_report(name)


staff_urls = [
    path("admin/", dashboard, name="staff"),
    path("admin/login/", StaffLogin.as_view(), name="staff-login"),
    path("admin/logout/", local_staff(LogoutView.as_view(next_page="staff-login")), name="staff-logout"),
    path("admin/config/", update_config, name="staff-config"),
    path("admin/warnings/<int:event_id>/", mark_warning, name="staff-warning"),
    path("admin/report/<uuid:observation_id>/", download_report, name="staff-report"),
    path("admin/archive/<str:name>/", download_legacy, name="staff-archive"),
]
