"""Жизненный цикл активных сеансов наблюдения."""

import atexit
import logging
import time
from uuid import UUID

from django.utils import timezone


logger = logging.getLogger(__name__)
ACTIVE_LIMIT = 3
IDLE_TIMEOUT = 90


class Monitor:
    def __init__(self, observation, detector, video_log):
        self.observation = observation
        self.detector = detector
        self.video_log = video_log
        self.started = self.last_frame = self.last_save = time.monotonic()

    def record(self, metrics):
        entry = self.observation
        self.last_frame = time.monotonic()
        entry.frames += 1
        entry.face_frames += int(metrics.detected)
        entry.last_frame_at = timezone.now()
        self.video_log.write(entry.frames, metrics)
        if self.last_frame - self.last_save >= 10:
            entry.save(update_fields=["frames", "face_frames", "last_frame_at"])
            self.last_save = self.last_frame

    def close(self):
        self.video_log.close()
        try:
            self.detector.close()
        finally:
            entry = self.observation
            entry.ended_at = entry.last_frame_at if entry.frames else timezone.now()
            entry.save(update_fields=["ended_at", "last_frame_at", "frames", "face_frames"])


monitors: dict[UUID, Monitor] = {}


def close_monitor(observation_id):
    monitor = monitors.pop(observation_id, None)
    if monitor:
        try:
            monitor.close()
        except Exception:
            logger.exception("Не удалось закрыть сеанс %s", observation_id)


def expire_monitors():
    now = time.monotonic()
    for observation_id, monitor in list(monitors.items()):
        if now - monitor.last_frame >= IDLE_TIMEOUT:
            close_monitor(observation_id)


@atexit.register
def close_monitors():
    for observation_id in list(monitors):
        close_monitor(observation_id)
