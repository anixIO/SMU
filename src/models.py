from uuid import uuid4

from django.db import models
from django.utils import timezone


class Observation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    started_at = models.DateTimeField(default=timezone.now, db_index=True)
    ended_at = models.DateTimeField(null=True)
    last_frame_at = models.DateTimeField(default=timezone.now)
    frames = models.PositiveIntegerField(default=0)
    face_frames = models.PositiveIntegerField(default=0)

    @property
    def report_name(self):
        return f"smu-{self.id.hex}.csv"

    @property
    def seconds(self):
        end = self.ended_at or self.last_frame_at
        return max(0, int((end - self.started_at).total_seconds()))

    @property
    def coverage(self):
        return round(100 * self.face_frames / self.frames) if self.frames else None


class WarningEvent(models.Model):
    observation = models.ForeignKey(Observation, related_name="warnings", on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    reason = models.CharField(max_length=300)
    false_alarm = models.BooleanField(default=False)
