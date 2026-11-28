"""Общие настройки для веб-интерфейса, оконной версии и проверки на видео."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Setting:
    label: str
    default: float
    minimum: float
    maximum: float
    hint: str


SETTINGS = {
    "EYE_AR_THRESH": Setting("Базовый порог EAR", 0.22, 0.08, 0.45, "Применяется, пока нет персональной калибровки."),
    "EAR_CLOSED_RATIO": Setting("Доля от обычной открытости глаз", 0.70, 0.45, 0.90, "При калибровке умножается на базовый EAR пользователя."),
    "EYE_CLOSED_SECONDS": Setting("Длительное закрытие глаз, с", 0.4, 0.2, 3, "Непрерывное закрытие обоих глаз."),
    "MAR_THRESH": Setting("Порог открытого рта", 0.3, 0.08, 0.9, "Используется для сигнала о возможном зевке."),
    "YAWN_SECONDS": Setting("Длительность возможного зевка, с", 1.2, 0.3, 6, "Отдельный сигнал, без голосовой тревоги."),
    "PERCLOS_WINDOW_SECONDS": Setting("Окно PERCLOS, с", 15.0, 2, 120, "Период для подсчёта доли закрытых глаз."),
    "PERCLOS_MIN_SECONDS": Setting("Разогрев PERCLOS, с", 8.0, 1, 120, "Минимальная длительность наблюдения для расчёта."),
    "PERCLOS_THRESH": Setting("Порог PERCLOS", 0.4, 0.05, 0.95, "Доля от 0 до 1."),
    "MAX_FRAME_GAP_SECONDS": Setting("Допустимый разрыв кадров, с", 1.0, 0.2, 3, "После более длинной паузы история сбрасывается."),
    "HEAD_PITCH_THRESH": Setting("Наклон головы, °", 25, 10, 65, "Отклонение от положения при калибровке."),
    "HEAD_YAW_THRESH": Setting("Поворот головы, °", 30, 10, 65, "Отклонение от положения при калибровке."),
    "HEAD_AWAY_SECONDS": Setting("Длительность отвлечения, с", 0.8, 0.2, 5, "Отклонение, которое удерживается это время."),
    "CALIBRATION_SECONDS": Setting("Калибровка, с", 3.0, 0, 10, "0 отключает калибровку. Для адаптации смотрите прямо в камеру."),
    "ALERT_COOLDOWN": Setting("Интервал голосовых предупреждений, с", 4, 1, 120, "Минимальное время между голосовыми предупреждениями."),
    "CAMERA_INDEX": Setting("Камера оконной версии", 0, 0, 15, "Порядковый номер устройства."),
}

LEGACY = {
    "EYE_AR_CONSEC_FRAMES": "EYE_CLOSED_SECONDS",
    "YAWN_CONSEC_FRAMES": "YAWN_SECONDS",
    "PERCLOS_WINDOW": "PERCLOS_WINDOW_SECONDS",
}


class Config:
    DEFAULTS = {name: field.default for name, field in SETTINGS.items()}

    def __init__(self, config_path: str | Path = "config.yaml", *, overrides=None):
        path = Path(config_path)
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError(f"Не удалось прочитать настройки из {path.name}.") from error
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError("YAML должен содержать пары «название: значение».")
        values = dict(loaded or {})
        for old, new in LEGACY.items():
            if old in values:
                value = values.pop(old)
                if type(value) not in (int, float):
                    raise ValueError(f"{old} должен быть числом.")
                values.setdefault(new, value / 30.0)
                if old == "PERCLOS_WINDOW":
                    values.setdefault("PERCLOS_MIN_SECONDS", values[new])
        self.data = self.validate({**values, **(overrides or {})})

    @classmethod
    def validate(cls, values):
        unknown = values.keys() - SETTINGS.keys()
        if unknown:
            raise ValueError("Неизвестные настройки: " + ", ".join(map(str, unknown)))
        config = {**cls.DEFAULTS, **values}
        for name, field in SETTINGS.items():
            value = config[name]
            if type(value) not in (int, float) or not field.minimum <= value <= field.maximum or not isfinite(value):
                raise ValueError(f"{field.label}: укажите число от {field.minimum:g} до {field.maximum:g}.")
        if type(config["CAMERA_INDEX"]) is not int:
            raise ValueError("Номер камеры должен быть целым числом.")
        if config["PERCLOS_MIN_SECONDS"] > config["PERCLOS_WINDOW_SECONDS"]:
            raise ValueError("Разогрев PERCLOS не может быть больше окна PERCLOS.")
        if 0 < config["CALIBRATION_SECONDS"] < 1:
            raise ValueError("Для калибровки укажите 0 или не меньше 1 секунды.")
        return config

    def __getitem__(self, key):
        return self.data[key]


def profile_path(directory: Path) -> Path:
    local = directory / "config.local.yaml"
    return local if local.is_file() else directory / "config.yaml"
