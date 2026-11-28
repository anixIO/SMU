# Отчёты: системный журнал, события усталости, звуковые оповещения, CSV и графики.
# Результаты работы сохраняются в output/; графики строятся при завершении программы.
# Для запуска приложения откройте main.py; этот файл содержит вспомогательные функции.

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    # Тип нужен редактору; при запуске загружать модуль расчётов не требуется.
    from src.metrics import FrameMetrics

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "output/fatigue_system.log", level: int = logging.INFO) -> None:
    # Настраивает корневой логгер: вывод в файл и в консоль.
    #
    # Параметры:
    #     log_file: путь к файлу лога.
    #     level: минимальный уровень сообщений.
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


class AlertSystem:
    # Оповещение пользователя о детектированной усталости.

    @staticmethod
    def play_sound() -> None:
        # Проигрывает предупреждение в фоновом потоке (не блокирует видео).
        #
        # macOS — голосом через ``say``, Windows — beep, иначе — системный звонок.
        def speak() -> None:
            try:
                if sys.platform == "darwin":
                    subprocess.run(["say", "Обнаружена усталость. Сделайте перерыв."])
                elif sys.platform == "win32":
                    import winsound
                    winsound.Beep(1000, 600)
                else:
                    print("\a")
            except Exception as e:
                logger.warning(f"не смог проиграть звук: {e}")
                print("\a")

        threading.Thread(target=speak, daemon=True).start()

    @staticmethod
    def log_event(reason: str, path: str = "output/fatigue_log.txt") -> None:
        # Дописывает событие с временной меткой в текстовый журнал.
        #
        # Параметры:
        #     reason: причины срабатывания.
        #     path: путь к журналу.
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {reason}\n")
        except Exception as e:
            logger.error(f"не смог записать событие: {e}")


class VideoLogger:
    # Пишет метрики каждого кадра в CSV-файл (перезаписывает при старте).

    HEADER: list[str] = ["Frame", "Timestamp", "EAR", "MAR", "PERCLOS",
                         "Pitch", "Yaw", "Roll", "Fatigue", "FaceVisible", "State"]

    def __init__(self, filename: str = "output/fatigue_metrics_log.csv") -> None:
        self.filename = filename
        self.file: IO[str] | None = None
        self.writer = None
        self.start_time = time.monotonic()
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            self.file = open(filename, "w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            self.writer.writerow(self.HEADER)
            self.file.flush()
            logger.info(f"csv лог создан: {filename}")
        except Exception as e:
            logger.error(f"не смог создать csv: {e}")

    def write(self, frame_id: int, m: FrameMetrics) -> None:
        # Записывает строку с метриками кадра.
        #
        # Параметры:
        #     frame_id: порядковый номер кадра.
        #     m: метрики кадра.
        if not self.writer:
            return
        try:
            self.writer.writerow([
                frame_id,
                round(time.monotonic() - self.start_time, 2),
                round(m.ear, 4), round(m.mar, 4),
                round(m.perclos, 4),
                *(round(angle, 2) if angle is not None else None for angle in (m.pitch, m.yaw, m.roll)),
                int(m.fatigue), int(m.detected), m.status,
            ])
            self.file.flush()
        except Exception as e:
            logger.error(f"ошибка записи в csv: {e}")

    def close(self) -> None:
        # Закрывает файл.
        if self.file:
            self.file.close()
            logger.info("csv лог закрыт")


def plot_metrics(csv_path: str = "output/fatigue_metrics_log.csv", out_dir: str = "output/fatigue_charts") -> None:
    # Сохраняет PNG-графики EAR, MAR, PERCLOS и событий усталости.
    #
    # Тихо пропускает работу, если нет matplotlib или CSV пуст.
    #
    # Параметры:
    #     csv_path: путь к CSV из VideoLogger.
    #     out_dir: папка для PNG.
    # Библиотека графиков нужна только при завершении, а не при запуске камеры.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib не установлен, графики пропускаем")
        return
    if not os.path.exists(csv_path):
        return
    try:
        with open(csv_path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            return
        timestamps = [float(row["Timestamp"]) for row in rows]
        os.makedirs(out_dir, exist_ok=True)
        plt.rcParams["axes.unicode_minus"] = False

        charts = [("EAR", "открытость глаз"), ("MAR", "открытость рта"),
                  ("PERCLOS", "доля закрытых глаз")]
        for col, title in charts:
            plt.figure(figsize=(12, 4))
            plt.plot(timestamps, [float(row[col]) for row in rows], linewidth=1.5)
            plt.xlabel("время, сек")
            plt.ylabel(title)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{col}.png"), dpi=100)
            plt.close()

        plt.figure(figsize=(12, 4))
        plt.fill_between(timestamps, [int(row["Fatigue"]) for row in rows], alpha=0.5, color="red")
        plt.xlabel("время, сек")
        plt.ylabel("усталость")
        plt.title("события усталости")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Fatigue.png"), dpi=100)
        plt.close()
        logger.info(f"графики сохранены в {out_dir}")
    except Exception as e:
        logger.error(f"ошибка построения графиков: {e}")


if __name__ == "__main__":
    # Если случайно запустили этот файл в IDE, подсказываем правильную точку входа.
    entry_point = Path(__file__).resolve().parent.parent / "main.py"
    print("Это модуль отчётов. Для запуска приложения выполните:")
    print(f'"{sys.executable}" "{entry_point}"')
