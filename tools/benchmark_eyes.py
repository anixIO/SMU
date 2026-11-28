"""Сравнение порогов EAR на исходной разметке Eyeblink8."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import cv2

from src.config import Config
from src.detector import FatigueDetector


def read_labels(archive, stem):
    labels = {}
    for line in archive.read(stem + ".tag").decode("utf-8", errors="replace").splitlines():
        fields = line.split(":")
        if len(fields) < 7 or not fields[0].isdigit():
            continue
        frame_id, blink, angle, left, left_hidden, right, right_hidden = fields[:7]
        if angle != "X" or left_hidden != "X" or right_hidden != "X":
            continue
        if left == right == "C":
            labels[int(frame_id)] = True
        elif blink == "-1" and left == right == "X":
            labels[int(frame_id)] = False
    return labels


def timestamps(archive, stem):
    return dict((int(index), float(stamp)) for index, stamp in (
        line.split() for line in archive.read(stem + ".txt").decode().splitlines() if line.strip()
    ))


def score(label, predicted):
    return ("true_" if label == predicted else "false_") + ("closed" if predicted else "open")


def describe(counts):
    closed_hits = counts["true_closed"]
    closed_misses = counts["false_open"]
    open_hits = counts["true_open"]
    open_misses = counts["false_closed"]
    return {
        "frames": dict(counts),
        "closed_recall": round(closed_hits / (closed_hits + closed_misses), 4) if closed_hits + closed_misses else None,
        "open_specificity": round(open_hits / (open_hits + open_misses), 4) if open_hits + open_misses else None,
    }


def evaluate(archive, name, folder, step, config):
    stem = name.removesuffix(".avi")
    labels = read_labels(archive, stem)
    timing = timestamps(archive, stem)
    path = Path(folder) / "sample.avi"
    with archive.open(name) as source, path.open("wb") as target:
        copyfileobj(source, target)
    video = cv2.VideoCapture(str(path))
    detector = FatigueDetector(config)
    fixed, adapted = Counter(), Counter()
    eligible = evaluated = 0
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        if not video.isOpened() or fps <= 0:
            raise ValueError(f"Не удалось открыть {name}")
        index = -1
        while video.grab():
            index += 1
            if index % step != 0:
                continue
            ok, frame = video.retrieve()
            if not ok:
                raise ValueError(f"Не удалось прочитать кадр {index} в {name}")
            metrics = detector.process(frame, now=timing.get(index, index / fps))
            label = labels.get(index)
            if label is None or not detector.rules.calibration_done:
                continue
            eligible += 1
            if not metrics.detected:
                continue
            evaluated += 1
            fixed[score(label, metrics.ear < 0.22)] += 1
            adapted[score(label, detector.rules.closed)] += 1
    finally:
        detector.close()
        video.release()
    return fixed, adapted, eligible, evaluated, detector.rules.ear_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Исходный eyeblink8.zip")
    parser.add_argument("--step", type=int, default=3, help="Шаг по кадрам; по умолчанию каждый третий")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step должен быть положительным")
    config = Config(Path(__file__).resolve().parents[1] / "config.yaml")
    fixed, adapted = Counter(), Counter()
    total_eligible = total_evaluated = 0
    with ZipFile(args.archive) as archive, TemporaryDirectory(prefix="smu-eyes-") as folder:
        videos = sorted(name for name in archive.namelist() if name.lower().endswith(".avi"))
        if not videos:
            parser.error("В архиве нет AVI с разметкой Eyeblink8")
        for name in videos:
            old, new, eligible, evaluated, threshold = evaluate(archive, name, folder, args.step, config)
            fixed.update(old)
            adapted.update(new)
            total_eligible += eligible
            total_evaluated += evaluated
            print(json.dumps({"clip": name, "threshold": round(threshold, 4),
                              "eligible": eligible, "evaluated": evaluated,
                              "fixed": describe(old), "adapted": describe(new)}), flush=True)
    print(json.dumps({"clips": len(videos), "eligible": total_eligible, "evaluated": total_evaluated,
                      "fixed": describe(fixed), "adapted": describe(adapted)}, indent=2))


if __name__ == "__main__":
    main()
