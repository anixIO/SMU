import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import time
import threading
from datetime import datetime
import yaml
from PIL import ImageFont, ImageDraw, Image
import csv
import os
import subprocess
import sys
import logging

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fatigue_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FontManager:

    def __init__(self):
        self.font_path = None
        self._find_font()

    def _find_font(self):
        candidates = [
            '/System/Library/Fonts/Supplemental/Arial.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
            'C:\\Windows\\Fonts\\arial.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
        ]
        for path in candidates:
            if os.path.exists(path):
                self.font_path = path
                logger.info(f"шрифт найден: {path}")
                return
        logger.warning("системный шрифт не найден, русский текст может отображаться криво")

    def put_text(self, img, text, position, font_size=24, color=(255, 255, 255)):
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)

            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()

            draw.text(position, text, font=font, fill=color[::-1])
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"ошибка при рисовании текста: {e}")
            return img


class Config:

    DEFAULTS = {
        'EYE_AR_THRESH': 0.25,
        'EYE_AR_CONSEC_FRAMES': 48,
        'MAR_THRESH': 0.7,
        'YAWN_CONSEC_FRAMES': 30,
        'PERCLOS_WINDOW': 100,
        'PERCLOS_THRESH': 0.4,
        'ALERT_COOLDOWN': 5,
        'HEAD_PITCH_THRESH': 15,
        'HEAD_YAW_THRESH': 20,
        'CAMERA_INDEX': 0
    }

    def __init__(self, config_path='config.yaml'):
        self.data = self._load(config_path)

    def _load(self, path):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    loaded = yaml.safe_load(f)
                if loaded:
                    logger.info(f"конфиг загружен: {path}")
                    return {**self.DEFAULTS, **loaded}
            except Exception as e:
                logger.warning(f"не смог прочитать {path}: {e}")
        logger.info("используем настройки по умолчанию")
        return dict(self.DEFAULTS)

    def __getitem__(self, key):
        return self.data[key]


class MetricsCalculator:

    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    MOUTH_IDX = [61, 291, 81, 178]
    FACE_RECT_IDX = [10, 152, 234, 454]
    HEAD_POSE_IDX = [33, 263, 1, 61, 291, 199]

    def __init__(self, perclos_window=100):
        self.perclos_history = deque(maxlen=perclos_window)

    @staticmethod
    def eye_aspect_ratio(eye):
        a = dist.euclidean(eye[1], eye[5])
        b = dist.euclidean(eye[2], eye[4])
        c = dist.euclidean(eye[0], eye[3])
        return (a + b) / (2.0 * c) if c > 0 else 0.0

    @staticmethod
    def mouth_aspect_ratio(mouth):
        width = dist.euclidean(mouth[0], mouth[1])
        height = dist.euclidean(mouth[2], mouth[3])
        return height / width if width > 0 else 0.0

    def compute_ear(self, pts):
        left = self.eye_aspect_ratio([pts[i] for i in self.LEFT_EYE_IDX])
        right = self.eye_aspect_ratio([pts[i] for i in self.RIGHT_EYE_IDX])
        return (left + right) / 2.0

    def compute_mar(self, pts):
        return self.mouth_aspect_ratio([pts[i] for i in self.MOUTH_IDX])

    def update_perclos(self, ear, threshold):
        self.perclos_history.append(1 if ear < threshold else 0)
        return sum(self.perclos_history) / len(self.perclos_history)

    @staticmethod
    def estimate_head_pose(pts, w, h):
        pts_2d = np.array(
            [(pts[i][0], pts[i][1]) for i in MetricsCalculator.HEAD_POSE_IDX],
            dtype=np.float64
        )
        pts_3d = np.array([
            (-0.3, -0.3, 0), (0.3, -0.3, 0),
            (0, 0, 0.5), (-0.2, 0.3, 0), (0.2, 0.3, 0),
            (0, 0.1, -0.2)
        ], dtype=np.float64) * 200

        focal = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        try:
            success, rvec, _ = cv2.solvePnP(
                pts_3d, pts_2d, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                return 0.0, 0.0, 0.0
            rmat, _ = cv2.Rodrigues(rvec)
            angles = cv2.decomposeProjectionMatrix(
                np.hstack((rmat, np.zeros((3, 1))))
            )[6]
            return float(angles[0]), float(angles[1]), float(angles[2])
        except Exception as e:
            logger.debug(f"solvepnp не сошёлся: {e}")
            return 0.0, 0.0, 0.0


class AlertSystem:

    @staticmethod
    def play_sound():
        def speak():
            try:
                if sys.platform == 'darwin':
                    subprocess.run(['say', 'Обнаружена усталость. Сделайте перерыв.'])
                elif sys.platform == 'win32':
                    import winsound
                    winsound.Beep(1000, 600)
                else:
                    print('\a')
            except Exception as e:
                logger.warning(f"не смог проиграть звук: {e}")
                print('\a')

        threading.Thread(target=speak, daemon=True).start()

    @staticmethod
    def log_event(reason):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open("fatigue_log.txt", "a", encoding='utf-8') as f:
                f.write(f"[{ts}] {reason}\n")
        except Exception as e:
            logger.error(f"не смог записать событие: {e}")


class VideoLogger:

    HEADER = ['Frame', 'Timestamp', 'EAR', 'MAR', 'PERCLOS',
              'Pitch', 'Yaw', 'Roll', 'Fatigue']

    def __init__(self, filename='fatigue_metrics_log.csv'):
        self.filename = filename
        self.file = None
        self.writer = None
        self.start_time = time.time()
        try:
            self.file = open(filename, 'w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
            self.writer.writerow(self.HEADER)
            self.file.flush()
            logger.info(f"csv лог создан: {filename}")
        except Exception as e:
            logger.error(f"не смог создать csv: {e}")

    def write(self, frame_id, m):
        if not self.writer:
            return
        try:
            self.writer.writerow([
                frame_id,
                round(time.time() - self.start_time, 2),
                round(m['ear'], 4), round(m['mar'], 4),
                round(m['perclos'], 4),
                round(m['pitch'], 2), round(m['yaw'], 2), round(m['roll'], 2),
                int(m['fatigue'])
            ])
            self.file.flush()
        except Exception as e:
            logger.error(f"ошибка записи в csv: {e}")

    def close(self):
        if self.file:
            self.file.close()
            logger.info("csv лог закрыт")


class HUDRenderer:

    def __init__(self, font_manager):
        self.font = font_manager

    def _draw_bar(self, img, x, y, w, h, value, max_val, warn, crit):
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

    def render(self, img, m, frame_id, fps):
        h, w = img.shape[:2]

        if not m['detected']:
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
        img = self._draw_bar(img, 15, y, 240, 12, m['ear'], 0.4, 0.2, 0.15)
        img = self.font.put_text(img, f"глаза (ear): {m['ear']:.2f}", (15, y - 5), 18, (200, 200, 200))

        img = self._draw_bar(img, 15, y + 35, 240, 12, m['mar'], 1.0, 0.5, 0.7)
        img = self.font.put_text(img, f"рот (mar): {m['mar']:.2f}", (15, y + 30), 18, (200, 200, 200))

        img = self._draw_bar(img, 15, y + 70, 240, 12, m['perclos'], 1.0, 0.3, 0.4)
        img = self.font.put_text(img, f"perclos: {m['perclos']:.2f}", (15, y + 65), 18, (200, 200, 200))

        img = self.font.put_text(img, f"наклон: {m['pitch']:.0f}", (15, y + 110), 16, (200, 200, 200))
        img = self.font.put_text(img, f"поворот: {m['yaw']:.0f}", (15, y + 140), 16, (200, 200, 200))
        img = self.font.put_text(img, f"крен: {m['roll']:.0f}", (15, y + 170), 16, (200, 200, 200))

        cv2.putText(img, f"frame: {frame_id}", (w - 130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"fps: {fps:.1f}", (w - 130, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if m['fatigue']:
            alpha = 0.5 + 0.3 * np.sin(time.time() * 10)
            cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            img = self.font.put_text(img, "Обнаружена усталость! сделайте перерыв",
                                     (w // 2 - 230, h - 48), 26, (255, 255, 255))

        cx, cy, r = w - 80, h - 80, 55
        angle = int(180 * min(m['perclos'], 1.0))
        cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 180, (50, 50, 50), -1)
        circle_color = (0, 255, 0) if m['perclos'] < 0.3 else (0, 255, 255) if m['perclos'] < 0.4 else (0, 0, 255)
        cv2.ellipse(img, (cx, cy), (r, r), 0, 0, angle, circle_color, -1)
        cv2.putText(img, f"{int(m['perclos'] * 100)}%", (cx - 23, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return img


class FatigueDetector:

    def __init__(self, config):
        self.cfg = config
        self.metrics = MetricsCalculator(config['PERCLOS_WINDOW'])
        self.font = FontManager()

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        self.connections = mp.solutions.face_mesh.FACEMESH_CONTOURS

        self.eye_counter = 0
        self.yawn_counter = 0
        self.last_alarm = 0
        logger.info("Детектор инициализирован")

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        m = {
            'detected': False, 'ear': 0.0, 'mar': 0.0, 'perclos': 0.0,
            'pitch': 0.0, 'yaw': 0.0,
            'roll': 0.0, 'fatigue': False, 'reasons': [], 'landmarks': None
        }

        if not result.multi_face_landmarks:
            return m

        landmarks = result.multi_face_landmarks[0]
        m['detected'] = True
        m['landmarks'] = landmarks
        pts = [(int(p.x * w), int(p.y * h)) for p in landmarks.landmark]

        m['ear'] = self.metrics.compute_ear(pts)
        m['mar'] = self.metrics.compute_mar(pts)
        m['perclos'] = self.metrics.update_perclos(m['ear'], self.cfg['EYE_AR_THRESH'])
        m['pitch'], m['yaw'], m['roll'] = MetricsCalculator.estimate_head_pose(pts, w, h)

        self.eye_counter = self.eye_counter + 1 if m['ear'] < self.cfg['EYE_AR_THRESH'] else 0
        self.yawn_counter = self.yawn_counter + 1 if m['mar'] > self.cfg['MAR_THRESH'] else 0

        reasons = []
        if self.eye_counter >= self.cfg['EYE_AR_CONSEC_FRAMES']:
            reasons.append(f"глаза закрыты {self.eye_counter} кадров")
        if self.yawn_counter >= self.cfg['YAWN_CONSEC_FRAMES']:
            reasons.append(f"зевота {self.yawn_counter} кадров")
        if m['perclos'] > self.cfg['PERCLOS_THRESH']:
            reasons.append(f"perclos {m['perclos']:.2f}")
        if m['pitch'] > self.cfg['HEAD_PITCH_THRESH']:
            reasons.append(f"наклон {m['pitch']:.0f}")
        if abs(m['yaw']) > self.cfg['HEAD_YAW_THRESH']:
            reasons.append(f"поворот {m['yaw']:.0f}")

        m['fatigue'] = len(reasons) > 0
        m['reasons'] = reasons
        return m

    def should_alert(self):
        now = time.time()
        if now - self.last_alarm > self.cfg['ALERT_COOLDOWN']:
            self.last_alarm = now
            return True
        return False

    def reset(self):
        self.eye_counter = 0
        self.yawn_counter = 0

    def close(self):
        self.face_mesh.close()


class App:

    def __init__(self, config_path='config.yaml'):
        self.cfg = Config(config_path)
        self.detector = FatigueDetector(self.cfg)
        self.video_log = VideoLogger()
        self.hud = HUDRenderer(self.detector.font)
        self.alert = AlertSystem()

        self.cap = cv2.VideoCapture(self.cfg['CAMERA_INDEX'])
        if not self.cap.isOpened():
            logger.error("не удалось открыть камеру")
            sys.exit(1)

        self.frame_id = 0
        self.fps = 30.0
        self._fps_count = 0
        self._fps_timer = time.time()

    @staticmethod
    def preprocess(frame):
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            return cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        except Exception:
            return frame

    def _update_fps(self):
        self._fps_count += 1
        if time.time() - self._fps_timer >= 1.0:
            self.fps = self._fps_count
            self._fps_count = 0
            self._fps_timer = time.time()

    def run(self):
        logger.info("система запущена, нажми esc для выхода")
        try:
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    logger.error("камера перестала отдавать кадры")
                    break

                frame = cv2.flip(frame, 1)
                self.frame_id += 1
                self._update_fps()

                frame = self.preprocess(frame)
                m = self.detector.process(frame)
                self.video_log.write(self.frame_id, m)

                if m['fatigue'] and self.detector.should_alert():
                    reason = "; ".join(m['reasons'])
                    logger.warning(f"усталость: {reason}")
                    self.alert.log_event(reason)
                    self.alert.play_sound()
                    self.detector.reset()

                display = self.hud.render(frame, m, self.frame_id, self.fps)

                if m['detected']:
                    self.detector.drawing.draw_landmarks(
                        display, m['landmarks'], self.detector.connections,
                        self.detector.drawing.DrawingSpec((0, 255, 0), 1, 1),
                        self.detector.drawing.DrawingSpec((0, 0, 255), 1, 1)
                    )

                cv2.imshow("детекция усталости", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        except KeyboardInterrupt:
            logger.info("остановлено пользователем")
        except Exception as e:
            logger.error(f"ошибка в основном цикле: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("завершаю работу")
        self.cap.release()
        self.video_log.close()
        self.detector.close()
        cv2.destroyAllWindows()
        self.plot()
        logger.info("готово")

    @staticmethod
    def plot(csv_path='fatigue_metrics_log.csv', out_dir='fatigue_charts'):
        if not PLOT_AVAILABLE:
            logger.info("matplotlib не установлен, графики пропускаем")
            return
        if not os.path.exists(csv_path):
            return
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                return
            os.makedirs(out_dir, exist_ok=True)
            plt.rcParams['axes.unicode_minus'] = False

            charts = [('EAR', 'открытость глаз'), ('MAR', 'открытость рта'),
                      ('PERCLOS', 'доля закрытых глаз')]
            for col, title in charts:
                plt.figure(figsize=(12, 4))
                plt.plot(df['Timestamp'], df[col], linewidth=1.5)
                plt.xlabel('время, сек')
                plt.ylabel(title)
                plt.title(title)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'{col}.png'), dpi=100)
                plt.close()

            plt.figure(figsize=(12, 4))
            plt.fill_between(df['Timestamp'], df['Fatigue'], alpha=0.5, color='red')
            plt.xlabel('время, сек')
            plt.ylabel('усталость')
            plt.title('события усталости')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'Fatigue.png'), dpi=100)
            plt.close()
            logger.info(f"графики сохранены в {out_dir}")
        except Exception as e:
            logger.error(f"ошибка построения графиков: {e}")


if __name__ == '__main__':
    App('config.yaml').run()