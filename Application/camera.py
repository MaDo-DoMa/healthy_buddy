"""
camera.py  –  Healthy Buddy vision module
==========================================
Compatible with mediapipe 0.10+
"""

import threading
import queue
import time
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from PIL import Image
import customtkinter as ctk
import cv2
import mediapipe as mp

# Używamy sprawdzonych ścieżek importu
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ──────────────────────────────────────────────
# Public data types
# ──────────────────────────────────────────────

class AlertType(Enum):
    SLOUCH        = auto()
    GAZE_AWAY     = auto()
    PHONE_VISIBLE = auto()
    ABSENT        = auto()
    POSTURE_OK    = auto()


@dataclass
class CameraEvent:
    alert:      AlertType
    confidence: float = 1.0
    message:    str   = ""
    timestamp:  float = field(default_factory=time.time)


# ──────────────────────────────────────────────
# EMA smoother
# ──────────────────────────────────────────────

class EMA:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, raw: float) -> float:
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None


# ──────────────────────────────────────────────
# Alert cooldown guard (Z DODANYM POWTARZANIEM)
# ──────────────────────────────────────────────

class AlertCooldown:
    def __init__(self, hold_secs: float = 3.0, repeat_secs: float = 10.0):
        self.hold_secs   = hold_secs
        self.repeat_secs = repeat_secs  # Co ile sekund powtarzać alert, jeśli błąd trwa
        self._triggered_at:  dict[AlertType, float] = {}
        self._last_fired_at: dict[AlertType, float] = {}

    def should_fire(self, alert: AlertType, active: bool) -> bool:
        now = time.time()
        
        if not active:
            # Jeśli stan błędu zniknął, usuwamy licznik trwania
            self._triggered_at.pop(alert, None)
            return False
            
        if alert not in self._triggered_at:
            self._triggered_at[alert] = now
            
        held_for = now - self._triggered_at[alert]
        last_fire = self._last_fired_at.get(alert, 0.0)
        
        # 1. Sprawdzamy czy błąd trwa wystarczająco długo (np. 3s)
        # 2. Sprawdzamy czy od ostatniego alertu minął czas powtórzenia (np. 10s)
        if held_for >= self.hold_secs:
            if (now - last_fire) >= self.repeat_secs:
                self._last_fired_at[alert] = now
                return True
                
        return False


# ──────────────────────────────────────────────
# Core monitor
# ──────────────────────────────────────────────

class CameraMonitor:
    SLOUCH_SHOULDER_ANGLE_DEG = 15.0
    SLOUCH_HEAD_DROP_RATIO    = 0.12  # Czułość opadnięcia głowy
    GAZE_IRIS_RATIO_THRESH    = 0.30
    ABSENT_FRAMES_THRESH      = 30
    FPS_TARGET                = 10
    CALIBRATION_FRAMES        = 40    # Dłuższa kalibracja na start dla stabilności

    def __init__(self, camera_index: int = 0, display_label: Optional[ctk.CTkLabel] = None):
        self.camera_index = camera_index
        self.display_label = display_label
        self.events: queue.Queue[CameraEvent] = queue.Queue(maxsize=20)
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Filtry wygładzające
        self._ema_shoulder = EMA(alpha=0.1)
        self._ema_head_y   = EMA(alpha=0.1)
        self._ema_gaze     = EMA(alpha=0.3)

        # Cooldowny (repeat_secs = 10.0 oznacza powtarzanie co 10s)
        self._cooldown = AlertCooldown(hold_secs=3.0, repeat_secs=10.0)

        # Logika kalibracji (Tylko raz!)
        self._baseline_head_y:    Optional[float] = None
        self._calibration_frames: int   = 0
        self._calibration_sum:    float = 0.0
        self._absent_frames:      int   = 0

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="CameraMonitor")
        self._thread.start()
        return True

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self._emit(AlertType.ABSENT, 1.0, "Camera not available")
            return

        frame_interval = 1.0 / self.FPS_TARGET
        last_process_time = 0.0

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as holistic:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                now = time.time()
                if (now - last_process_time) < frame_interval:
                    time.sleep(0.01)
                    continue
                last_process_time = now

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                
                # Przetwarzamy algorytmy AI
                self._process_results(results, w, h)

                # Wyświetlamy obraz w UI
                if self.display_label:
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(220, 140))
                    self.display_label.after(0, lambda i=ctk_img: self.display_label.configure(image=i, text=""))

        cap.release()

    def _process_results(self, results, w: int, h: int):
        face_lm = results.face_landmarks
        pose_lm = results.pose_landmarks

        # 1. LOGIKA NIEUBECNOŚCI (ABSENCE)
        if face_lm is None:
            self._absent_frames += 1
            if self._absent_frames >= self.ABSENT_FRAMES_THRESH:
                if self._cooldown.should_fire(AlertType.ABSENT, True):
                    self._emit(AlertType.ABSENT, 1.0, "NIE WYKRYTO TWARZY! CZY NADAL TU JESTEŚ?")
            return
        else:
            self._absent_frames = 0
            self._cooldown.should_fire(AlertType.ABSENT, False)

        if pose_lm is None:
            return

        # 2. KALIBRACJA (JEDNORAZOWA NA POCZĄTKU)
        nose_y = face_lm.landmark[1].y
        if self._baseline_head_y is None:
            self._calibration_sum    += nose_y
            self._calibration_frames += 1
            if self._calibration_frames >= self.CALIBRATION_FRAMES:
                self._baseline_head_y = self._calibration_sum / self._calibration_frames
                print(f"DEBUG: Kalibracja zakończona. Baza: {self._baseline_head_y}")
            return

        # --- 3. LOGIKA TELEFONU (NAJWYŻSZY PRIORYTET) ---
        try:
            l_wrist = pose_lm.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wrist = pose_lm.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            # Sprawdzamy czy którykolwiek nadgarstek jest blisko wysokości nosa
            is_phone = (abs(l_wrist.y - nose_y) < 0.15) or (abs(r_wrist.y - nose_y) < 0.15)
        except Exception:
            is_phone = False

        if is_phone:
            if self._cooldown.should_fire(AlertType.PHONE_VISIBLE, True):
                self._emit(AlertType.PHONE_VISIBLE, 1.0, "📱 WYKRYTO TELEFON! ODŁÓŻ GO I WRÓĆ DO PRACY!")
            
            # BARDZO WAŻNE: Jeśli wykryto telefon, przerywamy funkcję tutaj.
            # Dzięki temu nie sprawdzamy już garbienia się ani wzroku.
            return 
        else:
            # Jeśli telefonu nie ma, informujemy cooldown, że stan usterki minął
            self._cooldown.should_fire(AlertType.PHONE_VISIBLE, False)

        # --- 4. LOGIKA GARBIENIA (NIŻSZY PRIORYTET) ---
        # Obliczamy kąt ramion (EMA wygładza drgania)
        l_sh = pose_lm.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = pose_lm.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        dx = (r_sh.x - l_sh.x) * w
        dy = (r_sh.y - l_sh.y) * h
        shoulder_angle = abs(math.degrees(math.atan2(dy, dx))) if dx != 0 else 0.0
        shoulder_angle = self._ema_shoulder.update(shoulder_angle)

        # Obliczamy opadnięcie głowy względem kalibracji
        current_head_y = self._ema_head_y.update(nose_y)
        head_drop = current_head_y - self._baseline_head_y
        
        # Garbienie wykrywamy tylko gdy telefonu NIE MA (już zapewnione przez return wyżej)
        is_slouching = (shoulder_angle > self.SLOUCH_SHOULDER_ANGLE_DEG) or (head_drop > self.SLOUCH_HEAD_DROP_RATIO)

        if self._cooldown.should_fire(AlertType.SLOUCH, is_slouching):
            self._emit(AlertType.SLOUCH, 1.0, "USIĄDŹ PROSTO! TWOJA POSTAWA JEST NIEPRAWIDŁOWA.")
        
        # Jeśli się wyprostowałeś, wyślij alert o poprawie (tylko raz)
        elif not is_slouching and self._cooldown.should_fire(AlertType.POSTURE_OK, True):
            self._emit(AlertType.POSTURE_OK, 1.0, "ŚWIETNIE! TWOJA POSTAWA SIĘ POPRAWIŁA.")

        # --- 5. LOGIKA WZROKU (GAZE) ---
        try:
            iris_x   = face_lm.landmark[468].x
            eye_l_x  = face_lm.landmark[33].x
            eye_r_x  = face_lm.landmark[133].x
            eye_width = abs(eye_r_x - eye_l_x)
            if eye_width > 0:
                disp = abs(iris_x - (eye_l_x + eye_r_x) / 2) / eye_width
                gaze_away = self._ema_gaze.update(disp) > self.GAZE_IRIS_RATIO_THRESH
            else:
                gaze_away = False
        except IndexError:
            gaze_away = False

        if self._cooldown.should_fire(AlertType.GAZE_AWAY, gaze_away):
            self._emit(AlertType.GAZE_AWAY, 0.8, "SKUP SIĘ NA EKRANIE! TWOJE OCZY UCIEKAJĄ.")

    def _emit(self, alert: AlertType, confidence: float, message: str):
        print(f"DEBUG: Emitowano alert: {alert.name}")
        try:
            self.events.put_nowait(CameraEvent(alert=alert, confidence=confidence, message=message))
        except queue.Full:
            pass

# Pomocnik dla app.py
def poll_camera_events(monitor: CameraMonitor, callback):
    while not monitor.events.empty():
        try:
            callback(monitor.events.get_nowait())
        except queue.Empty:
            break

CameraFeed = CameraMonitor