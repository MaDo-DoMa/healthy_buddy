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
from ultralytics import YOLO  # Wymaga: pip install ultralytics

# Importy Mediapipe
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# --- Typy danych bez zmian ---
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

# --- Klasy pomocnicze (EMA i Cooldown bez zmian) ---
class EMA:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self._value: Optional[float] = None
    def update(self, raw: float) -> float:
        if self._value is None: self._value = raw
        else: self._value = self.alpha * raw + (1.0 - self.alpha) * self._value
        return self._value

class AlertCooldown:
    def __init__(self, hold_secs: float = 3.0, repeat_secs: float = 10.0):
        self.hold_secs = hold_secs
        self.repeat_secs = repeat_secs
        self._triggered_at: dict[AlertType, float] = {}
        self._last_fired_at: dict[AlertType, float] = {}

    def should_fire(self, alert: AlertType, active: bool) -> bool:
        now = time.time()
        if not active:
            self._triggered_at.pop(alert, None)
            return False
        if alert not in self._triggered_at:
            self._triggered_at[alert] = now
        held_for = now - self._triggered_at[alert]
        last_fire = self._last_fired_at.get(alert, 0.0)
        if held_for >= self.hold_secs:
            if (now - last_fire) >= self.repeat_secs:
                self._last_fired_at[alert] = now
                return True
        return False

# --- Główny Monitor ---
class CameraMonitor:
    SLOUCH_HEAD_DROP_RATIO = 0.07  # Czułość opadnięcia głowy
    HEAD_TURN_THRESHOLD    = 0.35  # Próg odwrócenia głowy
    ABSENT_FRAMES_THRESH   = 30
    FPS_TARGET             = 10
    CALIBRATION_FRAMES     = 40

    def __init__(self, camera_index: int = 0, display_label: Optional[ctk.CTkLabel] = None):
        self.camera_index = camera_index
        self.display_label = display_label
        self.events: queue.Queue[CameraEvent] = queue.Queue(maxsize=20)
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Inicjalizacja modeli
        # YOLOv8 Nano - lekki i szybki model do detekcji obiektów
        self.yolo_model = YOLO('yolov8n.pt') 
        
        # Filtry EMA
        self._ema_head_y = EMA(alpha=0.15)
        self._ema_turn   = EMA(alpha=0.2)

        # Cooldowny
        self._cooldown = AlertCooldown(hold_secs=2.5, repeat_secs=10.0)

        # Kalibracja
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
        if self._thread: self._thread.join(timeout=3.0)

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self._emit(AlertType.ABSENT, 1.0, "Kamera niedostępna")
            return

        frame_interval = 1.0 / self.FPS_TARGET
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as holistic:
            while not self._stop_event.is_set():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret: continue

                # --- 1. DETEKCJA YOLO (TELEFON) ---
                # Szukamy tylko klasy 67 (cell phone)
                # conf=0.4 (dość wysoka pewność, by uniknąć błędów)
                yolo_results = self.yolo_model.predict(frame, classes=[67], conf=0.4, verbose=False)
                phone_found = len(yolo_results[0].boxes) > 0

                # --- 2. DETEKCJA MEDIAPIPE (POSTAWA) ---
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_results = holistic.process(rgb)

                # --- 3. LOGIKA DECYZYJNA ---
                self._process_combined_logic(mp_results, phone_found)

                # --- 4. WYŚWIETLANIE OBRAZU ---
                if self.display_label:
                    # Rysujemy szkielet Mediapipe
                    if mp_results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, mp_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # Rysujemy ramki YOLO dla telefonu
                    if phone_found:
                        for box in yolo_results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "TELEFON!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(220, 140))
                    self.display_label.after(0, lambda i=ctk_img: self.display_label.configure(image=i, text=""))

                # Kontrola FPS
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        cap.release()

    def _process_combined_logic(self, mp_results, phone_detected: bool):
        face_lm = mp_results.face_landmarks

        # A. NIEOBECNOŚĆ (Mediapipe)
        if face_lm is None:
            self._absent_frames += 1
            if self._absent_frames >= self.ABSENT_FRAMES_THRESH:
                if self._cooldown.should_fire(AlertType.ABSENT, True):
                    self._emit(AlertType.ABSENT, 1.0, "NIE WIDZĘ CIĘ! WRACAJ DO PRACY.")
            return
        else:
            self._absent_frames = 0
            self._cooldown.should_fire(AlertType.ABSENT, False)

        # B. TELEFON (YOLO) - Najwyższy priorytet
        if phone_detected:
            if self._cooldown.should_fire(AlertType.PHONE_VISIBLE, True):
                self._emit(AlertType.PHONE_VISIBLE, 1.0, "TELEFON WIDOCZNY! ODŁÓŻ GO.")
            # Jeśli używa telefonu, ignorujemy błędy postawy (jeden alert na raz)
            self._cooldown.should_fire(AlertType.GAZE_AWAY, False)
            self._cooldown.should_fire(AlertType.SLOUCH, False)
            return
        else:
            self._cooldown.should_fire(AlertType.PHONE_VISIBLE, False)

        # C. KALIBRACJA (Mediapipe)
        nose = face_lm.landmark[1]
        if self._baseline_head_y is None:
            self._calibration_sum += nose.y
            self._calibration_frames += 1
            if self._calibration_frames >= self.CALIBRATION_FRAMES:
                self._baseline_head_y = self._calibration_sum / self._calibration_frames
                print("DEBUG: Kalibracja postawy zakończona.")
            return

        # D. WZROK / ODWRACANIE GŁOWY (Mediapipe)
        try:
            l_cheek, r_cheek = face_lm.landmark[234], face_lm.landmark[454]
            dist_l = abs(nose.x - l_cheek.x)
            dist_r = abs(nose.x - r_cheek.x)
            asymmetry = abs(dist_l - dist_r) / (dist_l + dist_r + 1e-6)
            is_turning = self._ema_turn.update(asymmetry) > self.HEAD_TURN_THRESHOLD
        except: is_turning = False

        if is_turning:
            if self._cooldown.should_fire(AlertType.GAZE_AWAY, True):
                self._emit(AlertType.GAZE_AWAY, 0.8, "SKUP SIĘ NA MONITORZE!")
            self._cooldown.should_fire(AlertType.SLOUCH, False)
            return
        else:
            self._cooldown.should_fire(AlertType.GAZE_AWAY, False)

        # E. GARBIENIE (Mediapipe)
        current_head_y = self._ema_head_y.update(nose.y)
        head_drop = current_head_y - self._baseline_head_y
        
        if head_drop > self.SLOUCH_HEAD_DROP_RATIO:
            if self._cooldown.should_fire(AlertType.SLOUCH, True):
                self._emit(AlertType.SLOUCH, 1.0, "PROSTUJEMY SIĘ! GŁOWA ZA NISKO.")
        else:
            self._cooldown.should_fire(AlertType.SLOUCH, False)

    def _emit(self, alert: AlertType, confidence: float, message: str):
        try:
            self.events.put_nowait(CameraEvent(alert=alert, confidence=confidence, message=message))
        except queue.Full: pass

# Alias dla kompatybilności z app.py
CameraFeed = CameraMonitor