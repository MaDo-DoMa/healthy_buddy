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
# Alert cooldown guard (Z POWTARZANIEM)
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


# ──────────────────────────────────────────────
# Core monitor
# ──────────────────────────────────────────────

class CameraMonitor:
    SLOUCH_SHOULDER_ANGLE_DEG = 15.0
    SLOUCH_HEAD_DROP_RATIO    = 0.08  # Czułość opadnięcia głowy
    HEAD_TURN_THRESHOLD       = 0.31  # PRÓG ODWRÓCENIA GŁOWY (0.3 - 0.5)
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
        self._ema_turn     = EMA(alpha=0.2)

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

    def _is_hand_gripping(self, hand_landmarks):
        """Sprawdza, czy dłoń jest w pozycji uścisku (zaciśnięta)"""
        if not hand_landmarks:
            return False
            
        # Punkty: 0 - nadgarstek, 8, 12, 16, 20 - końcówki palców, 5, 9, 13, 17 - stawy (MCP)
        tips = [8, 12, 16, 20]
        knuckles = [5, 9, 13, 17]
        wrist = hand_landmarks.landmark[0]
        
        gripping_fingers = 0
        for tip_idx, knuck_idx in zip(tips, knuckles):
            tip = hand_landmarks.landmark[tip_idx]
            knuckle = hand_landmarks.landmark[knuck_idx]
            
            # Liczymy dystans od nadgarstka do końcówki i do stawu
            dist_tip = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            dist_knuckle = math.sqrt((knuckle.x - wrist.x)**2 + (knuckle.y - wrist.y)**2)
            
            # Jeśli końcówka palca jest bliżej nadgarstka niż staw, palec jest zgięty
            if dist_tip < dist_knuckle:
                gripping_fingers += 1
                
        # Jeśli przynajmniej 3 palce są zgięte, uznajemy to za uścisk
        return gripping_fingers >= 3

    def _process_results(self, results, w: int, h: int):
        # Pobieramy wszystkie dane z MediaPipe
        face_lm = results.face_landmarks
        pose_lm = results.pose_landmarks
        left_h  = results.left_hand_landmarks
        right_h = results.right_hand_landmarks

        # --- 1. PRZYGOTOWANIE DANYCH (Zmienne pomocnicze) ---
        # Definiujemy 'nose' na samym początku, jeśli twarz jest widoczna
        nose = face_lm.landmark[1] if face_lm else None
        is_phone_detected = False

        # --- 2. SUPER DOKŁADNA LOGIKA TELEFONU (PRIORYTET #1) ---
        # Sprawdzamy dłonie nawet jeśli twarz jest zasłonięta przez telefon!
        for hand in [left_h, right_h]:
            if hand:
                wrist = hand.landmark[0]
                is_gripping = self._is_hand_gripping(hand) # Twoja nowa funkcja uścisku
                is_closer = wrist.z < -0.05 # Dłoń musi być wysunięta do przodu
                
                # Warunek zbliżenia do twarzy:
                if nose:
                    # Jeśli twarz jest widoczna, sprawdzamy dystans do nosa
                    near_face = (abs(wrist.y - nose.y) < 0.20) and (abs(wrist.x - nose.x) < 0.20)
                else:
                    # Jeśli twarz NIE jest widoczna (np. zasłonięta telefonem),
                    # sprawdzamy czy dłoń jest w centralnym/górnym obszarze ekranu
                    near_face = (0.2 < wrist.y < 0.6) and (0.3 < wrist.x < 0.7)

                if near_face and is_gripping and is_closer:
                    is_phone_detected = True
                    break

        if is_phone_detected:
            if self._cooldown.should_fire(AlertType.PHONE_VISIBLE, True):
                self._emit(AlertType.PHONE_VISIBLE, 1.0, "📱 WYKRYTO TELEFON! ODŁÓŻ GO I WRÓĆ DO PRACY!")
            
            # Resetujemy inne alerty i przerywamy (telefon jest najważniejszy)
            self._absent_frames = 0
            self._cooldown.should_fire(AlertType.GAZE_AWAY, False)
            self._cooldown.should_fire(AlertType.SLOUCH, False)
            self._cooldown.should_fire(AlertType.ABSENT, False)
            return

        # --- 3. LOGIKA NIEUBECNOŚCI ---
        # Wykonuje się tylko, gdy NIE wykryto telefonu
        if face_lm is None:
            self._absent_frames += 1
            if self._absent_frames >= self.ABSENT_FRAMES_THRESH:
                if self._cooldown.should_fire(AlertType.ABSENT, True):
                    self._emit(AlertType.ABSENT, 1.0, "WYKRYTO TWOJĄ NIEOBECNOŚĆ. WRACAJ DO PRACY!")
            return
        else:
            self._absent_frames = 0
            self._cooldown.should_fire(AlertType.ABSENT, False)

        if not pose_lm or not nose:
            return

        # --- 4. KALIBRACJA (JEDNORAZOWA) ---
        if self._baseline_head_y is None:
            self._calibration_sum    += nose.y
            self._calibration_frames += 1
            if self._calibration_frames >= self.CALIBRATION_FRAMES:
                self._baseline_head_y = self._calibration_sum / self._calibration_frames
                print("DEBUG: Kalibracja zakończona.")
            return

        # --- 5. LOGIKA ODWRACANIA GŁOWY ---
        is_turning = False
        try:
            l_cheek, r_cheek = face_lm.landmark[234], face_lm.landmark[454]
            dist_l = abs(nose.x - l_cheek.x)
            dist_r = abs(nose.x - r_cheek.x)
            asymmetry = abs(dist_l - dist_r) / (dist_l + dist_r)
            is_turning = self._ema_turn.update(asymmetry) > self.HEAD_TURN_THRESHOLD
        except: is_turning = False

        if is_turning:
            if self._cooldown.should_fire(AlertType.GAZE_AWAY, True):
                self._emit(AlertType.GAZE_AWAY, 0.8, "👀 NIE ODWRACAJ GŁOWY! SKUP SIĘ NA MONITORZE!")
            self._cooldown.should_fire(AlertType.SLOUCH, False)
            return

        # --- 6. LOGIKA GARBIENIA (TYLKO GŁOWA) ---
        head_drop = self._ema_head_y.update(nose.y) - self._baseline_head_y
        if self._cooldown.should_fire(AlertType.SLOUCH, head_drop > self.SLOUCH_HEAD_DROP_RATIO):
            self._emit(AlertType.SLOUCH, 1.0, "🧘 USIĄDŹ PROSTO! TWOJA GŁOWA ZBYT MOCNO OPADŁA.")

    def _emit(self, alert: AlertType, confidence: float, message: str):
        print(f"DEBUG: Emitowano alert: {alert.name} -> {message}")
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