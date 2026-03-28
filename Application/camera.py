"""
camera.py  –  Healthy Buddy vision module
==========================================
Runs in a background daemon thread.
Publishes CameraEvent objects via a queue consumed by FocusApp.

Dependencies:
    pip install opencv-python mediapipe
"""

import threading
import queue
import time
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import mediapipe as mp

# ──────────────────────────────────────────────
# Public data types consumed by FocusApp
# ──────────────────────────────────────────────

class AlertType(Enum):
    SLOUCH        = auto()   # user is hunching – spine angle too large
    GAZE_AWAY     = auto()   # user stopped looking at screen
    PHONE_VISIBLE = auto()   # phone detected in frame
    ABSENT        = auto()   # no face detected at all
    POSTURE_OK    = auto()   # recovery / positive feedback


@dataclass
class CameraEvent:
    alert:       AlertType
    confidence:  float = 1.0          # 0.0–1.0
    message:     str   = ""           # human-readable hint for the UI
    timestamp:   float = field(default_factory=time.time)


# ──────────────────────────────────────────────
# EMA (Exponential Moving Average) smoother
# ──────────────────────────────────────────────

class EMA:
    """Single-value EMA filter to reduce jitter in landmark coordinates."""

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha          # 0=heavy smoothing, 1=no smoothing
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
# Alert fatigue guard
# ──────────────────────────────────────────────

class AlertCooldown:
    """
    Prevents spamming the same alert type.
    An alert fires only after it has been continuously triggered
    for `hold_secs`, and then won't fire again for `cooldown_secs`.
    """

    def __init__(self, hold_secs: float = 3.0, cooldown_secs: float = 60.0):
        self.hold_secs     = hold_secs
        self.cooldown_secs = cooldown_secs
        self._triggered_at:  dict[AlertType, float] = {}   # when condition first seen
        self._last_fired_at: dict[AlertType, float] = {}   # when alert last sent

    def should_fire(self, alert: AlertType, active: bool) -> bool:
        now = time.time()

        if not active:
            # condition cleared → reset trigger timer
            self._triggered_at.pop(alert, None)
            return False

        # start (or keep) the hold timer
        if alert not in self._triggered_at:
            self._triggered_at[alert] = now

        held_for  = now - self._triggered_at[alert]
        last_fire = self._last_fired_at.get(alert, 0.0)
        on_cooldown = (now - last_fire) < self.cooldown_secs

        if held_for >= self.hold_secs and not on_cooldown:
            self._last_fired_at[alert] = now
            return True

        return False


# ──────────────────────────────────────────────
# Core detector
# ──────────────────────────────────────────────

class CameraMonitor:
    """
    Runs MediaPipe Holistic + optional phone heuristic in a background thread.
    Puts CameraEvent objects into `self.events` (a queue.Queue).

    Usage:
        monitor = CameraMonitor()
        monitor.start()
        ...
        event = monitor.events.get_nowait()  # in FocusApp's polling loop
        ...
        monitor.stop()
    """

    # ── Thresholds (tweak to taste) ──────────────
    SLOUCH_SHOULDER_ANGLE_DEG = 15.0   # shoulder tilt vs horizontal
    SLOUCH_HEAD_DROP_RATIO    = 0.10   # head y / frame_height vs baseline
    GAZE_IRIS_RATIO_THRESH    = 0.30   # iris center displacement / eye width
    ABSENT_FRAMES_THRESH      = 30     # frames without face → ABSENT event
    FPS_TARGET                = 10     # process at most N frames/sec (CPU friendly)

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.events: queue.Queue[CameraEvent] = queue.Queue(maxsize=20)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # EMA smoothers per metric
        self._ema_shoulder = EMA(alpha=0.2)
        self._ema_head_y   = EMA(alpha=0.2)
        self._ema_gaze     = EMA(alpha=0.3)

        # Cooldown guards
        self._cooldown = AlertCooldown(hold_secs=3.0, cooldown_secs=60.0)

        # Calibration baseline (set after 30 "good" frames at session start)
        self._baseline_head_y:   Optional[float] = None
        self._calibration_frames: int = 0
        self._calibration_sum:    float = 0.0
        self.CALIBRATION_FRAMES = 30

        # Absence counter
        self._absent_frames = 0

        # MediaPipe handles (created inside thread)
        self._holistic = None
        self._mp_drawing = mp.solutions.drawing_utils

    # ── Public API ───────────────────────────────

    def start(self):
        """Start background capture thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="CameraMonitor")
        self._thread.start()

    def stop(self):
        """Signal thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal loop ────────────────────────────

    def _run(self):
        mp_holistic = mp.solutions.holistic

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
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as holistic:
            self._holistic = holistic

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

                self._process_results(results, w, h)

        cap.release()

    def _process_results(self, results, w: int, h: int):
        face_lm   = results.face_landmarks
        pose_lm   = results.pose_landmarks

        # ── Absence check ─────────────────────────
        if face_lm is None:
            self._absent_frames += 1
            if self._absent_frames >= self.ABSENT_FRAMES_THRESH:
                if self._cooldown.should_fire(AlertType.ABSENT, True):
                    self._emit(AlertType.ABSENT, 1.0, "No face detected – are you still there?")
        else:
            self._absent_frames = 0
            self._cooldown.should_fire(AlertType.ABSENT, False)  # reset cooldown

        if face_lm is None or pose_lm is None:
            return

        # ── Calibration phase ─────────────────────
        nose_y = face_lm.landmark[1].y  # MediaPipe tip-of-nose index
        if self._baseline_head_y is None:
            self._calibration_sum += nose_y
            self._calibration_frames += 1
            if self._calibration_frames >= self.CALIBRATION_FRAMES:
                self._baseline_head_y = self._calibration_sum / self._calibration_frames
            return  # don't analyse until calibrated

        # ── Posture: shoulder tilt ─────────────────
        l_shoulder = pose_lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = pose_lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

        dx = (r_shoulder.x - l_shoulder.x) * w
        dy = (r_shoulder.y - l_shoulder.y) * h
        shoulder_angle = abs(math.degrees(math.atan2(dy, dx))) if dx != 0 else 0.0
        shoulder_angle = self._ema_shoulder.update(shoulder_angle)

        # ── Posture: head drop ─────────────────────
        current_head_y = self._ema_head_y.update(nose_y)
        head_drop = current_head_y - self._baseline_head_y   # positive = lower in frame

        slouch = (shoulder_angle > self.SLOUCH_SHOULDER_ANGLE_DEG) or \
                 (head_drop > self.SLOUCH_HEAD_DROP_RATIO)

        if self._cooldown.should_fire(AlertType.SLOUCH, slouch):
            conf = min(1.0, shoulder_angle / (self.SLOUCH_SHOULDER_ANGLE_DEG * 2))
            self._emit(AlertType.SLOUCH, conf,
                       "Sit up straight – your posture is drooping.")
        elif not slouch and self._cooldown.should_fire(AlertType.POSTURE_OK, True):
            self._emit(AlertType.POSTURE_OK, 1.0, "Great posture – keep it up!")

        # ── Gaze: iris displacement ────────────────
        # Use face mesh landmarks 468/469 (left iris center) vs 33/133 (left eye corners)
        try:
            iris_x = face_lm.landmark[468].x
            eye_l_x = face_lm.landmark[33].x
            eye_r_x = face_lm.landmark[133].x
            eye_width = abs(eye_r_x - eye_l_x)
            if eye_width > 0:
                iris_displacement = abs(iris_x - (eye_l_x + eye_r_x) / 2) / eye_width
                gaze_away = self._ema_gaze.update(iris_displacement) > self.GAZE_IRIS_RATIO_THRESH
            else:
                gaze_away = False
        except IndexError:
            gaze_away = False

        if self._cooldown.should_fire(AlertType.GAZE_AWAY, gaze_away):
            self._emit(AlertType.GAZE_AWAY, 0.8,
                       "Eyes drifting – refocus on your screen.")

        # ── Phone heuristic (hand raised to ear/face height) ──────────────
        # A proper solution uses an object detector (YOLOv8-nano or MobileNet-SSD).
        # This lightweight heuristic catches the most common case: wrist near face.
        try:
            l_wrist = pose_lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            r_wrist = pose_lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            nose    = pose_lm.landmark[mp.solutions.pose.PoseLandmark.NOSE]

            face_y = nose.y
            phone_heuristic = (abs(l_wrist.y - face_y) < 0.12) or \
                              (abs(r_wrist.y - face_y) < 0.12)
        except Exception:
            phone_heuristic = False

        if self._cooldown.should_fire(AlertType.PHONE_VISIBLE, phone_heuristic):
            self._emit(AlertType.PHONE_VISIBLE, 0.7,
                       "Phone detected – put it down and stay focused!")

    def _emit(self, alert: AlertType, confidence: float, message: str):
        """Push an event to the queue; drop if full (non-blocking)."""
        event = CameraEvent(alert=alert, confidence=confidence, message=message)
        try:
            self.events.put_nowait(event)
        except queue.Full:
            pass


# ──────────────────────────────────────────────
# FocusApp integration helpers
# ──────────────────────────────────────────────

def poll_camera_events(monitor: CameraMonitor, callback):
    """
    Drain all pending events and call `callback(event)` for each.
    Call this from FocusApp's `after()` polling loop, e.g.:

        def _camera_poll(self):
            poll_camera_events(self._monitor, self._on_camera_event)
            self.after(500, self._camera_poll)

        def _on_camera_event(self, event: CameraEvent):
            if event.alert == AlertType.SLOUCH:
                show_notification(event.message)
            elif event.alert == AlertType.ABSENT and self._running:
                self._toggle_timer()   # auto-pause
    """
    while not monitor.events.empty():
        try:
            event = monitor.events.get_nowait()
            callback(event)
        except queue.Empty:
            break
