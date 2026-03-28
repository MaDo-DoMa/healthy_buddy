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

import cv2
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
# Alert cooldown guard
# ──────────────────────────────────────────────

class AlertCooldown:
    def __init__(self, hold_secs: float = 3.0, cooldown_secs: float = 60.0):
        self.hold_secs     = hold_secs
        self.cooldown_secs = cooldown_secs
        self._triggered_at:  dict[AlertType, float] = {}
        self._last_fired_at: dict[AlertType, float] = {}

    def should_fire(self, alert: AlertType, active: bool) -> bool:
        now = time.time()
        if not active:
            self._triggered_at.pop(alert, None)
            return False
        if alert not in self._triggered_at:
            self._triggered_at[alert] = now
        held_for    = now - self._triggered_at[alert]
        last_fire   = self._last_fired_at.get(alert, 0.0)
        on_cooldown = (now - last_fire) < self.cooldown_secs
        if held_for >= self.hold_secs and not on_cooldown:
            self._last_fired_at[alert] = now
            return True
        return False


# ──────────────────────────────────────────────
# Core monitor
# ──────────────────────────────────────────────

class CameraMonitor:
    SLOUCH_SHOULDER_ANGLE_DEG = 15.0
    SLOUCH_HEAD_DROP_RATIO    = 0.10
    GAZE_IRIS_RATIO_THRESH    = 0.30
    ABSENT_FRAMES_THRESH      = 30
    FPS_TARGET                = 10
    CALIBRATION_FRAMES        = 30

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.events: queue.Queue[CameraEvent] = queue.Queue(maxsize=20)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._ema_shoulder = EMA(alpha=0.2)
        self._ema_head_y   = EMA(alpha=0.2)
        self._ema_gaze     = EMA(alpha=0.3)

        self._cooldown = AlertCooldown(hold_secs=3.0, cooldown_secs=60.0)

        self._baseline_head_y:    Optional[float] = None
        self._calibration_frames: int   = 0
        self._calibration_sum:    float = 0.0
        self._absent_frames:      int   = 0

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="CameraMonitor")
        self._thread.start()

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

        frame_interval    = 1.0 / self.FPS_TARGET
        last_process_time = 0.0

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
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
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                self._process_results(results, w, h)

        cap.release()

    def _process_results(self, results, w: int, h: int):
        face_lm = results.face_landmarks
        pose_lm = results.pose_landmarks

        # Absence
        if face_lm is None:
            self._absent_frames += 1
            if self._absent_frames >= self.ABSENT_FRAMES_THRESH:
                if self._cooldown.should_fire(AlertType.ABSENT, True):
                    self._emit(AlertType.ABSENT, 1.0, "No face detected – are you still there?")
        else:
            self._absent_frames = 0
            self._cooldown.should_fire(AlertType.ABSENT, False)

        if face_lm is None or pose_lm is None:
            return

        # Calibration
        nose_y = face_lm.landmark[1].y
        if self._baseline_head_y is None:
            self._calibration_sum    += nose_y
            self._calibration_frames += 1
            if self._calibration_frames >= self.CALIBRATION_FRAMES:
                self._baseline_head_y = self._calibration_sum / self._calibration_frames
            return

        # Slouch: shoulder tilt
        l_sh = pose_lm.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = pose_lm.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        dx   = (r_sh.x - l_sh.x) * w
        dy   = (r_sh.y - l_sh.y) * h
        shoulder_angle = abs(math.degrees(math.atan2(dy, dx))) if dx != 0 else 0.0
        shoulder_angle = self._ema_shoulder.update(shoulder_angle)

        # Slouch: head drop
        current_head_y = self._ema_head_y.update(nose_y)
        head_drop      = current_head_y - self._baseline_head_y
        slouch = (shoulder_angle > self.SLOUCH_SHOULDER_ANGLE_DEG) or \
                 (head_drop      > self.SLOUCH_HEAD_DROP_RATIO)

        if self._cooldown.should_fire(AlertType.SLOUCH, slouch):
            conf = min(1.0, shoulder_angle / (self.SLOUCH_SHOULDER_ANGLE_DEG * 2))
            self._emit(AlertType.SLOUCH, conf, "Sit up straight – your posture is drooping.")
        elif not slouch and self._cooldown.should_fire(AlertType.POSTURE_OK, True):
            self._emit(AlertType.POSTURE_OK, 1.0, "Great posture – keep it up!")

        # Gaze
        try:
            iris_x   = face_lm.landmark[468].x
            eye_l_x  = face_lm.landmark[33].x
            eye_r_x  = face_lm.landmark[133].x
            eye_width = abs(eye_r_x - eye_l_x)
            if eye_width > 0:
                disp      = abs(iris_x - (eye_l_x + eye_r_x) / 2) / eye_width
                gaze_away = self._ema_gaze.update(disp) > self.GAZE_IRIS_RATIO_THRESH
            else:
                gaze_away = False
        except IndexError:
            gaze_away = False

        if self._cooldown.should_fire(AlertType.GAZE_AWAY, gaze_away):
            self._emit(AlertType.GAZE_AWAY, 0.8, "Eyes drifting – refocus on your screen.")

        # Phone heuristic
        try:
            l_wrist = pose_lm.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wrist = pose_lm.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            nose    = pose_lm.landmark[mp_pose.PoseLandmark.NOSE]
            face_y  = nose.y
            phone   = (abs(l_wrist.y - face_y) < 0.12) or \
                      (abs(r_wrist.y - face_y) < 0.12)
        except Exception:
            phone = False

        if self._cooldown.should_fire(AlertType.PHONE_VISIBLE, phone):
            self._emit(AlertType.PHONE_VISIBLE, 0.7, "Phone detected – put it down and stay focused!")

    def _emit(self, alert: AlertType, confidence: float, message: str):
        try:
            self.events.put_nowait(CameraEvent(alert=alert, confidence=confidence, message=message))
        except queue.Full:
            pass


# ──────────────────────────────────────────────
# FocusApp integration helpers
# ──────────────────────────────────────────────

def poll_camera_events(monitor: CameraMonitor, callback):
    while not monitor.events.empty():
        try:
            callback(monitor.events.get_nowait())
        except queue.Empty:
            break


# alias for app.py
CameraFeed = CameraMonitor