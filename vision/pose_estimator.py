"""
pose_estimator.py — Head Pose + Eye Gaze Estimation via MediaPipe FaceLandmarker

Computes:
  • Head pitch angle (degrees) using solvePnP on 6 canonical face landmarks
  • Eye gaze direction using iris position relative to eye bounds
  • Chin-to-chest distance ratio as secondary confirmation
  • Combined "looking down" verdict (head pitch OR eyes pointing down)

Uses MediaPipe FaceLandmarker Tasks API (478 landmarks, CPU-optimised).
Model file: vision/face_landmarker.task
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# ── MediaPipe Tasks API aliases ─────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# ────────────────────────────────────────────────────────────────────
# 3-D reference model points (canonical face — generic human)
# ────────────────────────────────────────────────────────────────────
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),         # Nose tip
        (0.0, -330.0, -65.0),    # Chin
        (-225.0, 170.0, -135.0), # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),# Left mouth corner
        (150.0, -150.0, -125.0), # Right mouth corner
    ],
    dtype=np.float64,
)

_LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

# Iris / eye landmarks (MediaPipe FaceLandmarker 478 landmarks)
# Left eye vertical bounds: top 159, bottom 145, iris center 468
# Right eye vertical bounds: top 386, bottom 374, iris center 473
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145
_LEFT_IRIS_CENTER = 468
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374
_RIGHT_IRIS_CENTER = 473

# Default model path (relative to project root)
_DEFAULT_MODEL = str(Path(__file__).parent / "face_landmarker.task")


@dataclass
class PoseResult:
    """Result of a single-frame head-pose estimation."""

    head_down: bool = False
    eyes_down: bool = False
    looking_down: bool = False  # combined: head_down OR eyes_down
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    eye_gaze_ratio: float = 0.0  # 0 = looking up, 0.5 = center, 1 = down
    chin_chest_ratio: float = 0.0
    gaze_vector: tuple[float, float, float] = (0.0, 0.0, -1.0)
    landmarks_px: list[tuple[int, int]] = field(default_factory=list)


class PoseEstimator:
    """Wraps MediaPipe FaceLandmarker with head pose + eye gaze detection."""

    def __init__(self, config: dict) -> None:
        pose_cfg = config.get("pose", {})
        self._pitch_threshold = pose_cfg.get("pitch_threshold_deg", 20.0)
        self._eye_gaze_thr = pose_cfg.get("eye_gaze_threshold", 0.07)
        self._chin_ratio_thr = pose_cfg.get("chin_chest_ratio_threshold", 0.18)
        det_conf = pose_cfg.get("min_detection_confidence", 0.6)
        trk_conf = pose_cfg.get("min_tracking_confidence", 0.5)
        model_path = pose_cfg.get("model_path", _DEFAULT_MODEL)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at: {model_path}\n"
                "Download it with:\n"
                "  curl -o vision/face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=det_conf,
            min_face_presence_confidence=det_conf,
            min_tracking_confidence=trk_conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_ts_ms = 0

        # Camera intrinsics placeholder (set on first frame)
        self._cam_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        logger.info(
            "PoseEstimator ready  |  pitch_thr=%.1f°  eye_gaze_thr=%.2f  model=%s",
            self._pitch_threshold,
            self._eye_gaze_thr,
            model_path,
        )

    # ── public ──────────────────────────────────────────────────────
    def estimate(self, frame: np.ndarray) -> PoseResult:
        """Run head-pose + eye gaze estimation on a BGR frame."""
        h, w = frame.shape[:2]
        self._ensure_cam_matrix(w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not result.face_landmarks:
            return PoseResult()

        face_lms = result.face_landmarks[0]

        # ── Head pose via solvePnP ──────────────────────────────────
        image_pts = np.array(
            [(face_lms[i].x * w, face_lms[i].y * h) for i in _LANDMARK_IDS],
            dtype=np.float64,
        )

        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS, image_pts, self._cam_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return PoseResult()

        rmat, _ = cv2.Rodrigues(rvec)
        raw_pitch, yaw, roll = self._rotation_matrix_to_euler(rmat)

        # ── Normalise pitch ─────────────────────────────────────────
        # solvePnP with our 3-D model gives pitch centred at ±180°
        # when looking straight ahead.  Re-centre so 0° = straight,
        # positive = looking DOWN, negative = looking UP.
        pitch = (raw_pitch % 360) - 180

        # ── Eye gaze detection ──────────────────────────────────────
        eye_gaze_ratio = self._compute_eye_gaze(face_lms)
        eyes_down = eye_gaze_ratio > (0.5 + self._eye_gaze_thr)

        # ── Other metrics ───────────────────────────────────────────
        nose_y = face_lms[1].y
        chin_y = face_lms[199].y
        chin_chest_ratio = abs(chin_y - nose_y)

        gaze = (float(rmat[0, 2]), float(rmat[1, 2]), float(rmat[2, 2]))
        head_down = pitch > self._pitch_threshold

        # Combined verdict: head tilted down OR eyes looking down
        looking_down = head_down or eyes_down

        landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms]

        return PoseResult(
            head_down=head_down,
            eyes_down=eyes_down,
            looking_down=looking_down,
            pitch_deg=round(pitch, 1),
            yaw_deg=round(yaw, 1),
            roll_deg=round(roll, 1),
            eye_gaze_ratio=round(eye_gaze_ratio, 3),
            chin_chest_ratio=round(chin_chest_ratio, 4),
            gaze_vector=gaze,
            landmarks_px=landmarks_px,
        )

    def release(self) -> None:
        self._landmarker.close()
        logger.info("PoseEstimator released.")

    # ── Eye gaze ────────────────────────────────────────────────────
    def _compute_eye_gaze(self, face_lms) -> float:
        """
        Compute vertical eye gaze ratio using iris position relative to
        upper/lower eyelid. Returns 0.0 (up) → 0.5 (center) → 1.0 (down).

        Uses MediaPipe's refined iris landmarks (468-477).
        """
        try:
            # Left eye
            l_top_y = face_lms[_LEFT_EYE_TOP].y
            l_bot_y = face_lms[_LEFT_EYE_BOTTOM].y
            l_iris_y = face_lms[_LEFT_IRIS_CENTER].y
            l_range = l_bot_y - l_top_y
            l_ratio = (l_iris_y - l_top_y) / l_range if l_range > 1e-6 else 0.5

            # Right eye
            r_top_y = face_lms[_RIGHT_EYE_TOP].y
            r_bot_y = face_lms[_RIGHT_EYE_BOTTOM].y
            r_iris_y = face_lms[_RIGHT_IRIS_CENTER].y
            r_range = r_bot_y - r_top_y
            r_ratio = (r_iris_y - r_top_y) / r_range if r_range > 1e-6 else 0.5

            return (l_ratio + r_ratio) / 2.0
        except (IndexError, AttributeError):
            # Iris landmarks not available (model doesn't have them)
            return 0.5

    # ── private ─────────────────────────────────────────────────────
    def _ensure_cam_matrix(self, w: int, h: int) -> None:
        if self._cam_matrix is not None:
            return
        focal_length = w
        cx, cy = w / 2.0, h / 2.0
        self._cam_matrix = np.array(
            [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_matrix_to_euler(rmat: np.ndarray) -> tuple[float, float, float]:
        """Convert 3×3 rotation matrix → (pitch, yaw, roll) in degrees."""
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = 0.0
        return (
            math.degrees(pitch),
            math.degrees(yaw),
            math.degrees(roll),
        )
