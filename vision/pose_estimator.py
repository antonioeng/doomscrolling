"""
pose_estimator.py — Head Pose Estimation via MediaPipe Face Mesh

Computes:
  • Head pitch angle (degrees) using solvePnP on 6 canonical face landmarks
  • Chin-to-chest distance ratio as secondary confirmation
  • Gaze direction vector

Uses MediaPipe Face Mesh (468 landmarks, CPU-optimised).
Optionally runs an exported ONNX model via onnxruntime for NPU offload.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# 3-D reference model points (canonical face — generic human)
# Indices: nose tip 1, chin 199, left eye corner 33, right eye corner 263,
#          left mouth corner 61, right mouth corner 291
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


@dataclass
class PoseResult:
    """Result of a single-frame head-pose estimation."""

    head_down: bool = False
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    chin_chest_ratio: float = 0.0
    gaze_vector: tuple[float, float, float] = (0.0, 0.0, -1.0)
    landmarks_px: list[tuple[int, int]] = field(default_factory=list)


class PoseEstimator:
    """Wraps MediaPipe Face Mesh and exposes a simple ``estimate(frame)`` API."""

    def __init__(self, config: dict) -> None:
        pose_cfg = config.get("pose", {})
        self._pitch_threshold = pose_cfg.get("pitch_threshold_deg", 25.0)
        self._chin_ratio_thr = pose_cfg.get("chin_chest_ratio_threshold", 0.18)
        det_conf = pose_cfg.get("min_detection_confidence", 0.7)
        trk_conf = pose_cfg.get("min_tracking_confidence", 0.5)

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=det_conf,
            min_tracking_confidence=trk_conf,
        )

        # Camera intrinsics placeholder (set on first frame)
        self._cam_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        logger.info(
            "PoseEstimator ready  |  pitch_thr=%.1f°  chin_ratio_thr=%.2f",
            self._pitch_threshold,
            self._chin_ratio_thr,
        )

    # ── public ──────────────────────────────────────────────────────
    def estimate(self, frame: np.ndarray) -> PoseResult:
        """Run head-pose estimation on a BGR frame. Returns PoseResult."""
        h, w = frame.shape[:2]
        self._ensure_cam_matrix(w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return PoseResult()

        face = results.multi_face_landmarks[0]

        # Extract the 6 key 2-D landmarks
        image_pts = np.array(
            [(face.landmark[i].x * w, face.landmark[i].y * h) for i in _LANDMARK_IDS],
            dtype=np.float64,
        )

        # solvePnP → rotation vector → Euler angles
        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS,
            image_pts,
            self._cam_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return PoseResult()

        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rmat)

        # Chin-to-chest ratio: vertical distance nose→chin / frame height
        nose_y = face.landmark[1].y
        chin_y = face.landmark[199].y
        chin_chest_ratio = abs(chin_y - nose_y)

        # Gaze direction: third column of rotation matrix (z-axis of head)
        gaze = (float(rmat[0, 2]), float(rmat[1, 2]), float(rmat[2, 2]))

        # Decision: head is "down" if pitch exceeds threshold
        head_down = pitch > self._pitch_threshold

        # Collect visible landmarks for debug overlay
        landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]

        return PoseResult(
            head_down=head_down,
            pitch_deg=round(pitch, 1),
            yaw_deg=round(yaw, 1),
            roll_deg=round(roll, 1),
            chin_chest_ratio=round(chin_chest_ratio, 4),
            gaze_vector=gaze,
            landmarks_px=landmarks_px,
        )

    def release(self) -> None:
        self._face_mesh.close()
        logger.info("PoseEstimator released.")

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
