"""
phone_detector.py — Real-time cell phone detection via YOLOv8 / ONNX

Detects objects of class "cell phone" (COCO class 67) and validates that the
bounding box is in the lower-centre region of the frame (torso / hands area)
to reduce false positives from monitors, tablets, or background clutter.

Supports two backends:
  • Ultralytics YOLOv8 (.pt)   — easiest, GPU via CUDA
  • ONNX Runtime (.onnx)       — NPU via DmlExecutionProvider / TensorRT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# COCO class index for "cell phone"
_CELL_PHONE_CLASS = 67


@dataclass
class PhoneDetection:
    """Single detected phone instance."""

    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    confidence: float = 0.0
    in_torso_zone: bool = False


@dataclass
class PhoneResult:
    """Aggregated phone detection result for one frame."""

    phone_detected: bool = False
    detections: list[PhoneDetection] = field(default_factory=list)


class PhoneDetector:
    """Detects cell phones using YOLOv8 (PyTorch) or ONNX Runtime."""

    def __init__(self, config: dict) -> None:
        phone_cfg = config.get("phone_detection", {})
        self._conf_thr = phone_cfg.get("confidence_threshold", 0.45)
        self._iou_thr = phone_cfg.get("iou_threshold", 0.5)
        self._use_onnx = phone_cfg.get("use_onnx", False)
        self._proximity_margin = phone_cfg.get("proximity_margin", 0.15)

        npu_cfg = config.get("npu", {})
        self._npu_enabled = npu_cfg.get("enabled", False)
        self._ep = npu_cfg.get("execution_provider", "DmlExecutionProvider")
        self._fallback_ep = npu_cfg.get("fallback_provider", "CPUExecutionProvider")

        if self._use_onnx:
            self._init_onnx(phone_cfg.get("onnx_model_path", "vision/yolov8n.onnx"))
        else:
            self._init_ultralytics(phone_cfg.get("model_path", "yolov8n.pt"))

    # ── Backend initialisation ──────────────────────────────────────
    def _init_ultralytics(self, model_path: str) -> None:
        from ultralytics import YOLO

        self._backend = "ultralytics"
        self._model = YOLO(model_path)
        logger.info("PhoneDetector ready  |  backend=ultralytics  model=%s", model_path)

    def _init_onnx(self, onnx_path: str) -> None:
        import onnxruntime as ort

        providers = []
        if self._npu_enabled:
            providers.append(self._ep)
        providers.append(self._fallback_ep)

        self._backend = "onnx"
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        actual = self._session.get_providers()
        logger.info(
            "PhoneDetector ready  |  backend=onnx  model=%s  providers=%s",
            onnx_path,
            actual,
        )

    # ── Public API ──────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> PhoneResult:
        """Run phone detection on a BGR frame. Returns PhoneResult."""
        if self._backend == "ultralytics":
            return self._detect_ultralytics(frame)
        return self._detect_onnx(frame)

    def release(self) -> None:
        if self._backend == "onnx":
            del self._session
        logger.info("PhoneDetector released.")

    # ── Ultralytics inference ───────────────────────────────────────
    def _detect_ultralytics(self, frame: np.ndarray) -> PhoneResult:
        h, w = frame.shape[:2]
        results = self._model.predict(
            frame,
            conf=self._conf_thr,
            iou=self._iou_thr,
            classes=[_CELL_PHONE_CLASS],
            verbose=False,
        )

        detections: list[PhoneDetection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                in_zone = self._is_in_torso_zone(x1, y1, x2, y2, w, h)
                detections.append(
                    PhoneDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, in_torso_zone=in_zone)
                )

        # Only count phones that are actually in the torso/hands zone
        valid = [d for d in detections if d.in_torso_zone]
        return PhoneResult(phone_detected=len(valid) > 0, detections=detections)

    # ── ONNX inference (YOLOv8 exported) ────────────────────────────
    def _detect_onnx(self, frame: np.ndarray) -> PhoneResult:
        import cv2

        h, w = frame.shape[:2]

        # Pre-process: resize to 640×640, normalise, NCHW
        input_img = cv2.resize(frame, (640, 640))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))  # HWC → CHW
        input_img = np.expand_dims(input_img, 0)         # add batch

        outputs = self._session.run(None, {self._input_name: input_img})
        preds = outputs[0]  # shape: (1, 84, 8400) for YOLOv8

        detections = self._parse_yolov8_output(preds, w, h)
        valid = [d for d in detections if d.in_torso_zone]
        return PhoneResult(phone_detected=len(valid) > 0, detections=detections)

    def _parse_yolov8_output(
        self, preds: np.ndarray, orig_w: int, orig_h: int
    ) -> list[PhoneDetection]:
        """Parse raw YOLOv8 ONNX output (1, 84, 8400) → list[PhoneDetection]."""
        preds = np.squeeze(preds, axis=0)  # (84, 8400)
        preds = preds.T                     # (8400, 84)

        detections: list[PhoneDetection] = []
        scale_x, scale_y = orig_w / 640.0, orig_h / 640.0

        for row in preds:
            cx, cy, bw, bh = row[:4]
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])

            if class_id != _CELL_PHONE_CLASS or conf < self._conf_thr:
                continue

            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)

            in_zone = self._is_in_torso_zone(x1, y1, x2, y2, orig_w, orig_h)
            detections.append(
                PhoneDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, in_torso_zone=in_zone)
            )

        return detections

    # ── Proximity validation ────────────────────────────────────────
    def _is_in_torso_zone(
        self, x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int
    ) -> bool:
        """
        Check if the phone bounding box is in the torso/hands region.

        The torso zone is defined as:
          • Horizontal: centre 70 % of frame (margin on each side)
          • Vertical:   lower 70 % of frame (below face level)

        This eliminates false positives from TVs, monitors, distant phones.
        """
        margin = self._proximity_margin
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        in_x = (margin * frame_w) < cx < ((1 - margin) * frame_w)
        in_y = cy > (0.3 * frame_h)  # below top 30 %

        # Reject very large boxes (likely a monitor, not a phone)
        box_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_w * frame_h
        area_ratio = box_area / frame_area if frame_area > 0 else 1.0
        not_too_large = area_ratio < 0.35

        return in_x and in_y and not_too_large
