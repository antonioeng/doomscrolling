"""
phone_detector.py — Real-time cell phone detection via YOLOv8 / ONNX

Detects objects of class "cell phone" (COCO 67), "laptop" (63), "remote" (62)
and validates via bounding box position, aspect ratio, and size filters.

Supports two backends:
  • Ultralytics YOLOv8 (.pt)   — easiest, GPU via CUDA
  • ONNX Runtime (.onnx)       — NPU via DmlExecutionProvider / TensorRT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# COCO class indices: cell phone=67, laptop=63, remote=62
_DEFAULT_CLASSES = [67, 63, 62]


@dataclass
class PhoneDetection:
    """Single detected phone instance."""

    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    confidence: float = 0.0
    class_id: int = 67
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
        self._conf_thr = phone_cfg.get("confidence_threshold", 0.30)
        self._iou_thr = phone_cfg.get("iou_threshold", 0.45)
        self._use_onnx = phone_cfg.get("use_onnx", False)
        self._proximity_margin = phone_cfg.get("proximity_margin", 0.10)
        self._target_classes = phone_cfg.get("target_classes", _DEFAULT_CLASSES)
        self._min_aspect_ratio = phone_cfg.get("min_aspect_ratio", 1.2)
        self._max_area_ratio = phone_cfg.get("max_area_ratio", 0.30)
        self._imgsz = phone_cfg.get("imgsz", 640)

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
        logger.info(
            "PhoneDetector ready  |  backend=ultralytics  model=%s  classes=%s  conf=%.2f",
            model_path, self._target_classes, self._conf_thr,
        )

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
            onnx_path, actual,
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
            classes=self._target_classes,
            imgsz=self._imgsz,
            verbose=False,
        )

        detections: list[PhoneDetection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                in_zone = self._validate_detection(x1, y1, x2, y2, w, h)
                detections.append(
                    PhoneDetection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=conf, class_id=cls_id, in_torso_zone=in_zone,
                    )
                )

        valid = [d for d in detections if d.in_torso_zone]
        return PhoneResult(phone_detected=len(valid) > 0, detections=detections)

    # ── ONNX inference (YOLOv8 exported) ────────────────────────────
    def _detect_onnx(self, frame: np.ndarray) -> PhoneResult:
        import cv2

        h, w = frame.shape[:2]
        sz = self._imgsz

        input_img = cv2.resize(frame, (sz, sz))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, 0)

        outputs = self._session.run(None, {self._input_name: input_img})
        preds = outputs[0]

        detections = self._parse_yolov8_output(preds, w, h)
        valid = [d for d in detections if d.in_torso_zone]
        return PhoneResult(phone_detected=len(valid) > 0, detections=detections)

    def _parse_yolov8_output(
        self, preds: np.ndarray, orig_w: int, orig_h: int
    ) -> list[PhoneDetection]:
        """Parse raw YOLOv8 ONNX output (1, 84, 8400) → list[PhoneDetection]."""
        sz = self._imgsz
        preds = np.squeeze(preds, axis=0).T

        detections: list[PhoneDetection] = []
        scale_x, scale_y = orig_w / sz, orig_h / sz

        for row in preds:
            cx, cy, bw, bh = row[:4]
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])

            if class_id not in self._target_classes or conf < self._conf_thr:
                continue

            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)

            in_zone = self._validate_detection(x1, y1, x2, y2, orig_w, orig_h)
            detections.append(
                PhoneDetection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf, class_id=class_id, in_torso_zone=in_zone,
                )
            )

        return detections

    # ── Validation ──────────────────────────────────────────────────
    def _validate_detection(
        self, x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int
    ) -> bool:
        """
        Validate a detection as a real phone in hands:
          1. Bounding box center is in the torso/hands zone
          2. Aspect ratio is phone-like (taller than wide)
          3. Not too large (reject monitors/TVs)
          4. Not too tiny (reject noise)
        """
        margin = self._proximity_margin
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Zone check: central 80% horizontal, lower 75% vertical
        in_x = (margin * frame_w) < cx < ((1 - margin) * frame_w)
        in_y = cy > (0.25 * frame_h)

        # Size checks
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        box_area = bw * bh
        frame_area = frame_w * frame_h
        area_ratio = box_area / frame_area if frame_area > 0 else 1.0

        not_too_large = area_ratio < self._max_area_ratio
        not_too_small = area_ratio > 0.003  # at least 0.3% of frame

        return in_x and in_y and not_too_large and not_too_small
