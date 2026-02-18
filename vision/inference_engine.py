"""
inference_engine.py — Unified inference orchestrator

Runs pose estimation and phone detection in parallel using
concurrent.futures.ThreadPoolExecutor, collects results, and returns
a combined FrameAnalysis.

Thread-safety: each model owns its own session/state and is only
called from one thread at a time (one-shot submit per frame).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass

import numpy as np

from vision.pose_estimator import PoseEstimator, PoseResult
from vision.phone_detector import PhoneDetector, PhoneResult

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysis:
    """Combined result of one frame's analysis pipeline."""

    pose: PoseResult
    phone: PhoneResult
    inference_ms: float = 0.0


class InferenceEngine:
    """
    Orchestrates parallel inference of pose + phone detection.

    Uses a 2-thread pool so that both models can run simultaneously,
    maximising throughput on multi-core CPUs and allowing NPU/GPU
    overlap when using separate ONNX sessions.
    """

    def __init__(self, config: dict) -> None:
        self._pose = PoseEstimator(config)
        self._phone = PhoneDetector(config)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")
        logger.info("InferenceEngine initialised (parallel mode).")

    def analyse(self, frame: np.ndarray) -> FrameAnalysis:
        """Submit both models in parallel, await results."""
        t0 = time.perf_counter()

        fut_pose: Future[PoseResult] = self._pool.submit(self._pose.estimate, frame)
        fut_phone: Future[PhoneResult] = self._pool.submit(self._phone.detect, frame)

        pose_result = fut_pose.result()
        phone_result = fut_phone.result()

        elapsed = (time.perf_counter() - t0) * 1000.0

        return FrameAnalysis(
            pose=pose_result,
            phone=phone_result,
            inference_ms=round(elapsed, 1),
        )

    def release(self) -> None:
        self._pool.shutdown(wait=False)
        self._pose.release()
        self._phone.release()
        logger.info("InferenceEngine released.")
