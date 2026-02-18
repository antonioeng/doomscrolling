"""
main.py — Doomscrolling Detector entry point

Camera capture loop → InferenceEngine → DoomLogic → Audio trigger.
Optional debug preview window (toggle with config.debug.show_preview).

Usage:
    python main.py                    # default config.json
    python main.py --config my.json   # custom config
    python main.py --no-preview       # headless mode
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import cv2
import numpy as np

from vision.inference_engine import InferenceEngine
from logic.doom_logic import DoomLogic

# ────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────────
# Debug overlay (only when preview enabled)
# ────────────────────────────────────────────────────────────────────
def draw_debug(
    frame: np.ndarray,
    analysis,
    doom_status: dict,
    config: dict,
    fps: float,
) -> np.ndarray:
    """Draw optional debug information on the frame."""
    debug_cfg = config.get("debug", {})
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # ── FPS + inference time ────────────────────────────────────────
    if debug_cfg.get("show_stats", True):
        cv2.putText(
            overlay,
            f"FPS: {fps:.0f}  |  Inf: {analysis.inference_ms:.0f}ms",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

    # ── Face landmarks ──────────────────────────────────────────────
    if debug_cfg.get("show_landmarks", True) and analysis.pose.landmarks_px:
        for i, (lx, ly) in enumerate(analysis.pose.landmarks_px):
            if i % 5 == 0:  # draw every 5th for performance
                cv2.circle(overlay, (lx, ly), 1, (0, 200, 200), -1)

    # ── Head pose info ──────────────────────────────────────────────
    pitch = analysis.pose.pitch_deg
    head_color = (0, 0, 255) if analysis.pose.head_down else (0, 255, 0)
    cv2.putText(
        overlay,
        f"Pitch: {pitch:.1f}deg  {'[DOWN]' if analysis.pose.head_down else '[OK]'}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        head_color,
        1,
        cv2.LINE_AA,
    )

    # ── Phone detections ────────────────────────────────────────────
    if debug_cfg.get("show_phone_bbox", True):
        for det in analysis.phone.detections:
            color = (0, 255, 0) if det.in_torso_zone else (0, 100, 255)
            cv2.rectangle(overlay, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"phone {det.confidence:.0%}"
            if not det.in_torso_zone:
                label += " [OUT]"
            cv2.putText(
                overlay,
                label,
                (det.x1, det.y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    phone_color = (0, 255, 0) if analysis.phone.phone_detected else (100, 100, 100)
    cv2.putText(
        overlay,
        f"Phone: {'YES' if analysis.phone.phone_detected else 'NO'}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        phone_color,
        1,
        cv2.LINE_AA,
    )

    # ── Doom status ─────────────────────────────────────────────────
    state = doom_status["state"]
    progress = doom_status["progress"]
    elapsed = doom_status["elapsed"]
    threshold = doom_status["threshold"]

    state_colors = {
        "IDLE": (180, 180, 180),
        "TRACKING": (0, 180, 255),
        "COOLDOWN": (255, 100, 100),
    }
    sc = state_colors.get(state, (255, 255, 255))

    cv2.putText(
        overlay,
        f"State: {state}",
        (10, h - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        sc,
        2,
        cv2.LINE_AA,
    )

    if state == "TRACKING":
        # Progress bar
        bar_w = 200
        bar_h = 14
        bx, by = 10, h - 35
        cv2.rectangle(overlay, (bx, by), (bx + bar_w, by + bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * progress)
        bar_color = (0, 140, 255) if progress < 0.8 else (0, 0, 255)
        cv2.rectangle(overlay, (bx, by), (bx + fill_w, by + bar_h), bar_color, -1)
        cv2.putText(
            overlay,
            f"{elapsed:.1f}s / {threshold:.0f}s",
            (bx + bar_w + 10, by + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if doom_status["triggered"]:
        cv2.putText(
            overlay,
            "!! TIENES QUE TLABAJAL !!",
            (w // 2 - 180, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    # Trigger count
    cv2.putText(
        overlay,
        f"Triggers: {doom_status['trigger_count']}",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return overlay


# ────────────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Doomscrolling Detector")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--no-preview", action="store_true", help="Disable debug window")
    args = parser.parse_args()

    config = load_config(args.config)

    log_level = config.get("debug", {}).get("log_level", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

    show_preview = (
        not args.no_preview and config.get("debug", {}).get("show_preview", True)
    )

    # ── Init camera ─────────────────────────────────────────────────
    cam_cfg = config.get("camera", {})
    cap = cv2.VideoCapture(cam_cfg.get("device_index", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))
    cap.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))

    if not cap.isOpened():
        logger.error("Cannot open camera device %d", cam_cfg.get("device_index", 0))
        sys.exit(1)

    logger.info(
        "Camera opened: %dx%d @ %dfps",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FPS)),
    )

    # ── Init pipeline ───────────────────────────────────────────────
    engine = InferenceEngine(config)
    doom = DoomLogic(config)

    fps_counter = 0
    fps_time = time.monotonic()
    current_fps = 0.0

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Doomscrolling Detector running. Press 'q' to quit.")
    logger.info("═══════════════════════════════════════════════════")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame capture failed, retrying...")
                time.sleep(0.01)
                continue

            # Flip horizontally for mirror view
            frame = cv2.flip(frame, 1)

            # ── Inference ───────────────────────────────────────────
            analysis = engine.analyse(frame)

            # ── Fusion logic ────────────────────────────────────────
            doom_status = doom.update(
                head_down=analysis.pose.head_down,
                phone_detected=analysis.phone.phone_detected,
            )

            # ── FPS ─────────────────────────────────────────────────
            fps_counter += 1
            now = time.monotonic()
            if now - fps_time >= 1.0:
                current_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            # ── Debug preview ───────────────────────────────────────
            if show_preview:
                debug_frame = draw_debug(frame, analysis, doom_status, config, current_fps)
                cv2.imshow("Doomscrolling Detector", debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User pressed 'q' — shutting down.")
                    break
            else:
                # Headless: still check for KeyboardInterrupt via small sleep
                time.sleep(0.001)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down.")
    finally:
        cap.release()
        engine.release()
        doom.release()
        if show_preview:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
