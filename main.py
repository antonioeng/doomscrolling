"""
main.py — Doomscrolling Detector entry point

Camera capture loop → InferenceEngine → DoomLogic → Audio trigger.
Optional debug preview window (toggle with config.debug.show_preview).

Features:
  • Centered, resizable camera window
  • Image overlay (bottom-right) while doomscrolling
  • Eye gaze + head pose debug info
  • Looping audio controlled by DoomLogic

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
# Load overlay image
# ────────────────────────────────────────────────────────────────────
def load_overlay_image(config: dict) -> np.ndarray | None:
    """Load and resize the overlay image from config."""
    overlay_cfg = config.get("overlay", {})
    img_path = overlay_cfg.get("image_path", "")
    size = overlay_cfg.get("size", 200)

    if not img_path:
        return None

    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Overlay image not found: %s", img_path)
            return None

        # Resize keeping aspect ratio, fitting within size x size
        h, w = img.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info("Overlay image loaded: %s (%dx%d)", img_path, new_w, new_h)
        return img
    except Exception as e:
        logger.warning("Failed to load overlay image: %s — %s", img_path, e)
        return None


def apply_overlay(frame: np.ndarray, overlay_img: np.ndarray, margin: int = 15) -> np.ndarray:
    """Paste overlay image in the bottom-right corner of the frame."""
    fh, fw = frame.shape[:2]
    oh, ow = overlay_img.shape[:2]

    x = fw - ow - margin
    y = fh - oh - margin

    if x < 0 or y < 0:
        return frame

    if overlay_img.shape[2] == 4:
        # Image with alpha channel — blend
        alpha = overlay_img[:, :, 3:4] / 255.0
        bgr = overlay_img[:, :, :3]
        roi = frame[y : y + oh, x : x + ow]
        blended = (bgr * alpha + roi * (1 - alpha)).astype(np.uint8)
        frame[y : y + oh, x : x + ow] = blended
    else:
        frame[y : y + oh, x : x + ow] = overlay_img

    return frame


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

    # ── Eye gaze info ───────────────────────────────────────────────
    if debug_cfg.get("show_eye_gaze", True):
        gaze_ratio = getattr(analysis.pose, "eye_gaze_ratio", -1.0)
        eyes_down = getattr(analysis.pose, "eyes_down", False)
        looking_down = getattr(analysis.pose, "looking_down", False)

        eye_color = (0, 0, 255) if eyes_down else (0, 255, 0)
        gaze_text = f"Gaze: {gaze_ratio:.3f}  {'[EYES DOWN]' if eyes_down else '[OK]'}"
        cv2.putText(
            overlay, gaze_text, (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, eye_color, 1, cv2.LINE_AA,
        )

        # Combined verdict
        combined_color = (0, 0, 255) if looking_down else (0, 255, 0)
        cv2.putText(
            overlay,
            f"Looking Down: {'YES' if looking_down else 'NO'}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            combined_color,
            1,
            cv2.LINE_AA,
        )
        phone_y = 125
    else:
        phone_y = 75

    # ── Phone detections ────────────────────────────────────────────
    if debug_cfg.get("show_phone_bbox", True):
        for det in analysis.phone.detections:
            color = (0, 255, 0) if det.in_torso_zone else (0, 100, 255)
            cv2.rectangle(overlay, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            class_id = getattr(det, "class_id", 67)
            label = f"cls{class_id} {det.confidence:.0%}"
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
        (10, phone_y),
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
    is_active = doom_status.get("is_active", False)

    state_colors = {
        "IDLE": (180, 180, 180),
        "TRACKING": (0, 180, 255),
        "TRIGGERED": (0, 0, 255),
    }
    sc = state_colors.get(state, (255, 255, 255))

    state_label = state
    if is_active:
        state_label = "TRIGGERED 🔊"

    cv2.putText(
        overlay,
        f"State: {state_label}",
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
            f"{elapsed:.1f}s / {threshold:.1f}s",
            (bx + bar_w + 10, by + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if doom_status.get("is_active", False):
        text = "!! TRABAJA !!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.8
        thickness = 4
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        tx = (w - tw) // 2
        ty = h - 50
        # Black outline for readability
        cv2.putText(overlay, text, (tx, ty), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        # Red text
        cv2.putText(overlay, text, (tx, ty), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

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
# Center window on screen
# ────────────────────────────────────────────────────────────────────
def center_window(window_name: str, win_w: int, win_h: int) -> None:
    """Move the OpenCV window to the center of the primary monitor."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        cv2.moveWindow(window_name, x, y)
        logger.info("Window centered: (%d, %d) on %dx%d screen", x, y, screen_w, screen_h)
    except Exception as e:
        logger.debug("Could not center window: %s", e)


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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 960))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))
    cap.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))

    if not cap.isOpened():
        logger.error("Cannot open camera device %d", cam_cfg.get("device_index", 0))
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    logger.info("Camera opened: %dx%d @ %dfps", actual_w, actual_h, actual_fps)

    # ── Init pipeline ───────────────────────────────────────────────
    engine = InferenceEngine(config)
    doom = DoomLogic(config)

    # ── Load overlay image ──────────────────────────────────────────
    overlay_img = load_overlay_image(config)
    overlay_margin = config.get("overlay", {}).get("margin", 15)

    fps_counter = 0
    fps_time = time.monotonic()
    current_fps = 0.0
    window_centered = False

    WINDOW_NAME = "Doomscrolling Detector"

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

            # ── Fusion logic (use combined looking_down) ────────────
            looking_down = getattr(analysis.pose, "looking_down", analysis.pose.head_down)
            doom_status = doom.update(
                looking_down=looking_down,
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

                # Overlay image in bottom-right when doomscrolling is active
                is_active = doom_status.get("is_active", False)
                if is_active and overlay_img is not None:
                    debug_frame = apply_overlay(debug_frame, overlay_img, overlay_margin)

                cv2.imshow(WINDOW_NAME, debug_frame)

                # Center window on first frame
                if not window_centered:
                    center_window(WINDOW_NAME, actual_w, actual_h)
                    window_centered = True

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
