"""
doom_logic.py — Doomscrolling fusion logic

Implements the core state machine:

  IDLE ──(head_down AND phone_detected)──► TRACKING
  TRACKING ──(timer ≥ threshold)──────────► TRIGGERED  → play audio
  TRACKING ──(conditions lost)────────────► IDLE       (with grace period)
  TRIGGERED ──(cooldown elapsed)──────────► IDLE

Features:
  • Grace period: brief interruptions (e.g. blink, hand shift) don't reset
  • Cooldown: prevents audio spam after a trigger
  • Progress callback: optional hook for UI / logging
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable

import pygame

logger = logging.getLogger(__name__)


class _State(Enum):
    IDLE = auto()
    TRACKING = auto()
    COOLDOWN = auto()


class DoomLogic:
    """
    Fusion engine that combines head-pose and phone-detection signals
    into a doomscrolling verdict.
    """

    def __init__(self, config: dict) -> None:
        doom_cfg = config.get("doom_logic", {})
        audio_cfg = config.get("audio", {})

        self._sustained_sec: float = doom_cfg.get("sustained_seconds", 7)
        self._cooldown_sec: float = doom_cfg.get("cooldown_seconds", 60)
        self._grace_ms: float = doom_cfg.get("grace_period_ms", 500)

        self._audio_path: str = audio_cfg.get("file_path", "audio/tienes_que_tlabajal.mp3")
        self._volume: float = audio_cfg.get("volume", 0.85)

        # Internal state
        self._state = _State.IDLE
        self._track_start: float = 0.0
        self._last_valid: float = 0.0
        self._cooldown_start: float = 0.0
        self._trigger_count: int = 0

        # Audio engine
        pygame.mixer.init()
        try:
            self._sound = pygame.mixer.Sound(self._audio_path)
            self._sound.set_volume(self._volume)
            logger.info("Audio loaded: %s", self._audio_path)
        except (pygame.error, FileNotFoundError):
            self._sound = None
            logger.warning(
                "Audio file not found: %s  — triggers will log only.", self._audio_path
            )

        # Optional progress callback: fn(elapsed_sec, threshold_sec)
        self.on_progress: Callable[[float, float], None] | None = None

        logger.info(
            "DoomLogic ready  |  sustained=%.1fs  cooldown=%.1fs  grace=%dms",
            self._sustained_sec,
            self._cooldown_sec,
            self._grace_ms,
        )

    # ── Public API ──────────────────────────────────────────────────
    def update(self, head_down: bool, phone_detected: bool) -> dict:
        """
        Called once per frame. Returns status dict:
          {
            "state": "IDLE" | "TRACKING" | "COOLDOWN",
            "elapsed": float,       # seconds in current tracking
            "threshold": float,     # target seconds
            "progress": float,      # 0.0 – 1.0
            "triggered": bool,      # True the frame audio fires
            "trigger_count": int,
          }
        """
        now = time.monotonic()
        triggered = False

        if self._state == _State.COOLDOWN:
            if now - self._cooldown_start >= self._cooldown_sec:
                self._state = _State.IDLE
                logger.debug("Cooldown expired → IDLE")

        if self._state == _State.IDLE:
            if head_down and phone_detected:
                self._state = _State.TRACKING
                self._track_start = now
                self._last_valid = now
                logger.debug("Conditions met → TRACKING")

        elif self._state == _State.TRACKING:
            if head_down and phone_detected:
                self._last_valid = now
            else:
                # Grace period: allow brief interruptions
                gap_ms = (now - self._last_valid) * 1000.0
                if gap_ms > self._grace_ms:
                    self._state = _State.IDLE
                    logger.debug("Conditions lost (gap=%.0fms) → IDLE", gap_ms)

            if self._state == _State.TRACKING:
                elapsed = now - self._track_start
                if elapsed >= self._sustained_sec:
                    triggered = True
                    self._trigger_count += 1
                    self._play_audio()
                    self._state = _State.COOLDOWN
                    self._cooldown_start = now
                    logger.info(
                        "🚨 DOOMSCROLLING DETECTED (#%d)  elapsed=%.1fs",
                        self._trigger_count,
                        elapsed,
                    )

        elapsed = (
            now - self._track_start if self._state == _State.TRACKING else 0.0
        )
        progress = min(elapsed / self._sustained_sec, 1.0) if self._sustained_sec > 0 else 0.0

        if self.on_progress and self._state == _State.TRACKING:
            self.on_progress(elapsed, self._sustained_sec)

        return {
            "state": self._state.name,
            "elapsed": round(elapsed, 2),
            "threshold": self._sustained_sec,
            "progress": round(progress, 3),
            "triggered": triggered,
            "trigger_count": self._trigger_count,
        }

    def release(self) -> None:
        pygame.mixer.quit()
        logger.info("DoomLogic released. Total triggers: %d", self._trigger_count)

    # ── Private ─────────────────────────────────────────────────────
    def _play_audio(self) -> None:
        if self._sound is not None:
            self._sound.play()
            logger.info("▶ Playing: %s", self._audio_path)
        else:
            logger.warning("▶ TRIGGER (no audio file loaded)")
