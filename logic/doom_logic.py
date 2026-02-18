"""
doom_logic.py — Doomscrolling fusion logic

Implements the core state machine:

  IDLE ──(looking_down AND phone_detected for N sec)──► TRIGGERED  → loop audio
  TRIGGERED ──(conditions lost)───────────────────────► IDLE       → stop audio

Features:
  • Grace period: brief interruptions (e.g. blink, hand shift) don't reset
  • Audio loops continuously while doomscrolling, stops when user recovers
  • No cooldown — triggers again immediately if user resumes scrolling
  • "is_triggered" flag stays True the entire time audio is playing
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
    TRIGGERED = auto()


class DoomLogic:
    """
    Fusion engine that combines head-pose, eye-gaze, and phone-detection
    signals into a doomscrolling verdict with looping audio.
    """

    def __init__(self, config: dict) -> None:
        doom_cfg = config.get("doom_logic", {})
        audio_cfg = config.get("audio", {})

        self._sustained_sec: float = doom_cfg.get("sustained_seconds", 2.5)
        self._grace_ms: float = doom_cfg.get("grace_period_ms", 400)

        self._audio_path: str = audio_cfg.get("file_path", "audio/tienes_que_tlabajal.mp3")
        self._volume: float = audio_cfg.get("volume", 0.9)

        # Internal state
        self._state = _State.IDLE
        self._track_start: float = 0.0
        self._last_valid: float = 0.0
        self._trigger_count: int = 0

        # Audio engine
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        self._sound: pygame.mixer.Sound | None = None
        self._channel: pygame.mixer.Channel | None = None
        try:
            self._sound = pygame.mixer.Sound(self._audio_path)
            self._sound.set_volume(self._volume)
            self._channel = pygame.mixer.Channel(0)
            logger.info("Audio loaded: %s", self._audio_path)
        except (pygame.error, FileNotFoundError) as e:
            logger.warning("Audio file not found: %s  — %s", self._audio_path, e)

        # Optional progress callback: fn(elapsed_sec, threshold_sec)
        self.on_progress: Callable[[float, float], None] | None = None

        logger.info(
            "DoomLogic ready  |  sustained=%.1fs  grace=%dms  no_cooldown",
            self._sustained_sec,
            self._grace_ms,
        )

    # ── Public API ──────────────────────────────────────────────────
    def update(self, looking_down: bool, phone_detected: bool) -> dict:
        """
        Called once per frame with combined looking_down signal and phone detection.
        Returns status dict.
        """
        now = time.monotonic()
        just_triggered = False
        conditions_met = looking_down and phone_detected

        if self._state == _State.IDLE:
            if conditions_met:
                self._state = _State.TRACKING
                self._track_start = now
                self._last_valid = now
                logger.debug("Conditions met → TRACKING")

        elif self._state == _State.TRACKING:
            if conditions_met:
                self._last_valid = now
            else:
                gap_ms = (now - self._last_valid) * 1000.0
                if gap_ms > self._grace_ms:
                    self._state = _State.IDLE
                    logger.debug("Conditions lost (gap=%.0fms) → IDLE", gap_ms)

            if self._state == _State.TRACKING:
                elapsed = now - self._track_start
                if elapsed >= self._sustained_sec:
                    just_triggered = True
                    self._trigger_count += 1
                    self._state = _State.TRIGGERED
                    self._last_valid = now
                    self._start_audio_loop()
                    logger.info(
                        "🚨 DOOMSCROLLING DETECTED (#%d)  elapsed=%.1fs",
                        self._trigger_count, elapsed,
                    )

        elif self._state == _State.TRIGGERED:
            if conditions_met:
                self._last_valid = now
            else:
                gap_ms = (now - self._last_valid) * 1000.0
                if gap_ms > self._grace_ms:
                    self._state = _State.IDLE
                    self._stop_audio()
                    logger.info("User stopped doomscrolling → IDLE (audio stopped)")

        # Compute progress for TRACKING state
        elapsed = 0.0
        if self._state == _State.TRACKING:
            elapsed = now - self._track_start
        elif self._state == _State.TRIGGERED:
            elapsed = self._sustained_sec  # full

        progress = min(elapsed / self._sustained_sec, 1.0) if self._sustained_sec > 0 else 0.0

        if self.on_progress and self._state == _State.TRACKING:
            self.on_progress(elapsed, self._sustained_sec)

        return {
            "state": self._state.name,
            "elapsed": round(elapsed, 2),
            "threshold": self._sustained_sec,
            "progress": round(progress, 3),
            "triggered": just_triggered,
            "is_active": self._state == _State.TRIGGERED,
            "trigger_count": self._trigger_count,
        }

    def release(self) -> None:
        self._stop_audio()
        pygame.mixer.quit()
        logger.info("DoomLogic released. Total triggers: %d", self._trigger_count)

    # ── Private ─────────────────────────────────────────────────────
    def _start_audio_loop(self) -> None:
        """Start looping the audio indefinitely."""
        if self._sound is not None and self._channel is not None:
            self._channel.play(self._sound, loops=-1)  # -1 = infinite loop
            logger.info("▶ Looping audio: %s", self._audio_path)
        else:
            logger.warning("▶ TRIGGER (no audio file loaded)")

    def _stop_audio(self) -> None:
        """Stop the looping audio."""
        if self._channel is not None and self._channel.get_busy():
            self._channel.fadeout(300)  # 300ms fade out
            logger.info("⏹ Audio stopped")
