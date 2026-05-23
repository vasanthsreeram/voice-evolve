"""Voice-agent providers for tgcallskill.

Every provider implements the same async interface (see base.Provider) so
bridge.py is agnostic to which one is configured. The active provider is
chosen by the VOICE_PROVIDER env var.
"""
from __future__ import annotations
import os

from .base import Provider


def make_provider() -> Provider:
    kind = os.environ.get("VOICE_PROVIDER", "elevenlabs").lower()
    if kind in ("elevenlabs", "el", "11labs"):
        from .elevenlabs import ElevenLabsProvider
        return ElevenLabsProvider()
    if kind in ("openai", "openai_realtime"):
        from .openai_realtime import OpenAIRealtimeProvider
        return OpenAIRealtimeProvider()
    if kind in ("gemini", "gemini_live", "google"):
        from .gemini_live import GeminiLiveProvider
        return GeminiLiveProvider()
    if kind in ("grok", "xai"):
        from .grok import GrokProvider
        return GrokProvider()
    raise SystemExit(
        f"Unknown VOICE_PROVIDER={kind!r}. "
        "Supported: elevenlabs, openai, gemini, grok."
    )


__all__ = ["Provider", "make_provider"]
