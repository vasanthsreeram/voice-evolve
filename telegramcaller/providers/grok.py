"""xAI Grok provider — placeholder.

As of the time of writing, xAI has not published a public real-time voice
WebSocket API. If/when they do, fill in `open / send_audio / recv` here
following the OpenAI Realtime template.
"""
from __future__ import annotations
import os
from typing import AsyncIterator

from .base import Provider


class GrokProvider:
    name = "grok"
    input_rate = 16000
    output_rate = 16000

    def __init__(self) -> None:
        raise SystemExit(
            "xAI/Grok does not currently expose a public realtime voice API. "
            "Use VOICE_PROVIDER=elevenlabs|openai|gemini instead, or open a PR "
            "implementing this provider when xAI ships it."
        )

    async def open(self) -> None: ...
    async def send_audio(self, pcm16: bytes) -> None: ...
    async def recv(self) -> AsyncIterator[dict]:
        if False: yield {}
    async def close(self) -> None: ...
