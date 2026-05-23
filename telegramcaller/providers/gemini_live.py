"""Gemini Live API provider — pcm16 16 kHz in, 24 kHz out.

Docs: https://ai.google.dev/api/multimodal-live
"""
from __future__ import annotations
import base64
import json
import os
from typing import AsyncIterator

import websockets

from .base import Provider

SYSTEM_PROMPT = os.environ.get(
    "VOICE_SYSTEM_PROMPT",
    "You are answering an incoming phone call. Be friendly, concise, and useful.",
)


class GeminiLiveProvider:
    name = "gemini"
    input_rate = 16000
    output_rate = 24000

    def __init__(self) -> None:
        self.api_key = os.environ["GOOGLE_API_KEY"]
        self.model = os.environ.get("GEMINI_LIVE_MODEL", "models/gemini-2.0-flash-exp")
        self.voice = os.environ.get("GEMINI_LIVE_VOICE", "Aoede")
        self.ws = None

    async def open(self) -> None:
        url = (
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
            f"?key={self.api_key}"
        )
        self.ws = await websockets.connect(url)
        await self.ws.send(json.dumps({
            "setup": {
                "model": self.model,
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {"voice_name": self.voice}
                        }
                    },
                },
                "system_instruction": {
                    "parts": [{"text": SYSTEM_PROMPT}],
                },
            }
        }))

    async def send_audio(self, pcm16: bytes) -> None:
        if not self.ws:
            return
        await self.ws.send(json.dumps({
            "realtime_input": {
                "media_chunks": [{
                    "mime_type": "audio/pcm;rate=16000",
                    "data": base64.b64encode(pcm16).decode("ascii"),
                }]
            }
        }))

    async def recv(self) -> AsyncIterator[dict]:
        assert self.ws is not None
        async for msg in self.ws:
            d = json.loads(msg) if isinstance(msg, str) else json.loads(msg.decode())
            sc = d.get("serverContent") or {}
            if sc.get("interrupted"):
                yield {"type": "interrupt"}
            mt = sc.get("modelTurn") or {}
            for part in mt.get("parts", []):
                inline = part.get("inlineData") or {}
                data = inline.get("data")
                if data and (inline.get("mimeType", "").startswith("audio/")):
                    yield {"type": "audio", "pcm": base64.b64decode(data)}
                text = part.get("text")
                if text:
                    yield {"type": "agent_text", "text": text}

    async def close(self) -> None:
        if self.ws:
            try: await self.ws.close()
            except Exception: pass
            self.ws = None
