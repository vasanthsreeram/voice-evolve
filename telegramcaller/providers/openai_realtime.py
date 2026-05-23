"""OpenAI Realtime provider — pcm16 24 kHz mono both ways.

Docs: https://platform.openai.com/docs/guides/realtime
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


class OpenAIRealtimeProvider:
    name = "openai"
    input_rate = 24000
    output_rate = 24000

    def __init__(self) -> None:
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
        self.voice = os.environ.get("OPENAI_REALTIME_VOICE", "alloy")
        self.ws = None

    async def open(self) -> None:
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        self.ws = await websockets.connect(
            url,
            additional_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        )
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": SYSTEM_PROMPT,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad"},
                "input_audio_transcription": {"model": "whisper-1"},
            },
        }))

    async def send_audio(self, pcm16: bytes) -> None:
        if not self.ws:
            return
        await self.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("ascii"),
        }))

    async def recv(self) -> AsyncIterator[dict]:
        assert self.ws is not None
        async for msg in self.ws:
            d = json.loads(msg)
            t = d.get("type", "")
            if t == "response.audio.delta":
                yield {"type": "audio", "pcm": base64.b64decode(d["delta"])}
            elif t == "input_audio_buffer.speech_started":
                yield {"type": "interrupt"}
            elif t == "conversation.item.input_audio_transcription.completed":
                yield {"type": "user_text", "text": d.get("transcript", "")}
            elif t == "response.audio_transcript.done":
                yield {"type": "agent_text", "text": d.get("transcript", "")}

    async def close(self) -> None:
        if self.ws:
            try: await self.ws.close()
            except Exception: pass
            self.ws = None
