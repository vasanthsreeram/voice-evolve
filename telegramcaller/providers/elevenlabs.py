"""ElevenLabs Conversational AI websocket provider.

ElevenLabs default audio: 16 kHz PCM16 mono both directions.
"""
from __future__ import annotations
import base64
import json
import os
from typing import AsyncIterator

import websockets

from .base import Provider


class ElevenLabsProvider:
    name = "elevenlabs"
    input_rate = 16000
    output_rate = 16000

    def __init__(self) -> None:
        self.api_key = os.environ["ELEVENLABS_API_KEY"]
        self.agent_id = os.environ["EL_AGENT_ID"]
        self.ws = None

    async def open(self) -> None:
        url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={self.agent_id}"
        self.ws = await websockets.connect(
            url, additional_headers={"xi-api-key": self.api_key}
        )
        await self.ws.send(json.dumps({"type": "conversation_initiation_client_data"}))

    async def send_audio(self, pcm16: bytes) -> None:
        if not self.ws:
            return
        await self.ws.send(json.dumps({
            "user_audio_chunk": base64.b64encode(pcm16).decode("ascii"),
        }))

    async def recv(self) -> AsyncIterator[dict]:
        assert self.ws is not None
        async for msg in self.ws:
            d = json.loads(msg)
            t = d.get("type")
            if t == "audio":
                yield {"type": "audio",
                       "pcm": base64.b64decode(d["audio_event"]["audio_base_64"])}
            elif t == "ping":
                await self.ws.send(json.dumps({
                    "type": "pong",
                    "event_id": d["ping_event"]["event_id"],
                }))
            elif t == "interruption":
                yield {"type": "interrupt"}
            elif t == "user_transcript":
                yield {"type": "user_text",
                       "text": d.get("user_transcription_event", {}).get("user_transcript", "")}
            elif t == "agent_response":
                yield {"type": "agent_text",
                       "text": d.get("agent_response_event", {}).get("agent_response", "")}

    async def close(self) -> None:
        if self.ws:
            try: await self.ws.close()
            except Exception: pass
            self.ws = None
