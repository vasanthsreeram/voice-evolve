"""Common interface every voice-agent provider implements."""
from __future__ import annotations
from typing import AsyncIterator, Protocol


class Provider(Protocol):
    name: str
    input_rate: int   # PCM16 mono sample rate this provider wants us to send
    output_rate: int  # PCM16 mono sample rate it emits

    async def open(self) -> None: ...

    async def send_audio(self, pcm16: bytes) -> None:
        """Caller audio in PCM16 mono at `input_rate`."""

    async def recv(self) -> AsyncIterator[dict]:
        """Yield events:
          {'type': 'audio',      'pcm': bytes}   # PCM16 mono at `output_rate`
          {'type': 'user_text',  'text': str}
          {'type': 'agent_text', 'text': str}
          {'type': 'interrupt'}
        """
        if False:
            yield {}

    async def close(self) -> None: ...
