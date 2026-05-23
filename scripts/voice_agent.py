"""VoiceAgent — a self-contained, full-duplex ASR→LLM→TTS loop for one role.

Pure offline. No Flask, no SocketIO. Designed to be driven by an external
audio carrier (Telegram via pytgcalls in `run_duet.py`):

  - external code calls `agent.feed_16k_pcm(bytes)` whenever inbound PCM arrives
  - external code drains `agent.pop_48k_frame(n)` (or iterates) to get TTS PCM
    bytes to send out as the speaker

Models (vad / asr / llm / tts) are passed in at construction so two agents
in one process can share the same loaded weights.

Per-turn events are appended to `agent.turns` as dicts:
  {"t": ts, "role": "user"|"assistant", "text": str,
   "latency": {"asr_ms": int, "llm_ms": int, "tts_ttfa_ms": int, "tts_total_ms": int}}
"""
from __future__ import annotations

import asyncio
import audioop
import concurrent.futures
import fcntl
import io
import os
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Pull TextChunker from lintvoiceagent — same phrase-splitting logic that
# makes app.py's TTS start streaming audio while the LLM is still generating.
_LINT = Path(__file__).resolve().parent.parent / "lintvoiceagent"
if str(_LINT) not in sys.path:
    sys.path.insert(0, str(_LINT))
from streaming_tts import TextChunker  # noqa: E402

import re as _re


def _norm_for_echo(s: str) -> str:
    """Normalize text for echo comparison: lowercase, alphanum + spaces only."""
    return " ".join("".join(c if c.isalnum() else " " for c in s.lower()).split())


def _word_overlap(a: str, b: str) -> float:
    """Fraction of `a`'s words that appear in `b` (length-weighted)."""
    if not a:
        return 0.0
    aw = a.split()
    if not aw:
        return 0.0
    bw_set = set(b.split())
    return sum(1 for w in aw if w in bw_set) / len(aw)


class FastChunker:
    """Aggressive phrase chunker for low TTFA.

    Yields chunks on ANY of: . , ; : ? ! — - — when the buffer has at least
    MIN_CHARS chars. That trips TTS on the first comma/dash a few tokens into
    the LLM stream rather than waiting for ~10 chars + sentence-end.
    """
    BREAK_RE = _re.compile(r"[\.,;:!?—–\-\n]+\s*")
    MIN_CHARS = 3

    def __init__(self):
        self.buf = ""

    def add_token(self, tok: str):
        self.buf += tok
        while True:
            if len(self.buf) < self.MIN_CHARS:
                return
            m = self.BREAK_RE.search(self.buf)
            if not m:
                return
            end = m.end()
            chunk = self.buf[:end].strip()
            self.buf = self.buf[end:]
            if chunk and len(chunk) >= self.MIN_CHARS:
                yield chunk

    def flush(self):
        tail = self.buf.strip()
        self.buf = ""
        return tail or None


CALL_RATE = 48000
ASR_RATE = 16000
TTS_RATE_DEFAULT = 24000     # Kokoro
FRAME_MS = 10
FRAME_BYTES_48K = int(CALL_RATE * 2 * FRAME_MS / 1000)  # 960

# How many bytes of 16k mono int16 we buffer before running VAD on the tail.
# 100 ms = 1600 samples = 3200 bytes. Tried 50 ms (1600 bytes) — FireRedVAD
# became unreliable on shorter windows and Gemini observed longer silences
# between turns. Reverted.
VAD_CHUNK_BYTES = int(os.environ.get("VAD_CHUNK_BYTES", "3200"))

# Minimum bytes of accumulated user utterance to bother sending to ASR.
# 1.5 s @ 16k mono = 48000 bytes. Anything shorter is almost certainly noise
# or a fragment that Qwen-ASR will hallucinate "The." / "Thank you." onto.
MIN_UTTERANCE_BYTES = int(os.environ.get("MIN_UTTERANCE_BYTES", "48000"))

# Minimum RMS of the utterance buffer to bother running ASR.
# Below this the audio is silence / line noise and Qwen-ASR hallucinates.
MIN_UTTERANCE_RMS = float(os.environ.get("MIN_UTTERANCE_RMS", "0.005"))

# Qwen3-ASR-0.6B's stock phantom outputs on near-silence / fragments.
# If the model returns one of these we drop the turn.
ASR_JUNK_PATTERNS = {
    "the.", "thank you.", "thanks.", "you.", "okay.", "ok.", "um.", "uh.",
    "hmm.", "yeah.", ".", ",", "so.", "...",
    "the", "thank you", "thanks", "you", "okay", "ok", "um", "uh",
    "hmm", "yeah", "so",
    "thank you for watching.", "thanks for watching.",  # common Qwen freebie
}

# How long after our own TTS finishes before we trust inbound speech again.
TTS_TAIL_GUARD_S = float(os.environ.get("TTS_TAIL_GUARD_S", "0.3"))

# How long to ignore inbound at the very start of the call (drains the
# initial connection-noise burst that Telegram produces).
INITIAL_DEAFEN_S = float(os.environ.get("INITIAL_DEAFEN_S", "1.0"))


@dataclass
class TurnRecord:
    t: float
    role: str
    text: str
    latency_ms: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        return {"t": self.t, "role": self.role, "text": self.text,
                "latency_ms": self.latency_ms}


class VoiceAgent:
    def __init__(
        self,
        *,
        name: str,
        system_prompt: str,
        vad,                # FireRedVAD instance
        asr_model,          # mlx_audio.stt loaded model
        asr_generate,       # callable: generate_transcription
        llm_model,
        llm_tokenizer,
        llm_stream_generate,  # callable: stream_generate
        tts,                # KokoroStreamingTTS instance
        mlx_executor: concurrent.futures.Executor,  # single-thread pool that owns MLX
        tts_rate: int = TTS_RATE_DEFAULT,
        max_llm_tokens: int = 256,
        log,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.vad = vad
        self.asr_model = asr_model
        self.asr_generate = asr_generate
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.llm_stream_generate = llm_stream_generate
        self.tts = tts
        self.mlx_executor = mlx_executor
        self.tts_rate = tts_rate
        # Prompt cache for KV-state reuse across turns.
        try:
            from mlx_lm.models.cache import make_prompt_cache
            self.prompt_cache = make_prompt_cache(llm_model)
        except Exception:
            self.prompt_cache = None
        self.max_llm_tokens = max_llm_tokens
        self.log = log

        # Conversation history (excluding system prompt).
        self.messages: list[dict] = []
        self.turns: list[TurnRecord] = []

        # Inbound PCM buffers (16 kHz mono s16le).
        self._utterance_buf = bytearray()   # accumulating user speech bytes
        self._vad_tail = bytearray()        # rolling tail for VAD decisions
        self._inbound_lock = asyncio.Lock()

        # Outbound playback (48 kHz mono s16le).
        # threading.Lock (not asyncio): MLX-thread writes phrases while the
        # asyncio player task pops 10ms frames. Critical section is microsec.
        self._playback_buf = bytearray()
        self._playback_lock = threading.Lock()
        self._tts_resample_state = None

        # State flags.
        self._is_speaking = False            # we are currently outputting TTS
        self._turn_in_progress = False       # an ASR→LLM→TTS cycle is running
        self._closed = False
        self._first_turn_done = False
        self._call_started_at: float | None = None  # set when first PCM arrives
        # Barge-in: when peer audio fires during our TTS, we set this to True
        # — the streaming TTS loop checks it between phrases and bails out,
        # and the playback buffer is cleared so the user hears us stop.
        self._bargein_requested = False
        # Track last assistant text for echo suppression of self-bleed-back.
        self._last_assistant_text = ""

        # Half-duplex floor coordination: only one agent speaks at a time.
        # The shared floor file is created when SET via env DUET_FLOOR_PATH.
        # Whoever holds the OS-level exclusive flock is the floor-holder.
        self._floor_path = os.environ.get("DUET_FLOOR_PATH")
        self._floor_fd: int | None = None

    # ------------------------------------------------------------------ inbound

    async def feed_16k_pcm(self, chunk: bytes) -> None:
        """Receive ~chunk of inbound s16le mono 16 kHz PCM from peer."""
        if self._closed:
            return
        # While we're speaking, run a fast energy-based barge-in check.
        # If the peer is genuinely talking (RMS comfortably above floor),
        # cancel our in-flight TTS so they can take the floor.
        if self._is_speaking:
            try:
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                if samples.size > 0:
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    # Threshold deliberately strict — we don't want our own
                    # echo to barge-in. Real peer speech registers >> echo.
                    if rms > 0.02 and not self._bargein_requested:
                        self.log(f"[{self.name}] BARGE-IN detected (rms={rms:.3f}) — stopping TTS")
                        self._bargein_requested = True
                        with self._playback_lock:
                            self._playback_buf.clear()
            except Exception:
                pass
            return
        # Deafen briefly at the start of the call to skip connection-init noise
        # that Telegram emits before either side actually speaks.
        if self._call_started_at is None:
            self._call_started_at = time.time()
        if time.time() - self._call_started_at < INITIAL_DEAFEN_S:
            return

        async with self._inbound_lock:
            # Accumulate everything; track a saw_speech flag and a silence
            # counter from the VAD's view of the rolling tail. A real turn-end
            # = sustained silence AND we saw speech at some point AND the
            # utterance has the size + energy of real speech.
            self._utterance_buf.extend(chunk)
            self._vad_tail.extend(chunk)
            if len(self._vad_tail) > VAD_CHUNK_BYTES * 8:
                del self._vad_tail[: len(self._vad_tail) - VAD_CHUNK_BYTES * 8]
            if len(self._vad_tail) < VAD_CHUNK_BYTES:
                return
            tail_np = np.frombuffer(bytes(self._vad_tail[-VAD_CHUNK_BYTES:]),
                                    dtype=np.int16).astype(np.float32) / 32768.0
            try:
                speech_now = bool(self.vad.has_speech(tail_np))
            except Exception as e:
                self.log(f"[{self.name}] VAD error: {e}")
                return

            if speech_now:
                self._saw_speech = True
                self._silence_counter = 0
                return

            self._silence_counter = getattr(self, "_silence_counter", 0) + 1
            silence_threshold = self.vad.silence_threshold

            if self._silence_counter < silence_threshold:
                return
            # Sustained silence — decide if this utterance is real.
            saw_speech = getattr(self, "_saw_speech", False)
            if not saw_speech or len(self._utterance_buf) < MIN_UTTERANCE_BYTES:
                # No real speech this utterance — reset cleanly.
                self._utterance_buf.clear()
                self._vad_tail.clear()
                self._silence_counter = 0
                self._saw_speech = False
                return
            ut_np = np.frombuffer(bytes(self._utterance_buf), dtype=np.int16)
            ut_rms = float(np.sqrt(np.mean((ut_np.astype(np.float32) / 32768.0) ** 2)))
            if ut_rms < MIN_UTTERANCE_RMS:
                self.log(f"[{self.name}] drop low-energy utterance "
                         f"(rms={ut_rms:.4f}, {len(self._utterance_buf)} bytes)")
                self._utterance_buf.clear()
                self._vad_tail.clear()
                self._silence_counter = 0
                self._saw_speech = False
                return
            self.log(f"[{self.name}] turn-complete confirmed "
                     f"({len(self._utterance_buf)} bytes rms={ut_rms:.4f})")
            utterance = bytes(self._utterance_buf)
            self._utterance_buf.clear()
            self._vad_tail.clear()
            self._silence_counter = 0
            self._saw_speech = False

        # Run the pipeline outside the lock so feed_16k_pcm can keep buffering
        # (it will be ignored because _is_speaking flips True below).
        if self._turn_in_progress:
            return
        self._turn_in_progress = True
        try:
            await self._run_turn(utterance)
        finally:
            self._turn_in_progress = False

    def _try_acquire_floor(self) -> bool:
        """Non-blocking exclusive flock on the shared floor file. Returns True
        if we got the floor (other agent is not speaking)."""
        if not self._floor_path:
            return True  # no coordination configured → free-for-all
        try:
            self._floor_fd = os.open(self._floor_path, os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(self._floor_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if self._floor_fd is not None:
                try: os.close(self._floor_fd)
                except Exception: pass
                self._floor_fd = None
            return False
        except Exception as e:
            self.log(f"[{self.name}] floor acquire error: {e}")
            return True   # fall through to free-for-all on errors

    def _release_floor(self):
        if self._floor_fd is not None:
            try: fcntl.flock(self._floor_fd, fcntl.LOCK_UN)
            except Exception: pass
            try: os.close(self._floor_fd)
            except Exception: pass
            self._floor_fd = None

    async def _run_turn(self, utterance_pcm_16k: bytes) -> None:
        # ASR (must run on MLX-owning thread — model is bound to it)
        t0 = time.time()
        loop = asyncio.get_running_loop()
        asr_text = await loop.run_in_executor(
            self.mlx_executor, self._asr_blocking, utterance_pcm_16k)
        asr_ms = int((time.time() - t0) * 1000)
        if not asr_text or not asr_text.strip():
            self.log(f"[{self.name}] ASR empty — skipping turn")
            return
        if asr_text.strip().lower() in ASR_JUNK_PATTERNS:
            self.log(f"[{self.name}] ASR junk {asr_text!r} — skipping turn")
            return
        # Echo suppression: if ASR text is mostly a substring of what WE just
        # said, it's our own audio bleeding back through the call. Drop it.
        if self._last_assistant_text:
            a = _norm_for_echo(asr_text)
            b = _norm_for_echo(self._last_assistant_text)
            if a and b and (a in b or _word_overlap(a, b) >= 0.6):
                self.log(f"[{self.name}] echo of own TTS suppressed: {asr_text!r}")
                return
        self.log(f"[{self.name}] HEARD: {asr_text!r}  ({asr_ms} ms)")
        self.turns.append(TurnRecord(time.time(), "user", asr_text,
                                     {"asr_ms": asr_ms}))
        self.messages.append({"role": "user", "content": asr_text})

        # LLM + TTS
        await self._speak_response(asr_ms_for_log=asr_ms)

    def _asr_blocking(self, pcm_16k: bytes) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(ASR_RATE)
                wf.writeframes(pcm_16k)
            result = self.asr_generate(model=self.asr_model, audio=tmp.name,
                                       format="txt", verbose=False)
            text = getattr(result, "text", "") or ""
            return text.strip()
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    # ------------------------------------------------------------- LLM + speak

    async def _speak_response(self, *, asr_ms_for_log: Optional[int] = None) -> None:
        """Streaming LLM → phrase chunks → TTS → playback buffer.

        The whole loop runs on the MLX executor thread (single thread that
        owns model state). As each phrase comes out of the chunker we
        immediately TTS it and write the PCM into the playback buffer, so the
        first audio reaches the line ~one phrase after the LLM's first token
        rather than after the full generation.
        """
        # Half-duplex was attempted via fcntl.flock but caused turn-blocking
        # bugs locally; reverted. Self-echo is prevented by `_is_speaking`.
        self._bargein_requested = False
        prompt = self._build_prompt()
        self._is_speaking = True
        t_start = time.time()

        state = {
            "full_text": "",
            "first_tok_t": None,
            "tts_first_audio_t": None,
            "phrases_spoken": 0,
            "tts_pcm_48k_bytes": 0,
        }

        def _emit_phrase(phrase: str):
            """Synthesize one phrase and push 48k PCM into the playback buffer."""
            if self._bargein_requested:
                return
            phrase = phrase.strip()
            if not phrase:
                return
            try:
                for wav_bytes in self.tts.generate_audio_chunk(phrase):
                    if self._bargein_requested:
                        return
                    pcm = self._wav_to_pcm(wav_bytes)
                    if not pcm:
                        continue
                    pcm_48k, self._tts_resample_state = audioop.ratecv(
                        pcm, 2, 1, self.tts_rate, CALL_RATE,
                        self._tts_resample_state)
                    with self._playback_lock:
                        self._playback_buf.extend(pcm_48k)
                    state["tts_pcm_48k_bytes"] += len(pcm_48k)
                    if state["tts_first_audio_t"] is None:
                        state["tts_first_audio_t"] = time.time()
                state["phrases_spoken"] += 1
            except Exception as e:
                self.log(f"[{self.name}] TTS phrase error: {e}")

        def _stream_llm_to_tts():
            chunker = FastChunker()
            kw = {"max_tokens": self.max_llm_tokens}
            if self.prompt_cache is not None:
                kw["prompt_cache"] = self.prompt_cache
            for resp in self.llm_stream_generate(
                self.llm_model, self.llm_tokenizer,
                prompt=prompt, **kw,
            ):
                if self._bargein_requested:
                    break
                tok = resp.text if hasattr(resp, "text") else str(resp)
                if state["first_tok_t"] is None:
                    state["first_tok_t"] = time.time()
                state["full_text"] += tok
                for phrase in chunker.add_token(tok):
                    if self._bargein_requested:
                        break
                    _emit_phrase(phrase)
                if self._bargein_requested:
                    break
            if not self._bargein_requested:
                tail = chunker.flush()
                if tail:
                    _emit_phrase(tail)

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self.mlx_executor, _stream_llm_to_tts)
        except Exception as e:
            self.log(f"[{self.name}] streaming error: {e}")

        text = state["full_text"].strip()
        if not text:
            self.log(f"[{self.name}] LLM empty")
            self._is_speaking = False
            return
        self.messages.append({"role": "assistant", "content": text})
        self._last_assistant_text = text
        llm_ttft_ms = (int((state["first_tok_t"] - t_start) * 1000)
                       if state["first_tok_t"] else -1)
        tts_ttfa_ms = (int((state["tts_first_audio_t"] - t_start) * 1000)
                       if state["tts_first_audio_t"] else -1)
        self.log(f"[{self.name}] SAYS:  {text!r}  "
                 f"(llm_ttft={llm_ttft_ms}ms tts_ttfa={tts_ttfa_ms}ms)")

        # Wait for the playback buffer to drain so we don't mute mid-utterance.
        while True:
            with self._playback_lock:
                remaining = len(self._playback_buf)
            if remaining <= FRAME_BYTES_48K or self._closed or self._bargein_requested:
                break
            await asyncio.sleep(0.05)
        # Brief tail guard before unmuting inbound — only if we weren't barged-in.
        if not self._bargein_requested:
            await asyncio.sleep(TTS_TAIL_GUARD_S)
        else:
            # Barge-in already happened — user wants to speak. Unmute fast.
            await asyncio.sleep(0.05)
        self._is_speaking = False
        self._release_floor()

        total_ms = int((time.time() - t_start) * 1000)
        self.turns.append(TurnRecord(
            time.time(), "assistant", text,
            {"llm_ttft_ms": llm_ttft_ms,
             "tts_ttfa_ms": tts_ttfa_ms,
             "total_ms": total_ms,
             "phrases": state["phrases_spoken"],
             "tts_pcm_48k_bytes": state["tts_pcm_48k_bytes"]},
        ))

    def _build_prompt(self) -> str:
        msgs = [{"role": "system", "content": self.system_prompt}] + self.messages
        tok = self.llm_tokenizer
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        return ("System: " + self.system_prompt + "\n" +
                "\n".join(f"{m['role']}: {m['content']}" for m in self.messages) +
                "\nassistant:")

    @staticmethod
    def _wav_to_pcm(wav_bytes: bytes) -> bytes:
        """Kokoro yields full WAV containers per chunk — strip header to raw PCM."""
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                return wf.readframes(wf.getnframes())
        except Exception:
            return b""

    # ----------------------------------------------------- outbound (playback)

    async def pop_48k_frame(self) -> bytes:
        """Return one 10ms 48k mono frame for `send_frame`. Returns silence if empty."""
        with self._playback_lock:
            if len(self._playback_buf) >= FRAME_BYTES_48K:
                chunk = bytes(self._playback_buf[:FRAME_BYTES_48K])
                del self._playback_buf[:FRAME_BYTES_48K]
                return chunk
        return b"\x00" * FRAME_BYTES_48K

    # ----------------------------------------------------------------- kickoff

    async def kickoff(self, opener_text: str) -> None:
        """Speak `opener_text` as the first assistant turn, no LLM call.

        Used on the initiator side (katey) so she greets first. The text is
        appended to messages as our assistant turn so subsequent LLM calls
        see the conversation flow correctly.
        """
        self.log(f"[{self.name}] kickoff opener: {opener_text!r}")
        self.messages.append({"role": "assistant", "content": opener_text})
        self._last_assistant_text = opener_text
        self._bargein_requested = False
        # Kickoff is the first turn — claim the floor before speaking.
        self._try_acquire_floor()
        # Synthesize and queue for playback.
        self._is_speaking = True
        t_tts_start = time.time()
        tts_ttfa_ms = None
        try:
            loop = asyncio.get_running_loop()
            wav_chunks = await loop.run_in_executor(
                self.mlx_executor,
                lambda: list(self.tts.generate_audio_chunk(opener_text)),
            )
            for wav_bytes in wav_chunks:
                pcm = self._wav_to_pcm(wav_bytes)
                if not pcm:
                    continue
                pcm_48k, self._tts_resample_state = audioop.ratecv(
                    pcm, 2, 1, self.tts_rate, CALL_RATE, self._tts_resample_state)
                with self._playback_lock:
                    self._playback_buf.extend(pcm_48k)
                if tts_ttfa_ms is None:
                    tts_ttfa_ms = int((time.time() - t_tts_start) * 1000)
        except Exception as e:
            self.log(f"[{self.name}] kickoff TTS error: {e}")

        # Wait for playback to drain.
        while True:
            with self._playback_lock:
                remaining = len(self._playback_buf)
            if remaining <= FRAME_BYTES_48K or self._closed:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(TTS_TAIL_GUARD_S)
        self._is_speaking = False
        self._release_floor()

        self.turns.append(TurnRecord(
            time.time(), "assistant", opener_text,
            {"llm_ms": 0, "tts_ttfa_ms": tts_ttfa_ms or -1,
             "tts_total_ms": int((time.time() - t_tts_start) * 1000),
             "kickoff": True},
        ))

    # ------------------------------------------------------------------ shutdown

    async def close(self) -> None:
        self._closed = True
