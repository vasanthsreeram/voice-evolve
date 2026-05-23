#!/usr/bin/env python3
"""Per-session conversation logger.

Writes for each turn:
  - logs/<sid>/user_NNN.wav         (user utterance, 16kHz mono PCM)
  - logs/<sid>/assistant_NNN.wav    (assistant TTS output, 16kHz mono PCM)
  - logs/<sid>/turns.jsonl          (transcript + latency breakdown)

Latency stages tracked per turn:
  asr_ms        — ASR transcription time (final pass)
  llm_ttft_ms   — LLM first-token time (after ASR finishes)
  llm_total_ms  — total LLM generation time
  tts_ttft_ms   — time to first TTS audio chunk (after LLM start)
  tts_total_ms  — total TTS generation time across all chunks
  total_ms      — end-to-end (user finishes speaking → last TTS chunk emitted)
"""

import io
import json
import os
import time
import wave
from threading import Lock
from collections import defaultdict, deque

import numpy as np

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def _ms(t_start, t_end):
    return None if (t_start is None or t_end is None) else int((t_end - t_start) * 1000)


def _ts():
    return time.time()


def _write_wav(path, audio_float32, sample_rate=16000):
    arr = np.clip(audio_float32, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


class SessionLogger:
    """One instance per sid. Thread-safe enough for our eventlet-greened threads."""

    def __init__(self, sid):
        self.sid = sid
        self.dir = os.path.join(LOGS_DIR, sid)
        os.makedirs(self.dir, exist_ok=True)
        self.turn_idx = 0
        self.lock = Lock()
        self._current = None
        self._tts_chunks = []  # concatenated assistant audio for this turn
        self._tts_sample_rate = 16000

    # ---- turn lifecycle ----------------------------------------------------
    def start_turn(self, user_audio_float32, user_sample_rate=16000):
        with self.lock:
            self.turn_idx += 1
            self._current = {
                "turn": self.turn_idx,
                "started_at": _ts(),
                "user_text": None,
                "assistant_text": None,
                "stages": {
                    "user_speech_end": _ts(),
                    "asr_start": None, "asr_end": None,
                    "llm_start": None, "llm_first_token": None, "llm_end": None,
                    "tts_first_audio": None, "tts_end": None,
                },
            }
            self._tts_chunks = []
            self._tts_sample_rate = user_sample_rate
            self._tts_synth_total_s = 0.0   # cumulative time INSIDE tts.generate_audio_chunk
            self._tts_audio_total_s = 0.0   # cumulative duration of generated audio
            # Save user audio
            try:
                user_path = os.path.join(self.dir, f"user_{self.turn_idx:03d}.wav")
                _write_wav(user_path, user_audio_float32, sample_rate=user_sample_rate)
                self._current["user_audio"] = os.path.relpath(user_path, LOGS_DIR)
            except Exception as e:
                print(f"[LOG] user wav write failed: {e}")

    def mark(self, stage):
        """Mark a stage timestamp on the current turn."""
        if self._current is None:
            return
        self._current["stages"][stage] = _ts()

    def set_user_text(self, text):
        if self._current is not None:
            self._current["user_text"] = text

    def set_assistant_text(self, text):
        if self._current is not None:
            self._current["assistant_text"] = text

    def add_tts_chunk_wav(self, wav_bytes, synth_ms=None):
        """Decode a WAV chunk (as emitted to client) and append its PCM samples.

        If `synth_ms` is provided, accumulate it as the actual model-inference
        time for this chunk (excludes wall-time waiting for upstream LLM tokens).
        """
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sr = wf.getframerate()
                n = wf.getnframes()
                pcm = wf.readframes(n)
                arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            self._tts_chunks.append((sr, arr))
            self._tts_audio_total_s += n / float(sr)
            if synth_ms is not None:
                self._tts_synth_total_s += float(synth_ms) / 1000.0
        except Exception as e:
            print(f"[LOG] tts chunk decode failed: {e}")

    def finish_turn(self):
        """Flush assistant wav + write turn record to turns.jsonl."""
        if self._current is None:
            return None
        with self.lock:
            rec = self._current
            self._current = None

            # Concatenate TTS chunks → assistant_NNN.wav (use SR of first chunk).
            if self._tts_chunks:
                sr0 = self._tts_chunks[0][0]
                # Resample-mismatched chunks would be unusual; just keep same SR.
                parts = [a for (s, a) in self._tts_chunks if s == sr0]
                if parts:
                    full = np.concatenate(parts)
                    try:
                        a_path = os.path.join(self.dir, f"assistant_{rec['turn']:03d}.wav")
                        _write_wav(a_path, full, sample_rate=sr0)
                        rec["assistant_audio"] = os.path.relpath(a_path, LOGS_DIR)
                    except Exception as e:
                        print(f"[LOG] assistant wav write failed: {e}")
            self._tts_chunks = []

            # Compute latencies
            s = rec["stages"]
            rec["latency_ms"] = {
                "asr":        _ms(s["asr_start"], s["asr_end"]),
                "llm_ttft":   _ms(s["llm_start"], s["llm_first_token"]),
                "llm_total":  _ms(s["llm_start"], s["llm_end"]),
                "tts_ttft":   _ms(s["llm_start"], s["tts_first_audio"]),
                # TTS synthesis time only (sum of model-inference per chunk).
                # Excludes wall-time waiting for upstream LLM tokens between phrases.
                "tts_synth":  int(self._tts_synth_total_s * 1000) if self._tts_synth_total_s else None,
                # Total audio duration produced (informational, not a latency).
                "tts_audio":  int(self._tts_audio_total_s * 1000) if self._tts_audio_total_s else None,
                # Wall time from first audio chunk to last (kept for back-compat).
                "tts_wall":   _ms(s["tts_first_audio"], s["tts_end"]),
                "total":      _ms(s["user_speech_end"], s["tts_end"]),
            }
            rec["finished_at"] = _ts()

            try:
                with open(os.path.join(self.dir, "turns.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                print(f"[LOG] turns.jsonl write failed: {e}")
            return rec


# Global registry — one logger per sid.
_loggers = {}
_loggers_lock = Lock()

# Cache last 10 turn summaries per sid for the /turns endpoint.
_recent_turns = defaultdict(lambda: deque(maxlen=10))


def get_logger(sid):
    with _loggers_lock:
        if sid not in _loggers:
            _loggers[sid] = SessionLogger(sid)
        return _loggers[sid]


def drop_logger(sid):
    with _loggers_lock:
        _loggers.pop(sid, None)
    _recent_turns.pop(sid, None)


def record_turn_summary(sid, rec):
    if rec is not None:
        _recent_turns[sid].append({
            "turn": rec["turn"],
            "user_text": rec.get("user_text"),
            "assistant_text": rec.get("assistant_text"),
            "latency_ms": rec["latency_ms"],
        })


def recent_turns(sid=None, limit=10):
    if sid is not None:
        return list(_recent_turns.get(sid, deque()))[-limit:]
    out = []
    for s, q in _recent_turns.items():
        out.extend(q)
    return out[-limit:]
