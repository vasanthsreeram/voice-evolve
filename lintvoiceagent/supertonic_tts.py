#!/usr/bin/env python3
"""
Supertonic-3 TTS engine wrapper.

Mirrors the StreamingTTS interface (generate_audio_chunk) so app.py can swap
engines per session. Supertonic-3 is an ONNX-based on-device TTS model
(~99M params, 44.1kHz output) supporting 31 languages and preset voice styles.
"""

import io
import wave
import numpy as np
import librosa
import traceback

from supertonic import TTS

SUPERTONIC_SAMPLE_RATE = 44100

# Preset voice styles shipped with Supertonic-3.
AVAILABLE_VOICES = {
    "Supertonic M1 (Male)":   "M1",
    "Supertonic M2 (Male)":   "M2",
    "Supertonic F1 (Female)": "F1",
    "Supertonic F2 (Female)": "F2",
}

DEFAULT_VOICE = "M1"


class SupertonicStreamingTTS:
    """Supertonic-3 TTS engine — same interface as StreamingTTS."""

    def __init__(self, voice=DEFAULT_VOICE, speed=1.0, lang="en"):
        self.voice = voice
        self.speed = speed
        self.lang = lang

        print("Preloading Supertonic-3 TTS model...")
        self.model = TTS(auto_download=True)
        self._style_cache = {}
        print("Supertonic-3 model loaded!")

    def _get_style(self, voice_name):
        if voice_name not in self._style_cache:
            try:
                self._style_cache[voice_name] = self.model.get_voice_style(voice_name=voice_name)
            except FileNotFoundError:
                # Voice name from another engine — fall back to default.
                print(f"[Supertonic] Voice '{voice_name}' unknown, falling back to '{DEFAULT_VOICE}'")
                if DEFAULT_VOICE not in self._style_cache:
                    self._style_cache[DEFAULT_VOICE] = self.model.get_voice_style(voice_name=DEFAULT_VOICE)
                self._style_cache[voice_name] = self._style_cache[DEFAULT_VOICE]
        return self._style_cache[voice_name]

    def generate_audio_chunk(self, text, voice=None, speed=None):
        if not text or not text.strip():
            return

        active_voice = voice or self.voice
        active_speed = speed if speed is not None else self.speed

        try:
            style = self._get_style(active_voice)
            wav, _duration = self.model.synthesize(text, voice_style=style, lang=self.lang)

            audio_np = np.asarray(wav, dtype=np.float32).squeeze()

            if active_speed != 1.0:
                audio_np = librosa.effects.time_stretch(audio_np, rate=active_speed)

            audio_np = np.clip(audio_np, -1.0, 1.0)
            pcm16 = (audio_np * 32767).astype(np.int16)

            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SUPERTONIC_SAMPLE_RATE)
                wf.writeframes(pcm16.tobytes())
            yield buf.getvalue()

        except Exception as e:
            print(f"[Supertonic TTS ERROR] {e}")
            traceback.print_exc()
