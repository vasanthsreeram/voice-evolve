#!/usr/bin/env python3
"""Kokoro-82M streaming TTS engine — mirrors the StreamingTTS interface.

Kokoro is tiny (82M params), runs fast on Apple Silicon via mlx_audio, and
streams sub-phrase audio chunks. Good fit when you want lower TTFA than
Qwen3-TTS at the cost of voice cloning (Kokoro uses fixed preset voices).
"""

import io
import wave
import re
import traceback
import numpy as np
import librosa

from mlx_audio.tts.utils import get_model_path, load_model

TTS_MODEL_ID = "mlx-community/Kokoro-82M-bf16"

# Curated preset voices — Kokoro ships dozens; these are the main English ones.
AVAILABLE_VOICES = {
    "Heart (F, warm)":     "af_heart",
    "Nova (F, clear)":     "af_nova",
    "Bella (F, expressive)": "af_bella",
    "Sarah (F)":           "af_sarah",
    "Emma (F, British)":   "bf_emma",
    "Adam (M)":            "am_adam",
    "Michael (M)":         "am_michael",
}

DEFAULT_VOICE = "af_heart"


class KokoroStreamingTTS:
    """Streaming Kokoro TTS via mlx_audio."""

    def __init__(self, voice=DEFAULT_VOICE, speed=1.0, lang_code="a"):
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code  # 'a' = American English, 'b' = British, 'j' = JP, 'z' = ZH

        print(f"Preloading Kokoro TTS ({TTS_MODEL_ID})...")
        model_path = get_model_path(TTS_MODEL_ID)
        self.model = load_model(model_path)
        print("Kokoro model loaded!")

    def generate_audio_chunk(self, text, voice=None, speed=None):
        if not text or not text.strip():
            return

        active_voice = voice or self.voice
        active_speed = speed if speed is not None else self.speed

        try:
            for result in self.model.generate(
                text=text,
                voice=active_voice,
                speed=active_speed,
                lang_code=self.lang_code,
                stream=True,
                streaming_interval=float(__import__('os').environ.get('KOKORO_STREAM_INTERVAL', '0.3')),
                verbose=False,
            ):
                audio_np = np.array(result.audio, dtype=np.float32)
                # Kokoro outputs 24kHz; librosa time-stretch keeps pitch.
                if active_speed != 1.0:
                    audio_np = librosa.effects.time_stretch(audio_np, rate=active_speed)
                audio_np = np.clip(audio_np, -1.0, 1.0)
                pcm16 = (audio_np * 32767).astype(np.int16)
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.model.sample_rate)
                    wf.writeframes(pcm16.tobytes())
                yield buf.getvalue()
        except Exception as e:
            print(f"[Kokoro TTS ERROR] {e}")
            traceback.print_exc()
