#!/usr/bin/env python3
"""
Streaming TTS Module for Voice Agent - Qwen3-TTS Edition

Provides real-time text-to-speech streaming using Qwen3-TTS via mlx_audio.
Converts text chunks into audio as they arrive from the LLM.
"""

import os
import numpy as np
import base64
import io
import wave
import re
import traceback

from mlx_audio.tts.utils import get_model_path, load_model
from mlx_audio.utils import load_audio
import librosa

TTS_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
TTS_SAMPLE_RATE = 24000
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")

# Available voices: label -> voice id (None = use ref_audio cloning)
AVAILABLE_VOICES = {
    "Serena (Female)": "serena",
    "Vivian (Female)": "vivian",
    "Sohee (Female)":  "sohee",
    "Aiden (Male)":    "aiden",
    "Ryan (Male)":     "ryan",
    "Dylan (Male)":    "dylan",
    "Eric (Male)":     "eric",
    "Rick Sanchez": "rick_sanchez",
}

# Ref audio for cloned voices (voice id -> {path, text})
REF_VOICES = {
    "rick_sanchez": {
        "ref_audio": os.path.join(VOICES_DIR, "rick_ref.wav"),
        "ref_text": "Listen, Morty, I hate to break it to you, but what people call love is just a chemical reaction that compels animals to breed. It hits hard, Morty, then it slowly fades, leaving you stranded in a failing marriage. I did it. Your parents are gonna do it. Break the cycle.",
    }
}

DEFAULT_VOICE = "rick_sanchez"


class StreamingTTS:
    """Handles streaming text-to-speech conversion using Qwen3-TTS via mlx_audio."""

    def __init__(self, voice=DEFAULT_VOICE, speed=1.0, use_gpu=False):
        """
        Initialize the streaming TTS engine.

        Args:
            voice: Voice preset (e.g., 'af_heart', 'af_bella')
            speed: Speech speed multiplier (0.5 - 2.0)
            use_gpu: Unused (mlx_audio handles device placement via MLX)
        """
        self.voice = voice
        self.speed = speed
        self.text_buffer = ""

        # Preload model so first inference isn't slow
        print(f"Preloading Qwen3-TTS model ({TTS_MODEL_ID})...")
        model_path = get_model_path(TTS_MODEL_ID)
        self.model = load_model(model_path)
        print("Qwen3-TTS model loaded!")

    def generate_audio_chunk(self, text, voice=None, speed=None):
        """
        Generate audio for a text chunk, streaming chunks as they arrive.

        Args:
            text: Text to convert to speech
            voice: Optional voice override (uses self.voice if not set)

        Yields:
            bytes: WAV audio data (streamed — first chunk arrives before full generation)
        """
        if not text or not text.strip():
            return

        active_voice = voice or self.voice
        ref_meta = REF_VOICES.get(active_voice)

        try:
            gen_kwargs = dict(
                text=text,
                speed=speed if speed is not None else self.speed,
                lang_code='auto',
                stream=True,
                streaming_interval=1.5,
                temperature=0.8,
                split_pattern=None,
                verbose=False,
            )

            if ref_meta:
                # Voice cloning mode — use ref audio instead of named speaker
                ref_audio_arr = load_audio(ref_meta["ref_audio"], sample_rate=self.model.sample_rate)
                gen_kwargs["ref_audio"] = ref_audio_arr
                gen_kwargs["ref_text"] = ref_meta["ref_text"]
                gen_kwargs["voice"] = None
            else:
                gen_kwargs["voice"] = active_voice

            for result in self.model.generate(**gen_kwargs):
                audio_np = np.array(result.audio, dtype=np.float32)
                # Apply speed via time-stretch (pitch-preserving)
                effective_speed = speed if speed is not None else self.speed
                if effective_speed != 1.0:
                    audio_np = librosa.effects.time_stretch(audio_np, rate=effective_speed)
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
            print(f"[TTS ERROR] {e}")
            traceback.print_exc()

    def add_text(self, token):
        """
        Add a token to the buffer and return complete sentences.

        Args:
            token: Text token from LLM

        Returns:
            str or None: Complete sentence if available, None otherwise
        """
        self.text_buffer += token

        sentence_pattern = r'([.!?;]+[\s\n]+)'
        match = re.search(sentence_pattern, self.text_buffer)

        if match:
            end_pos = match.end()
            sentence = self.text_buffer[:end_pos].strip()
            self.text_buffer = self.text_buffer[end_pos:]
            return sentence

        return None

    def flush(self):
        """
        Get any remaining text in the buffer.

        Returns:
            str: Remaining buffered text
        """
        remaining = self.text_buffer.strip()
        self.text_buffer = ""
        return remaining


class TextChunker:
    """Helper class to chunk text into speakable phrases."""

    # Sentence-ending punctuation
    SENTENCE_END = r'[.!?;]+[\s\n]+'

    # Phrase-breaking punctuation (for faster streaming)
    PHRASE_BREAK = r'[,:][\s]+'

    # Minimum characters before considering a break
    MIN_CHUNK_SIZE = 20

    def __init__(self, mode='sentence'):
        """
        Initialize chunker.

        Args:
            mode: 'sentence' for full sentences, 'phrase' for faster streaming
        """
        self.mode = mode
        self.buffer = ""

    def add_token(self, token):
        """
        Add a token and return complete chunks.

        Args:
            token: Text token from LLM

        Yields:
            str: Complete chunks ready for TTS
        """
        self.buffer += token

        if self.mode == 'phrase':
            pattern = f'({self.SENTENCE_END}|{self.PHRASE_BREAK})'
        else:
            pattern = f'({self.SENTENCE_END})'

        while True:
            if len(self.buffer) < self.MIN_CHUNK_SIZE:
                break

            match = re.search(pattern, self.buffer)
            if not match:
                break

            end_pos = match.end()
            chunk = self.buffer[:end_pos].strip()
            self.buffer = self.buffer[end_pos:]

            if chunk:
                yield chunk

    def flush(self):
        """
        Get any remaining text.

        Returns:
            str or None: Remaining text if any
        """
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None


def audio_to_base64(wav_bytes):
    """
    Convert WAV bytes to base64 for transmission.

    Args:
        wav_bytes: WAV file as bytes

    Returns:
        str: Base64-encoded audio
    """
    return base64.b64encode(wav_bytes).decode('utf-8')
