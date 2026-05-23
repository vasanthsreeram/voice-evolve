#!/usr/bin/env python3
"""
Kokoro TTS Helper Function

A simple helper function for text-to-speech using the Kokoro-82M model
optimized for Apple Silicon with MLX.

Based on benchmark results, uses the bf16 model for best performance.
"""

from mlx_audio.tts.generate import generate_audio
import os


def text_to_speech(
    text: str,
    output_path: str = "output.wav",
    voice: str = "af_heart",
    speed: float = 1.0,
    lang_code: str = "a",
    model: str = "mlx-community/Kokoro-82M-bf16",
    sample_rate: int = 24000,
    verbose: bool = False
) -> str:
    """
    Convert text to speech using Kokoro TTS model.

    Args:
        text: Text to convert to speech
        output_path: Path where the audio file will be saved (default: "output.wav")
        voice: Voice preset to use. Options:
               - "af_heart" (American female, warm)
               - "af_nova" (American female, clear)
               - "af_bella" (American female, expressive)
               - "bf_emma" (British female)
        speed: Speech speed multiplier (0.5 - 2.0, default: 1.0)
        lang_code: Language code:
                   - "a" (American English)
                   - "b" (British English)
                   - "j" (Japanese)
                   - "z" (Mandarin)
        model: Model to use (default: bf16 - fastest on Apple Silicon)
        sample_rate: Audio sample rate in Hz (default: 24000)
        verbose: Print generation details (default: False)

    Returns:
        str: Path to the generated audio file

    Example:
        >>> text_to_speech("Hello world!", "hello.wav")
        'hello.wav'

        >>> text_to_speech(
        ...     "This is a test.",
        ...     output_path="test.wav",
        ...     voice="af_nova",
        ...     speed=1.2
        ... )
        'test.wav'
    """

    # Extract directory and filename
    output_dir = os.path.dirname(output_path) or "."
    filename = os.path.basename(output_path)
    file_prefix = os.path.splitext(filename)[0]
    audio_format = os.path.splitext(filename)[1].lstrip(".") or "wav"

    # Generate audio
    generate_audio(
        text=text,
        model_path=model,
        voice=voice,
        speed=speed,
        lang_code=lang_code,
        file_prefix=f"{output_dir}/{file_prefix}",
        audio_format=audio_format,
        sample_rate=sample_rate,
        join_audio=True,
        verbose=verbose
    )

    return output_path


def main():
    """Example usage of the text_to_speech helper function."""

    # Example 1: Basic usage
    print("Example 1: Basic usage")
    text_to_speech(
        "Hello! This is a test of the Kokoro text to speech system.",
        output_path="example_basic.wav"
    )
    print("✓ Generated: example_basic.wav\n")

    # Example 2: Different voice and speed
    print("Example 2: Different voice and faster speed")
    text_to_speech(
        "This is spoken with a different voice at a faster speed.",
        output_path="example_fast.wav",
        voice="af_nova",
        speed=1.3
    )
    print("✓ Generated: example_fast.wav\n")

    # Example 3: Slower, more expressive
    print("Example 3: Slower, expressive speech")
    text_to_speech(
        "This is spoken more slowly with an expressive voice.",
        output_path="example_slow.wav",
        voice="af_bella",
        speed=0.85
    )
    print("✓ Generated: example_slow.wav\n")

    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
