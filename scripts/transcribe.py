"""Transcribe a WAV file using the same Qwen3-ASR-0.6B-4bit model lintvoiceagent uses.

Usage:
    uv run --project lintvoiceagent python scripts/transcribe.py path/to/file.wav

Prints the transcript to stdout and writes <file>.txt next to the input.
"""
import sys
from pathlib import Path

from mlx_audio.stt.utils import load_model as load_asr_model
from mlx_audio.stt.generate import generate_transcription


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: transcribe.py <audio.wav>")
    audio = Path(sys.argv[1])
    if not audio.exists():
        raise SystemExit(f"not found: {audio}")
    model = load_asr_model("mlx-community/Qwen3-ASR-0.6B-4bit")
    result = generate_transcription(model=model, audio=str(audio), format="txt", verbose=False)
    text = (result.text or "").strip()
    out = audio.with_suffix(".txt")
    out.write_text(text + "\n")
    print(text)
    print(f"\n[saved -> {out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
