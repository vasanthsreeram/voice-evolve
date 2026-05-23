# 🎙️ LintVoiceAgent

**Full-duplex voice conversation agent** — speak naturally, get real-time responses with streaming TTS. Runs entirely on Apple Silicon with local models.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey)

## What is this?

LintVoiceAgent is a real-time voice assistant that chains **ASR → LLM → TTS** in a single streaming pipeline. Talk to it like a person — it listens, thinks, and speaks back with minimal latency. Supports barge-in (interrupt the AI mid-sentence), camera input for visual context, and multiple TTS voices.

## Demo

> *You speak → Qwen3-ASR transcribes → Qwen3.5 generates → Kokoro TTS speaks back — all locally, all streaming.*

## Pipeline

```
🎤 Microphone
    │
    ▼
┌─────────────┐
│  Silero VAD  │  ← Voice Activity Detection (speech vs silence)
└──────┬──────┘
       │ speech detected
       ▼
┌─────────────┐
│  Qwen3-ASR  │  ← Speech-to-Text (mlx_audio, 0.6B 4-bit)
└──────┬──────┘
       │ transcribed text
       ▼
┌─────────────┐
│  Qwen3.5    │  ← LLM Response (mlx_vlm, 4B 4-bit) + optional camera frame
└──────┬──────┘
       │ streaming tokens
       ▼
┌─────────────┐
│  Kokoro TTS │  ← Text-to-Speech (streaming, 24kHz)
└──────┬──────┘
       │ audio chunks
       ▼
🔊 Speaker
```

## Features

- **Full Duplex** — Listen and speak simultaneously with barge-in support
- **Streaming Pipeline** — Each stage streams to the next; no waiting for full completion
- **Voice Activity Detection** — Silero VAD detects speech start/stop automatically
- **Partial Transcription** — See what's being heard in real-time while you speak
- **Camera Input** — Optional webcam feed gives the LLM visual context (VLM mode)
- **Multiple Voices** — Switch TTS voices on the fly
- **Adjustable Speed** — 0.5x to 2.0x TTS playback speed
- **Conversation Memory** — Multi-turn conversation history maintained per session
- **100% Local** — No API keys, no cloud, no data leaving your machine

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+

### Install

```bash
git clone https://github.com/lintware/lintvoiceagent.git
cd lintvoiceagent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open `http://localhost:3003` in your browser. Click the microphone button and start talking.

### Custom Port

```bash
PORT=8080 python app.py
```

## How It Works

1. **Browser** captures microphone audio via WebSocket, sends 16kHz PCM chunks to the server
2. **Silero VAD** detects speech boundaries — when you start and stop talking
3. **Qwen3-ASR** transcribes the audio chunk (partial transcriptions while speaking, final on silence)
4. **Qwen3.5 VLM** generates a response from the conversation history (+ camera frame if enabled)
5. **Kokoro TTS** converts the response to speech audio, streamed chunk-by-chunk back to the browser
6. **Barge-in** — if you start speaking while the AI is responding, it cancels the current generation immediately

## Models Used

| Component | Model | Size | Backend |
|-----------|-------|------|---------|
| ASR | `Qwen3-ASR-0.6B-4bit` | 0.6B | mlx_audio |
| LLM | `Qwen3.5-4B-MLX-4bit` | 4B | mlx_vlm |
| TTS | Kokoro (via streaming_tts.py) | — | Local |
| VAD | Silero VAD v5 | — | PyTorch |

All models run quantized (4-bit) on Apple Silicon's unified memory. Total memory footprint is ~4-6 GB.

## Project Structure

```
lintvoiceagent/
├── app.py               # Main server (Flask + SocketIO)
├── streaming_tts.py     # Kokoro TTS streaming engine
├── kokoro_tts.py        # Kokoro model wrapper
├── vad_detector.py      # Silero VAD implementation
├── vad_config.py        # VAD thresholds and timing config
├── templates/
│   └── index.html       # Web UI (mic, camera, chat display)
├── voices/
│   └── rick_ref.wav     # Reference voice for cloning
└── requirements.txt
```

## Configuration

### VAD Tuning

Edit `vad_config.py` to adjust speech detection sensitivity:

```python
STREAMING_INTERVAL_SECONDS = 2.0   # How often to run partial transcription
MIN_BUFFER_SECONDS = 0.8           # Minimum audio before processing
OVERLAP_SECONDS = 0.3              # Audio overlap between chunks
```

### Custom System Prompt

The default personality is Rick Sanchez (for fun). Change the `system_prompt` in `app.py` → `get_llm_response_streaming()` to whatever you want:

```python
system_prompt = "You are a helpful voice assistant. Keep responses concise and conversational."
```

### Swapping Models

Change `LLM_MODEL_NAME` in `app.py` to any MLX-compatible model:

```python
LLM_MODEL_NAME = "mlx-community/Qwen3.5-8B-MLX-4bit"  # Bigger, smarter
LLM_MODEL_NAME = "mlx-community/Qwen3.5-1.5B-MLX-4bit"  # Smaller, faster
```

## License

MIT

---

Built with 🔥 by [Lint Labs](https://github.com/lintware)
