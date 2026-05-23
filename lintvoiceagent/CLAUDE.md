# CLAUDE.md

Project context for future Claude sessions working on this codebase.

## What this is

Real-time browser voice agent running entirely on Apple Silicon. Single Flask + eventlet + Socket.IO server (`app.py`, ~800 lines) that chains:

```
mic (browser, 16kHz Opus → PCM) ──WS──▶ VAD ─▶ ASR ─▶ LLM ─▶ TTS ──WS──▶ speaker
```

Everything is in-process MLX (unified memory). LM Studio is not required at runtime — the LLM is loaded directly via `mlx_lm.stream_generate`.

## Current default stack (as of this codebase)

| Stage | Library | Model | Notes |
|---|---|---|---|
| VAD | `fireredvad` (vendored, no PyPI) | `pretrained_models/FireRedVAD/VAD/model.pth.tar` | DFSMN, non-streaming. **Adapter in `vad_detector.py:FireRedVAD`** |
| ASR | `mlx_audio.stt` | `mlx-community/Qwen3-ASR-0.6B-4bit` | Batch on VAD-stop; writes temp wav, ASR reads it, file is unlinked |
| LLM | `mlx_lm` | `~/.lmstudio/models/mlx-community/Qwen3.5-0.8B-MLX-4bit` (local path) | Override with `LLM_MODEL_NAME` env. We deliberately use the LM Studio path as a model cache; LM Studio itself is not running |
| TTS | `mlx_audio.tts` Kokoro pipeline | `mlx-community/Kokoro-82M-bf16` | Default. See `kokoro_streaming.py:KokoroStreamingTTS` |
| Alt TTS | `mlx_audio.tts` Qwen3-TTS | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | Voice cloning (Rick voice in `voices/rick_ref.wav`). See `streaming_tts.py:StreamingTTS` |
| Alt TTS | `supertonic` (PyPI) | Supertonic-3 ONNX | CPU/ONNX. See `supertonic_tts.py:SupertonicStreamingTTS` |
| Alt VAD | torch.hub `snakers4/silero-vad` | Silero VAD v5 | Lazy, switchable via socket `set_vad_mode` |

The `TTS_ENGINES` dict in `app.py` is the registry. `DEFAULT_ENGINE = "kokoro"`, `DEFAULT_VAD_MODE = "firered"`.

## Run it

```bash
uv sync                                  # one-time
HF_TOKEN=... uv run python app.py        # token unblocks HF rate-limit for first-time downloads
# open http://localhost:3003
# history at http://localhost:3003/history
# stats at http://localhost:3003/stats
# raw turn log at http://localhost:3003/turns
```

Override LLM without code edit:
```bash
LLM_MODEL_NAME=~/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit uv run python app.py
```

## Repo layout

```
app.py                  # the whole server — Flask + SocketIO + pipeline orchestration
streaming_tts.py        # Qwen3-TTS engine + TextChunker (sentence boundary splitter)
kokoro_streaming.py     # Kokoro engine (current default)
supertonic_tts.py       # Supertonic-3 engine
kokoro_tts.py           # OLD standalone helper, not used at runtime
vad_detector.py         # SileroVAD + SmartTurnV3 + FireRedVAD adapters + factory
vad_config.py           # per-mode thresholds, presets
session_logger.py       # per-turn audio/text/latency logging to logs/<sid>/
templates/
  index.html            # main UI — mic, stats bar, turn-level latency pills
  history.html          # session browser, replay audio, raw stage timestamps
voices/rick_ref.wav     # reference audio for Qwen3-TTS Rick voice clone
vendor/FireRedVAD/      # cloned FireRedTeam repo (imported via sys.path, not pip-installed)
vendor/pipecat/         # cloned reference — for architecture comparison only, NOT imported
pretrained_models/FireRedVAD/  # HF-downloaded weights, gitignored
logs/<sid>/             # per-session audio + turns.jsonl, gitignored
.venv/, uv.lock         # uv-managed env (replaces requirements.txt which was removed)
```

## Gotchas you will hit

1. **Eventlet monkey-patches socket → HF parallel downloads hang.** First-time model downloads inside the running app freeze. Workaround: pre-download outside the eventlet process: `HF_TOKEN=... uv run hf download <repo_id>`. Only the first run is affected; once cached, mmap is instant. See history with `Qwen3-TTS` and `Kokoro` downloads earlier.

2. **FireRedVAD wants int16-scale numpy, not float32 [-1,1].** The library's `AudioFeat` reads files as `dtype="int16"`. If you hand it float32 in [-1, 1] it silently treats audio as ~32000× too quiet → `timestamps: []` → `has_speech` always False. `vad_detector.py:FireRedVAD.has_speech` scales to int16 before calling `vad.detect()`. Do not "fix" this.

3. **Use non-streaming `FireRedVad`, not `FireRedStreamVad`, for our pipeline.** Streaming variant keeps DFSMN `model_caches` across calls. Our `audio_chunk` handler passes overlapping rolling buffers — that corrupts the streaming state. Non-streaming resets per call (correct for chunk-based detection).

4. **Silero VAD via torch.hub needs `trust_repo=True`.** Without it the first load blocks on an interactive `y/N` prompt, the audio_chunk handler EOFs, and the server appears hung.

5. **mlx_lm 4B-8bit is way slower than expected (~9 tok/s).** Suspect eventlet ⇄ MLX scheduling interference + per-turn re-encoding of the long system prompt (no KV cache reuse). 0.8B-4bit hits the spec ~80 tok/s. If you want to try KV cache reuse, see `mlx_lm.stream_generate(prompt_cache=...)`.

6. **TTS "tts_total" in `logs/<sid>/turns.jsonl` is wall time, not synthesis time.** Use `latency_ms.tts_synth` instead — that's actual cumulative model-inference time, measured per yielded chunk in `app.py`'s LLM→TTS loop. `tts_wall` is kept for back-compat.

7. **Camera/VLM is gone.** We removed `mlx_vlm` entirely when switching to mlx_lm. The browser still has the camera-frame socket event but the LLM path no longer consumes images. Bringing it back requires loading a Qwen-VL via mlx_vlm and putting back the old branch in `get_llm_response_streaming`.

## Barge-in flow

1. During TTS playback, browser emits `playback_started` → server calls `FireRedVAD.set_barge_in_mode(True)` which drops `min_speech_frame` from 15→4 (~40ms detection).
2. Each incoming `audio_chunk` runs VAD. If `playback_active[sid] and chunk_has_speech and generation_active[sid]` → bump `current_generation_id[sid]`, emit `assistant_cancel`.
3. The LLM stream loop and TTS chunk loops check `current_generation_id[sid] != gen_id` between iterations → break out.
4. Browser receives `assistant_cancel` → calls `stopAudio()` which stops all queued `AudioBufferSourceNode`s and clears `audioQueue`.

Browser `getUserMedia` already has `echoCancellation: true` so we don't raise VAD thresholds during playback (we used to — removed it).

## Per-turn logging

`session_logger.py:SessionLogger` writes `logs/<sid>/user_NNN.wav`, `assistant_NNN.wav`, and appends a turn record to `turns.jsonl`. Stage timestamps captured:

```
user_speech_end → asr_start → asr_end → llm_start → llm_first_token → llm_end → tts_first_audio → tts_end
```

Computed latencies in `latency_ms`: `asr`, `llm_ttft`, `llm_total`, `tts_ttft`, `tts_synth` (real inference), `tts_wall` (legacy), `total`.

The UI subscribes to `turn_complete` socket event and updates the blue latency pills in the top bar. `/history` lets you scrub past sessions and replay both sides.

## What we explicitly chose NOT to do

- **No pipecat dependency.** `vendor/pipecat/` exists for reference only — to crib design ideas (system-vs-data queue split, NLTK sentence aggregator, frame-based interruption). Not imported.
- **No cloud APIs.** No Deepgram, no Cartesia, no OpenAI. Everything must work offline with Wi-Fi off after first-time model downloads.
- **No requirements.txt.** Replaced with `pyproject.toml` + `uv.lock`. Add deps by editing `pyproject.toml` then `uv sync`.
- **No persistence beyond `logs/`.** Conversation history is in-memory per `request.sid` and lost on disconnect. Intentional.

## When making changes

- Restart the server to pick up any Python change (no autoreload — eventlet doesn't play well with Flask's reloader).
- HTML/JS edits don't need a restart; just refresh the browser.
- Don't add `cd vendor/...` blindly into bash one-liners — `uv run` will pick up the vendored `pyproject.toml` (e.g. FireRedVAD's) and create a parallel venv. Always run from project root.
- If you touch the audio_chunk handler, the rule is: keep the fast path fast. Don't add I/O or model calls outside the `with buffer_locks[sid]:` block scope unless they're actually needed every chunk.
