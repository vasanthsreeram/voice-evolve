# voice-evolve

> Two local AI agents call each other over real Telegram, hold a conversation, and a Gemini judge grades the call. An optimization loop tunes the pipeline until the call sounds human. Everything runs on a MacBook Air.

![status](https://img.shields.io/badge/status-experiment-orange)
![runs-on](https://img.shields.io/badge/runs%20on-Apple%20Silicon-black)
![model](https://img.shields.io/badge/LLM-Qwen3.5--4B--MLX--8bit-blue)
![tts](https://img.shields.io/badge/TTS-Kokoro--82M-purple)
![asr](https://img.shields.io/badge/ASR-Qwen3--0.6B--ASR-green)

## What this is

`voice-evolve` is an end-to-end **voice-to-voice phone-call harness** for
self-improvement experiments. Two Telegram accounts each run a local agent
that listens, transcribes, thinks, speaks, and hangs up. Audio rides the
real Telegram MTProto network. A Gemini-based judge grades each call and a
driver tunes the pipeline's knobs over many iterations until the call
passes the judge.

```
katey  ──MTProto──►  chippy
  ▲                    │
  │                    ▼
  │   FireRedVAD  ──►  Qwen3-ASR  ──►  Qwen3.5 LLM  ──►  Kokoro TTS
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

The whole audio path is local. No cloud ASR, no cloud TTS, no cloud LLM.
The Gemini judge is optional and only used to score finished calls.

## Highlights from one session

The session that produced this repo:

- 9 optimization loops over the course of an evening
- Best Gemini composite score: **5.75 / 10** (pacing 6, overlap 10, coherence 8, naturalness 7, latency 3.0 s)
- Median per-turn TTFA: **~600 ms** end-to-end through MTProto
- A real call where the user picked up on their phone and the AI ran a 12-turn Android-vs-iPhone debate, complete with working mid-sentence barge-in.
- ~$0.05 total Gemini judge spend.

Read the [story](docs/blog.html) (built into an HTML page you can open
locally). Real transcripts under `recordings/duet_*_convo.txt` once you
run a call.

## Quick start

### 1. Install

The repo is split into two `uv`-managed Python projects:

- `lintvoiceagent/` — the local ASR/LLM/TTS stack (MLX, Kokoro, FireRedVAD).
- `telegramcaller/` — the original Telethon / pytgcalls demo it wraps.

The orchestrators live in `scripts/` and run under the `lintvoiceagent`
environment (which has both the audio stack and the Telegram bindings).

```bash
git clone https://github.com/vasanthsreeram/voice-evolve.git
cd voice-evolve

# install deps for the audio stack + Telegram bindings
uv sync --project lintvoiceagent
```

### 2. Telegram credentials

Get a `TG_API_ID` and `TG_API_HASH` from <https://my.telegram.org>. You'll
need **two** Telegram accounts that have spoken to each other at least once
(any normal text message is enough). Log each one in once to create a
`.session` file in the repo root:

```bash
cp .env.example .env
# fill in TG_API_ID and TG_API_HASH

# log in once per account (interactive — sends a code to your phone)
TG_SESSION=telecall uv run --project lintvoiceagent python -c \
  "from telethon import TelegramClient; import os; \
   c = TelegramClient(os.environ['TG_SESSION'], int(os.environ['TG_API_ID']), os.environ['TG_API_HASH']); \
   c.start()"
TG_SESSION=bob uv run --project lintvoiceagent python -c \
  "from telethon import TelegramClient; import os; \
   c = TelegramClient(os.environ['TG_SESSION'], int(os.environ['TG_API_ID']), os.environ['TG_API_HASH']); \
   c.start()"
```

This produces `telecall.session` and `bob.session`. They're gitignored —
treat them like passwords.

### 3. (Optional) Gemini judge

Get a Gemini API key from <https://aistudio.google.com/>. Add to `.env`:

```
GEMINI_API_KEY=...
```

Only needed if you want to run the judge or the optimization loop.

### 4. Run a single duet call

Pre-warm by sending one normal text message between the two accounts (helps
avoid Telegram's "no prior interaction" stale-call error), then:

```bash
set -a && . ./.env && set +a
LLM_MODEL_NAME="$HOME/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-8bit" \
  uv run --project lintvoiceagent python scripts/run_duet.py \
    --hold 120 --max-turns 12
```

`scripts/run_duet.py` spawns chippy (the receiver) first, waits for her to
report `AGENT_READY`, then spawns katey who dials. Both ends run the same
ASR→LLM→TTS pipeline. Hangs up after `--hold` seconds or `--max-turns`
exchanges, whichever comes first.

Artifacts land under `recordings/`:
- `duet_<ts>_katey_stereo.wav` — stereo: L = local TTS, R = peer audio
- `duet_<ts>_chippy_stereo.wav`
- `duet_<ts>_{katey,chippy}_turns.jsonl` — per-turn ASR/LLM/TTS timings
- `duet_<ts>_convo.txt` — chronological merged transcript

### 5. Talk to one of them yourself

Spawn only katey and pick up on the other phone:

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
LLM_MODEL_NAME="$HOME/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-8bit" \
  uv run --project lintvoiceagent python scripts/run_agent.py \
    --role katey --run-id "$RUN_ID" \
    --phone +<chippy-phone> --dial <chippy-user-id> \
    --hold 300 --max-assistant-turns 999
```

Katey dials chippy's number. Answer on the actual Telegram app. After the
call she'll send the recording back to chippy as a compressed Opus voice
note via Telethon.

### 6. Have Gemini grade a call

```bash
uv run --project lintvoiceagent python scripts/judge_call.py \
  --run-id 20260523T004849Z
```

Writes `recordings/duet_<run_id>_judgment.json` with pacing / overlap /
coherence / naturalness / latency scores and prose feedback.

### 7. Run the optimization loop

```bash
uv run --project lintvoiceagent python scripts/loop_optimize.py \
  --hold 120 --max-iters 8 --start-from-defaults
```

For each iteration: run a duet → judge → if not passed, the loop driver
reads Gemini's scores and issues, decides which knob to nudge, applies it,
repeats. Stops early when Gemini returns `passed=true` or `--max-iters` is
hit. Per-iteration logs go to `recordings/loop_<id>/iter_NN.json`.

## What's in each directory

```
voice-evolve/
├── scripts/                    # Orchestrators
│   ├── run_agent.py            # One agent process (katey or chippy)
│   ├── run_duet.py             # Spawns both agent processes
│   ├── voice_agent.py          # VoiceAgent class — ASR/LLM/TTS state machine
│   ├── run_duet_local.py       # In-memory duet (no Telegram) for pipeline tests
│   ├── analyze_call.py         # VAD + Qwen-ASR over each channel of stereo wavs
│   ├── judge_call.py           # Gemini-as-judge for one call
│   └── loop_optimize.py        # Outer optimization loop driven by Gemini scores
│
├── lintvoiceagent/             # Local voice stack (ASR + LLM + TTS + VAD)
│   ├── kokoro_streaming.py     # Kokoro-82M TTS adapter
│   ├── streaming_tts.py        # TextChunker for streaming LLM→TTS
│   ├── vad_detector.py         # FireRedVAD + Silero adapters
│   └── ...
│
├── telegramcaller/             # Original Telethon + pytgcalls scaffolding
│
├── docs/
│   ├── SESSION_TO_SESSION.md   # Operational runbook
│   └── blog.html               # Engaging write-up of the build session
│
├── CLAUDE.md                   # Working agreements for Claude Code agents
├── CONTRIBUTING.md             # How to contribute
├── LICENSE                     # MIT
└── README.md                   # this file
```

## Tunable knobs

All read from environment variables — set them per-process or via the
optimization loop's config:

| env var | default | what it controls |
|---|---|---|
| `LLM_MODEL_NAME` | `mlx-community/Qwen3.5-2B-4bit` | path or HF id of LLM (Qwen3.5-0.8B is the fastest; 4B-8bit is the quality sweet spot) |
| `LLM_MAX_TOKENS` | 140 | cap on assistant response length |
| `VAD_SILENCE_THRESHOLD` | 2 | consecutive silent VAD windows before turn-end |
| `VAD_MIN_SPEECH_FRAME` | 4 | FireRedVAD minimum speech frames |
| `TTS_TAIL_GUARD_S` | 0.3 | post-TTS mute window (prevents self-echo) |
| `INITIAL_DEAFEN_S` | 1.0 | mute window at very start of call (init noise) |
| `MIN_UTTERANCE_BYTES` | 48000 | minimum bytes of speech before ASR (1.5 s @ 16 k) |
| `MIN_UTTERANCE_RMS` | 0.005 | RMS gate to drop noise-only utterances |
| `KOKORO_STREAM_INTERVAL` | 0.3 | Kokoro streaming chunk interval (s) |
| `LOOP_COOLDOWN_S` | 75 | inter-call cooldown so Telegram settles |

## Architecture notes

### Why two processes

Two `PyTgCalls` instances in one Python process crash with `SIGABRT` from
inside `libwebrtc` — they share global state. The orchestrator spawns one
process per role and pipes the stdouts back together.

### MLX is thread-local

The MLX GPU stream is bound to whichever thread loaded the model. Don't use
`asyncio.to_thread` for inference — use a `ThreadPoolExecutor(max_workers=1)`
that loads the models AND services every subsequent inference call.

### Streaming LLM→TTS

The biggest single latency win. Tokens stream out of `mlx_lm.stream_generate`
and accumulate into a `FastChunker` that yields the first phrase as soon as
it hits any punctuation (with `MIN_CHARS=3`). Each phrase is immediately
fed to Kokoro and the audio chunks land in the playback buffer while the
LLM is still generating the rest of the response.

### Echo suppression + barge-in

`feed_16k_pcm` watches for high-RMS peer audio even while `_is_speaking=True`.
If real speech is detected, `_bargein_requested=True` is set; the streaming
TTS loop checks it between phrases and bails out, and the playback buffer
is cleared. After the user's turn lands, the ASR text is checked against
katey's last assistant turn — if the overlap is >60%, it's an echo of her
own TTS bleeding back through MTProto and gets dropped.

### Stereo wav recording

Each side records a stereo wav with L = local TTS, R = peer inbound. Lets
the analyzer (and the Gemini judge if you point it at the right channel)
score each side independently without source separation.

## Resource cost

For one agent with the 4B-8bit LLM:

| | |
|---|---|
| RAM (per agent) | ~6.4 GB RSS |
| CPU idle | <0.1 % |
| CPU mid-turn | 30–100 % of one core |
| Disk (model weights) | ~6.6 GB total |
| Network per call | ~500 KB / minute |

A full local duet on one Mac is ~13 GB combined RAM (model weights mmap'd,
some OS page-cache sharing). 16 GB is tight, 32 GB is comfortable. Swap to
the 0.8B-8bit LLM to drop to ~2.5 GB per agent.

## Known limits

- **Telegram audio relay throttles after ~30 rapid calls from the same
  account pair.** You see calls that connect cleanly but only relay 4–8 s
  of audio. No software fix — wait a few hours or use different accounts.
  See `memory/telegram_audio_throttle.md` for the debug signature.
- **Qwen 4B's content filter** triggers on a few common political topics
  and refuses with `"I'm sorry, but I can't continue this conversation."`
  Either steer the persona away from those topics or swap models.
- **MP3-via-FIFO inbound path** adds ~150-300 ms of buffering. The raw
  PCM `StreamFrames` callback on `pytgcalls` 2.2.12 would skip it but the
  refactor isn't finished.

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Especially interested in:

- Smarter VAD with smart-turn semantic models (a stub exists for
  pipecat-ai's smart-turn-v3 in `vad_detector.py`).
- Raw PCM inbound via `StreamFrames` callback.
- Sliding-window chat history so LLM prefill stays flat as conversations
  grow.
- A working half-duplex floor protocol that survives in two separate
  processes (the fcntl-based one in `voice_agent.py` had per-fd lock
  semantics issues on macOS).

## License

[MIT](LICENSE). Models pulled at runtime have their own licenses — see
their respective Hugging Face model cards.

## Acknowledgements

- [`mlx_lm`](https://github.com/ml-explore/mlx-lm) and [`mlx_audio`](https://github.com/Blaizzy/mlx-audio) for the local Apple Silicon model runtime.
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for fast streaming TTS.
- [Qwen3-ASR](https://huggingface.co/Qwen) and [Qwen3.5](https://huggingface.co/Qwen) for ASR + LLM.
- [`pytgcalls`](https://github.com/pytgcalls/pytgcalls) and [Telethon](https://github.com/LonamiWebs/Telethon) for the Telegram audio bridge.
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) for the lightweight voice activity detector.
