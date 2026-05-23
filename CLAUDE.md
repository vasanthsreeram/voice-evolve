# voice-evolve

## What this repo is for

A **multi-hour Claude Code optimization harness** for a local voice-to-voice
Telegram calling stack. Two local AI agents place real Telegram calls to each
other, hold a real conversation, and hang up. Claude Code iterates on the
voice interface against a goal function across many calls until it converges.

The session-to-session call setup is the **eval harness**, not the product.
Every iteration: place a call → record → transcribe → score → propose change →
rerun. Optimize for the goal function's behavior across many iterations, not
for any single call.

> Goal function is not finalized — confirm with the user before assuming.
> Likely candidates: ASR WER on the callee side, end-to-end turn latency,
> MOS-style naturalness, successful call completion rate, or a composite.

Everything must run locally.

## Layout

```
voice-evolve/
├── bob.session              # Telethon session — Chippy (+6588461730, id 8943154461)
├── telecall.session         # Telethon session — katey  (+917881174135, id 6564185941)
├── .env                     # TG_API_ID, TG_API_HASH, provider keys
│
├── telegramcaller/          # upstream — turns one Telegram session into a voice agent
│                            # (bridge.py, providers/, login.py). Treat as vendored.
├── lintvoiceagent/          # local voice stack — ASR (Qwen3-ASR-0.6B-4bit via mlx_audio),
│                            # TTS (kokoro / supertonic), LLM (mlx_lm), VAD. See its CLAUDE.md.
│
├── scripts/                 # eval-harness glue (don't move into telegramcaller/)
│   ├── receiver.py          # auto-accepts inbound call, records peer audio -> recordings/*.wav
│   ├── caller.py            # places outbound call, streams a given audio file
│   └── transcribe.py        # runs the lintvoiceagent ASR over a wav
├── test_audio/              # deterministic test phrases for the caller to stream
├── recordings/              # WAVs from accepted inbound calls (gitignored)
└── docs/
    └── SESSION_TO_SESSION.md  # full walkthrough of running the call+transcribe loop
```

## The two agent identities

| Session file       | Account | Phone           | Telegram user id |
| ------------------ | ------- | --------------- | ---------------- |
| `bob.session`      | Chippy  | +6588461730     | 8943154461       |
| `telecall.session` | katey   | +917881174135   | 6564185941       |

Both sessions share `TG_API_ID` / `TG_API_HASH` from `.env`. To re-derive
the table, see `docs/SESSION_TO_SESSION.md`.

## Running the harness loop (manual today)

See `docs/SESSION_TO_SESSION.md` for the full procedure. The short form:

```bash
# Terminal 1 — Chippy receives + records
set -a && . ./.env && set +a
TG_SESSION=bob uv run --project telegramcaller python scripts/receiver.py

# Terminal 2 — katey dials Chippy and streams a test phrase
set -a && . ./.env && set +a
TG_SESSION=telecall uv run --project telegramcaller python scripts/caller.py \
  --to-phone +6588461730 \
  --audio test_audio/test_phrase.wav \
  --hold 25

# After call ends — transcribe with the local ASR
uv run --project lintvoiceagent \
  python scripts/transcribe.py recordings/inbound_<ts>_6564185941.wav
```

## Working agreements

- **Don't put eval-harness code or docs inside `telegramcaller/`.** That's
  vendored upstream; keep our glue at the repo root (`scripts/`, `docs/`,
  `test_audio/`).
- **Don't commit or push unless explicitly asked.** This repo is in active
  exploration mode.
- **Don't mock the Telegram path.** The harness only proves something if the
  audio actually round-trips through MTProto. Mock-based scoring is misleading.
- **Local-only.** No cloud ASR/TTS/LLM by default. Provider keys in `.env`
  are for the existing bridge.py options, not for evaluating the agent.
- **Iteration-aware changes.** When tweaking the voice stack, optimize for the
  goal function across many iterations — don't hand-tune for one call.

## Known blockers

- **Outbound userbot→userbot calls fail with `TelegramServerError()`** from
  `pytgcalls 2.2.11`'s `calls.play()`. The receiver leg (auto-accept + record)
  works fine; the *initiation* from a script-driven account is what Telegram
  is rejecting — likely a new-account / no-prior-interaction throttle.
  Until this is unblocked, every harness iteration needs a human to dial from
  the official Telegram app, which makes the multi-hour loop impossible.
  Things to try: upgrade pytgcalls to 2.2.12, warm the katey↔Chippy
  relationship with normal messages, or swap katey for a more established account.

## Pointers

- `telegramcaller/bridge.py` — reference for how an inbound call gets bridged
  to a voice provider (OpenAI realtime by default). Useful as the template
  when wiring our local stack into the call instead of silence.
- `lintvoiceagent/CLAUDE.md` and `lintvoiceagent/app.py` — local ASR/TTS/LLM
  patterns to reuse.
- `docs/SESSION_TO_SESSION.md` — operational runbook for the harness.
