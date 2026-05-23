# Session-to-session calling and call-transcript verification

This walks through using **two Telegram accounts on the same machine** to verify
the audio path end-to-end: one session places an outbound private call, the other
auto-accepts and records what it hears, and we then run ASR over the recording
to confirm the audio is intelligible.

## Sessions in this repo

Each Telethon `.session` file in the repo root is a logged-in account. The
project currently has two:

| Session file        | Account name | Phone           | Telegram user id |
| ------------------- | ------------ | --------------- | ---------------- |
| `bob.session`       | Chippy       | +6588461730     | 8943154461       |
| `telecall.session`  | katey        | +91 7881174135  | 6564185941       |

Both sessions reuse the same `TG_API_ID` / `TG_API_HASH` from `.env`. To re-derive
this table:

```bash
set -a && . ./.env && set +a
uv run --project telegramcaller python - <<'PY'
import asyncio, os
from telethon import TelegramClient
API_ID, API_HASH = int(os.environ["TG_API_ID"]), os.environ["TG_API_HASH"]
async def show(name):
    async with TelegramClient(name, API_ID, API_HASH) as c:
        me = await c.get_me()
        print(name, me.id, me.first_name, "+"+me.phone)
asyncio.run(asyncio.gather(show("bob"), show("telecall")))
PY
```

## End-to-end flow

```
katey (telecall.session) --[ MTProto private call ]--> Chippy (bob.session)
        |                                                       |
   streams test_phrase.wav                              accepts, plays silence,
   via pytgcalls MediaStream                           records peer audio -> WAV
                                                                |
                                                  Qwen3-ASR-0.6B-4bit (mlx_audio)
                                                                |
                                                            transcript.txt
```

There are three scripts under `scripts/`:

- **`receiver.py`** — runs as the callee. Auto-accepts any incoming private call,
  records the caller's audio to `recordings/inbound_<ts>_<chat_id>.wav`
  (mono, 48 kHz), and plays silence back so the call stays connected.
- **`caller.py`** — runs as the caller. Resolves the callee (imports them as a
  contact first if needed, which is required when the accounts have never
  interacted), then places a private call and streams a local audio file into
  it via `pytgcalls.MediaStream`.
- **`transcribe.py`** — loads the same `mlx-community/Qwen3-ASR-0.6B-4bit` model
  `lintvoiceagent` uses and writes a `.txt` transcript next to the WAV.

## Running the test

Open two terminals.

### Terminal 1 — Chippy listens

```bash
cd /Users/vas/Documents/voice-evolve
set -a && . ./.env && set +a
TG_SESSION=bob uv run --project telegramcaller \
  python scripts/receiver.py
```

Wait for the `[recv] READY as Chippy (+...)` line.

### Terminal 2 — katey calls Chippy

```bash
cd /Users/vas/Documents/voice-evolve
set -a && . ./.env && set +a
TG_SESSION=telecall uv run --project telegramcaller \
  python scripts/caller.py \
    --to-phone +6588461730 \
    --audio test_audio/test_phrase.wav \
    --hold 25
```

The caller imports Chippy as a contact (only needed the first time), places the
call, streams `test_phrase.wav` for ~25 s, then hangs up.

Expected log on the receiver:

```
[recv] INCOMING_CALL from 6564185941 — accepting
[recv] recording -> .../recordings/inbound_<ts>_6564185941.wav
[recv] armed
[recv] DISCARDED_CALL 6564185941
[recv] saved .../recordings/inbound_<ts>_6564185941.wav
```

### Transcribe the recording

`lintvoiceagent` already pins `mlx_audio` with the ASR model weights, so run the
transcriber under that project's environment:

```bash
uv run --project lintvoiceagent \
  python scripts/transcribe.py \
    recordings/inbound_<ts>_6564185941.wav
```

The transcript prints to stdout and is also saved as
`inbound_<ts>_6564185941.txt` next to the WAV. Compare against the canned
phrase in `test_audio/test_phrase.wav` to confirm audio reached
the callee intact and the ASR path produces a usable transcript.

## Regenerating the test phrase

`test_phrase.wav` was synthesised with the macOS `say` tool so it's
deterministic and easy to ASR-check:

```bash
say -v Samantha -o /tmp/p.aiff "The quick brown fox jumps over the lazy dog. \
This is a test of the Telegram voice bridge. Calling from katey to chippy."
ffmpeg -y -i /tmp/p.aiff -ac 1 -ar 48000 test_audio/test_phrase.wav
rm /tmp/p.aiff
```

Any mono 48 kHz WAV/MP3 works as the source — pytgcalls hands it to ffmpeg
internally.

## Known pitfalls

- **`Could not find the input entity`** on the caller — the two accounts have
  no prior interaction. Use `--to-phone` so `caller.py` runs
  `ImportContactsRequest` before resolving the entity.
- **`TelegramServerError()` from `calls.play(...)`** — usually a stale call on
  Telegram's side from a previous run. Kill both scripts, wait ~10 s, and retry.
  If it persists, send a normal Telegram message between the two accounts once
  to "warm" the relationship, then try again.
- **`Unknown VOICE_PROVIDER='elevenlabs'`** — that's the original
  `bridge.py` complaining; the scripts in this doc don't touch a voice provider
  at all, they just stream/record raw audio. Leave `VOICE_PROVIDER` as-is.
- **Receiver records silence** — verify `ffmpeg` is on PATH, the `recordings/`
  dir is writable, and that the call actually connected (`DISCARDED_CALL`
  appears in the receiver log). The MP3-FIFO → ffmpeg → WAV pipeline only
  produces samples while peer audio is flowing.
