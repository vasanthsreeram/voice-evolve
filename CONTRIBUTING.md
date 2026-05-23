# Contributing to voice-evolve

Thanks for poking at this. It's an experiment — issues, PRs, and forks are
all welcome.

## Getting set up

You'll need:

- macOS on Apple Silicon (the MLX pieces don't run elsewhere). Other OSes
  could work for the orchestration but the ASR/LLM/TTS path is MLX-only.
- Python 3.12 and [`uv`](https://docs.astral.sh/uv/).
- Two Telegram accounts that have texted each other once. Get
  `TG_API_ID` / `TG_API_HASH` from <https://my.telegram.org>.
- (Optional) Gemini API key from <https://aistudio.google.com> for the
  judge + optimization loop.
- A reasonable amount of disk for model weights (~6–8 GB) and a few GB of
  scratch space under `recordings/`.

Install:

```bash
uv sync --project lintvoiceagent
cp .env.example .env  # then fill in your keys
```

See [README.md](README.md) for the full first-call walkthrough.

## What kinds of changes are most welcome

Roughly in priority order:

1. **Lower latency without breaking turns.** The current floor is
   architectural (MP3-via-FIFO inbound, ~300 ms of buffering). A working
   `StreamFrames` raw-PCM inbound path on `pytgcalls` 2.2.12 would be
   huge. Don't ship it if you can't keep the existing tests passing.
2. **A working half-duplex protocol** that prevents both agents from
   speaking simultaneously, that actually survives in two separate
   processes. We tried `fcntl.flock` and it had per-fd lock semantics
   issues on macOS that caused 6 s wait-and-speak fallbacks every turn.
3. **Sliding-window chat history** so LLM prefill cost stays roughly
   constant as a conversation grows. We saw TTFA grow from ~300 ms at
   turn 1 to ~750 ms at turn 12 because the cache had to prefill ~30
   tokens per turn.
4. **Smart-turn semantic VAD.** There's a stub for pipecat-ai's
   `smart-turn-v3` in `lintvoiceagent/vad_detector.py`. Wire it up as an
   alternative to FireRedVAD's silence-count heuristic.
5. **Better personas.** The system prompts in `scripts/run_agent.py` are
   passable. The Qwen 4B sometimes refuses politically charged topics —
   personas that route around that gracefully would help.
6. **Documentation.** If you read this and got stuck, that's a doc bug.
   Open an issue or PR with the section that needs work.

## What kinds of changes need more care

- **Anything that touches `voice_agent.py`'s VAD logic.** Three different
  bugs lived there over the course of one debug session. The current
  algorithm (accumulate-all, track saw-speech, gate on sustained-silence +
  bytes + RMS) is fragile. If you change it, run `scripts/run_duet_local.py`
  with the existing settings and confirm at least two clean turns before
  pushing.
- **Anything that changes the Gemini judge prompt.** The optimization loop
  reads structured JSON; if you change the schema you'll break the loop.
- **Adding cloud dependencies.** The point of this repo is that the agents
  run locally. Gemini-as-judge is the one allowed cloud call, and it's
  optional. Don't introduce a runtime dependency on an external LLM/ASR/TTS.

## Style

- Python target: 3.12.
- No formatter is enforced. Match the surrounding style.
- Don't add docstrings to obvious functions; do explain non-obvious
  invariants in comments. There are several "this used to break because X"
  comments in `voice_agent.py` that are load-bearing.
- Don't add new top-level dependencies if a vendored alternative exists.

## PR checklist

- [ ] Ran `scripts/run_duet_local.py` once and saw at least two clean turns
      in the resulting `recordings/local_*_convo.txt`.
- [ ] If you touched Telegram-side code: ran a real call (not just a local
      duet) and confirmed at least one round-trip in `recordings/duet_*_convo.txt`.
- [ ] No `.env`, `.session`, or `recordings/*` files committed.
- [ ] Updated `README.md` or `docs/` if user-facing behavior changed.
- [ ] If you added a new env var, added it to the knob table in `README.md`.

## Issue templates

When you open an issue:

- **Bug**: include the full `recordings/loop_duet_*.log` for a failing
  run, the run id, and the relevant `_turns.jsonl`.
- **Feature**: a sentence on what you'd build, a sentence on what it
  would let users do that they currently can't.
- **Question**: usually starts with re-reading the README's
  *Known limits* section, then asking.

## Code of conduct

Be kind, especially about the parts of the code that are clearly held
together with duct tape. We're all just trying to make two laptops have a
phone conversation.
