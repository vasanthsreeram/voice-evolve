#!/usr/bin/env python3
"""Benchmark SuperTonic TTS speed for full VAD->TTS pipeline analysis."""
import time
import numpy as np
from supertonic import TTS

print("Loading SuperTonic...")
t0 = time.perf_counter()
model = TTS(auto_download=True)
load_time = time.perf_counter() - t0
print(f"Loaded in {load_time:.2f}s\n")

style = model.get_voice_style(voice_name="M1")

# Warmup
print("Warmup...")
_ = model.synthesize("Hello world.", voice_style=style, lang="en")

SR = 44100

test_texts = [
    ("short (1 sentence)", "Yeah, what do you want."),
    ("medium (2 sentences)", "Look, I-I-I don't have time for this. Just ask your question and let me get back to my work."),
    ("long (3 sentences)", "Listen, the universe is basically a chaotic mess held together by physics nobody really understands. Most of what you call reality is just probability collapsing in ways that make you feel important. Uurp. Now, what is it."),
    ("very long (5 sentences)", "Okay, so here's the thing about consciousness, right. It's not some magical thing — it's just neurons firing in patterns that happen to model themselves. The hard problem isn't actually hard, it's just badly framed by people who never took a real physics class. I've been to dimensions where consciousness runs on crystal lattices, and guess what, same boring outcome. So stop asking deep questions and start asking useful ones."),
]

print(f"{'case':<25} {'gen_s':>8} {'audio_s':>8} {'RTF':>6} {'first_play_lat':>15}")
print("-" * 70)

results = []
for name, text in test_texts:
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        wav, dur = model.synthesize(text, voice_style=style, lang="en")
        gen = time.perf_counter() - t0
        audio_np = np.asarray(wav, dtype=np.float32).squeeze()
        audio_s = audio_np.shape[0] / SR
        times.append((gen, audio_s))
    gen = min(t[0] for t in times)
    audio_s = times[0][1]
    rtf = gen / audio_s
    # SuperTonic is non-streaming: client cannot play until full chunk is generated
    print(f"{name:<25} {gen:>8.3f} {audio_s:>8.2f} {rtf:>6.3f} {gen*1000:>13.0f}ms")
    results.append((name, gen, audio_s, rtf))

print()
print("=== Full pipeline estimate (SuperTonic for TTS) ===")
# Rough numbers from app.py logs / typical M-series Mac (VAD ~instant, ASR ~300ms, LLM Qwen3.5-4B 4-bit ~600ms TTFT + generation)
# We can't easily run the full LLM here; use known approximate latencies.
print("Per-stage typical latency on Apple Silicon (approx):")
print("  VAD (Silero) end-of-turn detect : ~200-400 ms (depends on silence_threshold)")
print("  ASR (Qwen3-ASR-0.6B 4-bit)      : ~250-500 ms for short utterance")
print("  LLM (Qwen3.5-4B 4-bit, VLM mode): generates full response BEFORE TTS in current code")
print("                                    ~800-2000 ms for 1-3 sentences")
print("  TTS (SuperTonic, measured above): see table — full chunk before first audio")
print()
print("Current app.py is NON-STREAMING for VLM path (line 218): vlm_generate blocks,")
print("then SuperTonic blocks. So mouth-to-ear latency ~= VAD + ASR + LLM_total + TTS_total.")
print()
print("Expected user-perceived delay (silence end -> first audio plays):")
for name, gen, audio_s, rtf in results:
    # assume 350ms VAD + 350ms ASR + 1200ms LLM (medium response) + gen
    total = 0.35 + 0.35 + 1.2 + gen
    print(f"  {name:<25} ~ {total*1000:>5.0f} ms  (TTS gen: {gen*1000:.0f}ms, RTF={rtf:.2f})")
