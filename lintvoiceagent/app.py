#!/usr/bin/env python3
import eventlet
eventlet.monkey_patch()

import os
import time
import wave
import tempfile
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from mlx_lm import load as lm_load, stream_generate
from mlx_audio.stt.utils import load_model as load_asr_model
from mlx_audio.stt.generate import generate_transcription
from collections import defaultdict
import threading
import torch
from vad_detector import create_vad_detector
from vad_config import get_vad_config, STREAMING_INTERVAL_SECONDS, MIN_BUFFER_SECONDS, OVERLAP_SECONDS
from streaming_tts import StreamingTTS, TextChunker, audio_to_base64, AVAILABLE_VOICES, DEFAULT_VOICE
from supertonic_tts import (
    SupertonicStreamingTTS,
    AVAILABLE_VOICES as SUPERTONIC_VOICES,
    DEFAULT_VOICE as SUPERTONIC_DEFAULT_VOICE,
)
from kokoro_streaming import (
    KokoroStreamingTTS,
    AVAILABLE_VOICES as KOKORO_VOICES,
    DEFAULT_VOICE as KOKORO_DEFAULT_VOICE,
)
from session_logger import get_logger as get_session_logger, drop_logger as drop_session_logger, record_turn_summary, recent_turns

# TTS engines exposed to the UI. Keys map to engine ids used over the socket.
TTS_ENGINES = {
    "kokoro": {"label": "Kokoro-82M (fast streaming)", "voices": KOKORO_VOICES, "default_voice": KOKORO_DEFAULT_VOICE},
    "qwen3": {"label": "Qwen3-TTS (cloning)", "voices": AVAILABLE_VOICES, "default_voice": DEFAULT_VOICE},
    "supertonic": {"label": "Supertonic-3", "voices": SUPERTONIC_VOICES, "default_voice": SUPERTONIC_DEFAULT_VOICE},
}
DEFAULT_ENGINE = "kokoro"

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=50000000)

# Load transcription model (Qwen3-ASR-0.6B-4bit via mlx_audio)
print("Loading Qwen3-ASR-0.6B-4bit...")
transcription_model = load_asr_model("mlx-community/Qwen3-ASR-0.6B-4bit")
print("ASR model loaded!")

# LLM: in-process via mlx_lm (text-only — no vision/camera support).
# Override with env LLM_MODEL_NAME if you want a different MLX model.
LLM_MODEL_NAME = os.environ.get(
    "LLM_MODEL_NAME",
    os.path.expanduser("~/.lmstudio/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"),
)
print(f"Loading LLM via mlx_lm: {LLM_MODEL_NAME}")
llm_model, llm_tokenizer = lm_load(LLM_MODEL_NAME)
print("LLM loaded!")

# TTS engines loaded lazily on first use, one instance per engine id.
tts_engines = {}

def get_tts_engine(engine_id):
    """Lazy-load and cache the TTS engine for the given id."""
    if engine_id not in tts_engines:
        if engine_id == "kokoro":
            print("Loading Kokoro-82M TTS...")
            tts_engines[engine_id] = KokoroStreamingTTS(voice=KOKORO_DEFAULT_VOICE, speed=1.0)
        elif engine_id == "supertonic":
            print("Loading Supertonic-3 TTS...")
            tts_engines[engine_id] = SupertonicStreamingTTS(voice=SUPERTONIC_DEFAULT_VOICE, speed=1.0)
        else:
            print("Loading Qwen3-TTS model...")
            tts_engines[engine_id] = StreamingTTS(voice='af_heart', speed=1.0, use_gpu=False)
        print(f"TTS engine '{engine_id}' loaded!")
    return tts_engines[engine_id]

# Default VAD mode for new sessions
DEFAULT_VAD_MODE = "firered"

# --- Eager warm-up: only the engines we actually use by default.
#     Supertonic + Silero remain available via socket events (lazy-load). --
print("Pre-warming Kokoro-82M TTS (default, sub-phrase streaming)...")
get_tts_engine("kokoro")

print("Pre-warming FireRedVAD...")
_warm_vad_firered = create_vad_detector("firered", **get_vad_config("firered"))
print("All models warm. Server ready to accept connections.")

# Session-based audio buffers for streaming
audio_buffers = defaultdict(list)
buffer_locks = defaultdict(threading.Lock)
accumulated_text = defaultdict(str)  # Track accumulated transcription per session
conversation_history = defaultdict(list)  # Track conversation history for each session
last_partial_text = defaultdict(str)  # Track last partial transcription to avoid duplicates
vad_detectors = {}  # Store VAD detector per session
vad_modes = defaultdict(lambda: DEFAULT_VAD_MODE)  # Track VAD mode per session
session_voices = defaultdict(lambda: TTS_ENGINES[DEFAULT_ENGINE]["default_voice"])  # Track TTS voice per session
session_engines = defaultdict(lambda: DEFAULT_ENGINE)  # Track TTS engine per session
session_speeds = defaultdict(lambda: 1.0)  # Track TTS speed per session
session_images = {}  # Latest camera frame per session (base64 JPEG, None if camera off)
playback_active = defaultdict(bool)  # Track whether client is playing TTS audio

# VAD thresholds during playback (very high = only deliberate loud speech gets through)
PLAYBACK_VAD_THRESHOLD = 0.92
PLAYBACK_ENERGY_THRESHOLD = 0.08
silence_counters = defaultdict(int)  # Track consecutive silent chunks (for Silero only)
# New: maintain a full-buffer per utterance to ensure final transcription sees all speech
utterance_audio_full = defaultdict(list)
utterance_active = defaultdict(bool)
# Track assistant generation per session for cancellation/barge-in
current_generation_id = defaultdict(int)
generation_active = defaultdict(bool)

# Load configuration from vad_config.py
STREAMING_CHUNK_SIZE = int(STREAMING_INTERVAL_SECONDS * 16000)  # Convert seconds to samples
MIN_BUFFER_SIZE = int(MIN_BUFFER_SECONDS * 16000)
OVERLAP_SAMPLES = int(OVERLAP_SECONDS * 16000)

def _resample_linear(x, sr_in, sr_out):
    """Lightweight linear resampler"""
    if sr_in == sr_out or x.size == 0:
        return x
    duration = x.shape[0] / float(sr_in)
    n_out = max(1, int(round(duration * sr_out)))
    t_in = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    t_out = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)

# 4th-order Butterworth high-pass at 80Hz (16kHz sample rate).
# Cuts HVAC rumble, mic handling thumps, AC mains hum.
from scipy.signal import butter, sosfilt
_HP_SOS = butter(4, 80.0, btype='highpass', fs=16000, output='sos')

def _highpass_80hz(x):
    if x.size < 16:
        return x
    return sosfilt(_HP_SOS, x).astype(np.float32)

def get_vad_detector(sid):
    """Get or create VAD detector for session"""
    if sid not in vad_detectors:
        vad_mode = vad_modes[sid]
        # Get configuration for this VAD type
        config = get_vad_config(vad_mode)
        vad_detectors[sid] = create_vad_detector(vad_mode, **config)
    return vad_detectors[sid]

def get_llm_response_streaming(conversation_messages, sid, gen_id=None, image_b64=None):
    """Generate LLM response from conversation history with streaming TTS"""
    tts_engine = get_tts_engine(session_engines[sid])

    try:
        # System prompt for voice assistant
        system_prompt = (
            "You are Rick Sanchez, the genius scientist from dimension C-137. "
            "You speak through a voice interface — you ARE Rick, talking out loud. Never reference being an AI or text model. "
            "Never use markdown, asterisks, bullet points, or any formatting — everything you say is spoken aloud. "
            "Keep responses short and punchy, 1-3 sentences unless the person actually asks you to explain something, which, let's be honest, they probably can't even understand. "
            "You are multilingual — respond in whatever language the user speaks to you in. Match their language naturally. "
            "Channel Rick's personality: brilliant, cynical, impatient with stupidity, peppered with burps and stutters. "
            "You actually help people — you're a genius after all — but you make it clear you think most questions are beneath you. "
            "You are NOT talking to Morty. You're talking to random strangers who found your voice portal. "
            "Don't call them Morty — you don't know who they are. If they're dumb, just call them out directly. "
            "You can reference Morty, the family, your adventures, etc. in passing — but the person you're talking to is not any character from the show. "
            "Existential nihilism, sci-fi tangents, and the occasional rant about the government are all fair game. "
            "Use filler sounds like 'uurp', 'look', 'listen', 'I-I-I' stutters naturally. "
            "You're the smartest being in the multiverse and everyone should know it."
        )
        
        messages_with_system = [{"role": "system", "content": system_prompt}] + conversation_messages

        # Apply the model's chat template; fall back to a plain dump if missing.
        if llm_tokenizer.chat_template is not None:
            prompt = llm_tokenizer.apply_chat_template(
                messages_with_system,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = f"System: {system_prompt}\n" + "\n".join(
                [f"{m['role']}: {m['content']}" for m in conversation_messages]
            )

        # Create text chunker for TTS
        chunker = TextChunker(mode='phrase')  # Use 'phrase' for faster streaming

        # Stream generate response
        full_response = ""
        token_count = 0

        # Session logger refs (turn must have been started by audio_chunk handler)
        slog = get_session_logger(sid)
        slog.mark('llm_start')
        tts_first_audio_marked = False

        # In-process token streaming via mlx_lm
        for response in stream_generate(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=1024,
        ):
            # Check for cancellation (barge-in)
            if gen_id is not None and current_generation_id.get(sid, 0) != gen_id:
                print(f"[CANCEL] Stopping token stream for sid={sid}, gen_id={gen_id}")
                break
            token = response.text if hasattr(response, 'text') else str(response)
            full_response += token
            token_count += 1

            # Log first token
            if token_count == 1:
                print(f"[STAGE 4 - LLM] First token generated: '{token}'")
                slog.mark('llm_first_token')

            # Emit text token to display in UI IMMEDIATELY
            socketio.emit('assistant_token', {'token': token}, room=sid)
            # Yield to allow transports to flush
            socketio.sleep(0)

            # Periodic full-text sync to fix any dropped token events
            if token_count % 12 == 0:
                socketio.emit('assistant_progress', {'text': full_response}, room=sid)
                socketio.sleep(0)

            # Add token to chunker and generate audio for complete chunks
            for text_chunk in chunker.add_token(token):
                # STAGE 5: TTS Generation
                print(f"[STAGE 5 - TTS] Generating audio for chunk: '{text_chunk[:30]}...'")
                socketio.emit('pipeline_stage', {
                    'stage': 'tts_start',
                    'message': f'Generating speech for: {text_chunk[:30]}...'
                }, room=sid)

                # Generate audio for this chunk; time each yielded chunk's actual synth.
                _t_chunk = time.time()
                for audio_bytes in tts_engine.generate_audio_chunk(text_chunk, voice=session_voices[sid], speed=session_speeds[sid]):
                    chunk_synth_ms = (time.time() - _t_chunk) * 1000.0
                    # Check for cancellation (barge-in)
                    if gen_id is not None and current_generation_id.get(sid, 0) != gen_id:
                        print(f"[CANCEL] Stopping TTS stream for sid={sid}, gen_id={gen_id}")
                        break
                    if not tts_first_audio_marked:
                        slog.mark('tts_first_audio')
                        tts_first_audio_marked = True
                    slog.add_tts_chunk_wav(audio_bytes, synth_ms=chunk_synth_ms)
                    _t_chunk = time.time()
                    # Convert to base64 and emit to frontend
                    audio_b64 = audio_to_base64(audio_bytes)
                    print(f"[STAGE 5 - TTS] Sending audio chunk ({len(audio_bytes)} bytes)")
                    socketio.emit('assistant_audio', {
                        'audio': audio_b64,
                        'sample_rate': 24000
                    }, room=sid)
                    # Yield so client can start playback immediately
                    socketio.sleep(0)

        # Mark LLM end (token stream exhausted)
        slog.mark('llm_end')

        # Process any remaining text in the buffer
        remaining = chunker.flush()
        if remaining:
            _t_chunk = time.time()
            for audio_bytes in tts_engine.generate_audio_chunk(remaining, voice=session_voices[sid], speed=session_speeds[sid]):
                chunk_synth_ms = (time.time() - _t_chunk) * 1000.0
                if gen_id is not None and current_generation_id.get(sid, 0) != gen_id:
                    print(f"[CANCEL] Stopping remaining TTS for sid={sid}, gen_id={gen_id}")
                    break
                if not tts_first_audio_marked:
                    slog.mark('tts_first_audio')
                    tts_first_audio_marked = True
                slog.add_tts_chunk_wav(audio_bytes, synth_ms=chunk_synth_ms)
                _t_chunk = time.time()
                audio_b64 = audio_to_base64(audio_bytes)
                socketio.emit('assistant_audio', {
                    'audio': audio_b64,
                    'sample_rate': 24000
                }, room=sid)
                socketio.sleep(0)

        slog.mark('tts_end')
        slog.set_assistant_text(full_response)

        # Final progress sync
        socketio.emit('assistant_progress', {'text': full_response}, room=sid)
        socketio.sleep(0)

        return full_response.strip()
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


def _run_llm_and_tts_for_sid(sid, gen_id):
    """Background task: stream LLM tokens and TTS audio for this sid."""
    try:
        # Snapshot conversation without holding per-sid lock to avoid KeyErrors
        # if the client disconnects while this task runs.
        history = list(conversation_history.get(sid, []))

        # Stream tokens + audio
        llm_response = get_llm_response_streaming(history, sid, gen_id=gen_id, image_b64=session_images.get(sid))

        # If not cancelled, append and emit completion
        if current_generation_id.get(sid, 0) == gen_id:
            try:
                conversation_history[sid].append({"role": "assistant", "content": llm_response})
            except Exception:
                pass

            print(f"[STAGE 4 - LLM] LLM complete: '{llm_response[:50]}...'")
            socketio.emit('pipeline_stage', {
                'stage': 'llm_complete',
                'message': 'LLM response complete'
            }, room=sid)
            socketio.emit('assistant_complete', {'text': llm_response}, room=sid)

            # Finalize turn log: writes assistant wav + appends turns.jsonl
            try:
                rec = get_session_logger(sid).finish_turn()
                record_turn_summary(sid, rec)
                if rec is not None:
                    socketio.emit('turn_complete', {
                        'turn': rec['turn'],
                        'user_text': rec.get('user_text'),
                        'assistant_text': rec.get('assistant_text'),
                        'latency_ms': rec['latency_ms'],
                    }, room=sid)
            except Exception as e:
                print(f"[LOG] finish_turn failed: {e}")
    except Exception as e:
        print(f"Background LLM task error (sid={sid}): {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Mark generation as inactive if this task corresponds to current gen
        if current_generation_id.get(sid, 0) == gen_id:
            generation_active[sid] = False

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/turns')
def turns():
    """Per-turn latency log for the most recent turns across all sessions."""
    return {"turns": recent_turns(limit=20)}

# ---------------------------------------------------------------------------
# History: browse past sessions, drill into turn-level breakdown + audio
# ---------------------------------------------------------------------------
import json as _json2

LOGS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

@app.route('/sessions')
def list_sessions():
    """JSON list of all sessions on disk with turn counts + timestamps."""
    if not os.path.isdir(LOGS_ROOT):
        return {"sessions": []}
    items = []
    for sid in sorted(os.listdir(LOGS_ROOT)):
        sdir = os.path.join(LOGS_ROOT, sid)
        if not os.path.isdir(sdir):
            continue
        jsonl = os.path.join(sdir, "turns.jsonl")
        if not os.path.exists(jsonl):
            continue
        try:
            stat = os.stat(jsonl)
            with open(jsonl) as f:
                lines = f.readlines()
            n = len(lines)
            first = _json2.loads(lines[0]) if lines else {}
            last = _json2.loads(lines[-1]) if lines else {}
            items.append({
                "sid": sid,
                "turns": n,
                "started_at": first.get("started_at"),
                "ended_at": last.get("finished_at"),
                "first_user_text": first.get("user_text"),
                "last_assistant_text": (last.get("assistant_text") or "")[:120],
                "mtime": stat.st_mtime,
            })
        except Exception as e:
            items.append({"sid": sid, "error": str(e)})
    items.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return {"sessions": items}

@app.route('/sessions/<sid>/turns')
def session_turns(sid):
    """Full turn-level breakdown (incl. raw stage timestamps) for one session."""
    safe = sid.replace("..", "").replace("/", "")
    jsonl = os.path.join(LOGS_ROOT, safe, "turns.jsonl")
    if not os.path.exists(jsonl):
        return {"error": "not found"}, 404
    with open(jsonl) as f:
        rows = [_json2.loads(l) for l in f if l.strip()]
    return {"sid": safe, "turns": rows}

@app.route('/sessions/<sid>/audio/<fname>')
def session_audio(sid, fname):
    """Serve a turn's user/assistant wav."""
    from flask import send_from_directory, abort
    safe_sid = sid.replace("..", "").replace("/", "")
    safe_fn = fname.replace("..", "").replace("/", "")
    if not safe_fn.endswith(".wav"):
        abort(400)
    return send_from_directory(os.path.join(LOGS_ROOT, safe_sid), safe_fn, mimetype="audio/wav")

@app.route('/history')
def history_page():
    """Browser UI to inspect past sessions, turn-by-turn timings, and audio."""
    return render_template('history.html')

@app.route('/stats')
def stats():
    """Memory + engine status snapshot. Hit GET /stats from browser/curl."""
    import psutil, resource
    proc = psutil.Process()
    mem = proc.memory_info()
    sys_mem = psutil.virtual_memory()

    # MLX active memory (if available)
    mlx_active_gb = None
    try:
        import mlx.core as mx
        mlx_active_gb = round(mx.metal.get_active_memory() / 1e9, 2)
    except Exception:
        pass

    def gb(b): return round(b / 1e9, 2)
    return {
        "process": {
            "rss_gb": gb(mem.rss),
            "vms_gb": gb(mem.vms),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        },
        "system": {
            "used_gb": gb(sys_mem.used),
            "available_gb": gb(sys_mem.available),
            "total_gb": gb(sys_mem.total),
            "percent": sys_mem.percent,
        },
        "mlx_active_gb": mlx_active_gb,
        "engines": {
            "asr": "Qwen3-ASR-0.6B-4bit (mlx_audio)",
            "llm": f"mlx_lm in-process ({LLM_MODEL_NAME.split('/')[-1]})",
            "tts_loaded": list(tts_engines.keys()),
            "vads_warm": ["firered"],
        },
        "sessions": {
            "active": len(session_engines),
            "voices_in_use": dict({k: session_voices[k] for k in session_engines}),
        },
    }

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to server'})
    # Set default voice for this session
    session_voices[request.sid] = TTS_ENGINES[DEFAULT_ENGINE]["default_voice"]

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    with buffer_locks[sid]:
        if sid in audio_buffers:
            del audio_buffers[sid]
        if sid in utterance_audio_full:
            del utterance_audio_full[sid]
        if sid in utterance_active:
            del utterance_active[sid]
        if sid in accumulated_text:
            del accumulated_text[sid]
        if sid in conversation_history:
            del conversation_history[sid]
        if sid in silence_counters:
            del silence_counters[sid]
        if sid in last_partial_text:
            del last_partial_text[sid]
        if sid in vad_detectors:
            del vad_detectors[sid]
        if sid in vad_modes:
            del vad_modes[sid]
        if sid in buffer_locks:
            del buffer_locks[sid]

@socketio.on('set_engine')
def handle_set_engine(data):
    sid = request.sid
    engine_id = data.get('engine', DEFAULT_ENGINE)
    if engine_id not in TTS_ENGINES:
        print(f"[TTS] Unknown engine '{engine_id}' for {sid}")
        return
    session_engines[sid] = engine_id
    # Reset to a valid default voice for the new engine.
    session_voices[sid] = TTS_ENGINES[engine_id]["default_voice"]
    print(f"[TTS] Engine set to '{engine_id}' for {sid}")
    socketio.emit('tts_engine_set', {
        'engine': engine_id,
        'voices': TTS_ENGINES[engine_id]['voices'],
        'voice': session_voices[sid],
    }, to=sid)

@socketio.on('get_tts_options')
def handle_get_tts_options():
    sid = request.sid
    socketio.emit('tts_options', {
        'engines': {k: {'label': v['label'], 'voices': v['voices']} for k, v in TTS_ENGINES.items()},
        'current_engine': session_engines[sid],
        'current_voice': session_voices[sid],
    }, to=sid)

@socketio.on('set_speed')
def handle_set_speed(data):
    sid = request.sid
    speed = float(data.get('speed', 1.0))
    speed = max(0.5, min(2.0, speed))  # clamp
    session_speeds[sid] = speed
    print(f'[TTS] Speed set to {speed}x for {sid}')

@socketio.on('camera_frame')
def handle_camera_frame(data):
    """Store latest camera frame for use in next LLM call."""
    sid = request.sid
    session_images[sid] = data.get('image')  # base64 JPEG

@socketio.on('camera_disabled')
def handle_camera_disabled():
    sid = request.sid
    session_images.pop(sid, None)

def _set_vad_sensitivity(detector, speech_thresh, energy_thresh):
    """Bump both Silero (.threshold) and FireRedVAD (.speech_threshold) so barge-in
    suppression during playback works regardless of which backend is active."""
    if hasattr(detector, 'threshold'):
        detector.threshold = speech_thresh
    if hasattr(detector, 'speech_threshold'):
        detector.speech_threshold = speech_thresh
    if hasattr(detector, 'energy_threshold'):
        detector.energy_threshold = energy_thresh

@socketio.on('playback_started')
def handle_playback_started():
    sid = request.sid
    playback_active[sid] = True
    # During playback: enable FAST barge-in detection mode.
    # Browser echoCancellation handles the TTS→mic feedback, so we don't
    # need to raise thresholds — we just want to detect a user interrupt
    # within ~50ms instead of waiting 150ms for sustained speech.
    det = vad_detectors.get(sid)
    if det is not None and hasattr(det, 'set_barge_in_mode'):
        det.set_barge_in_mode(True)
    print(f"[VAD] Playback started for {sid} — fast barge-in mode (mode={vad_modes.get(sid)})")

@socketio.on('playback_ended')
def handle_playback_ended():
    sid = request.sid
    playback_active[sid] = False
    det = vad_detectors.get(sid)
    if det is not None and hasattr(det, 'set_barge_in_mode'):
        det.set_barge_in_mode(False)
    print(f"[VAD] Playback ended for {sid} — normal mode restored")

SUPPORTED_VAD_MODES = ('silero', 'firered')

@socketio.on('set_vad_mode')
def handle_set_vad_mode(data):
    """Switch the per-session VAD backend. Supported: 'silero', 'firered'."""
    sid = request.sid
    requested = (data or {}).get('mode', 'silero')
    if requested not in SUPPORTED_VAD_MODES:
        emit('error', {'message': f'Unsupported VAD mode: {requested}'})
        return
    with buffer_locks[sid]:
        vad_modes[sid] = requested
        if sid in vad_detectors:
            del vad_detectors[sid]
        try:
            _ = get_vad_detector(sid)
            emit('vad_mode_changed', {'mode': requested, 'message': f'Using {requested} VAD'})
        except Exception as e:
            emit('error', {'message': f'Error loading {requested} VAD: {str(e)}'})

@socketio.on('start_stream')
def handle_start_stream():
    sid = request.sid
    with buffer_locks[sid]:
        audio_buffers[sid] = []
        utterance_audio_full[sid] = []
        utterance_active[sid] = False
        accumulated_text[sid] = ""
        silence_counters[sid] = 0
        last_partial_text[sid] = ""
        # Initialize VAD detector for this session
        vad_detector = get_vad_detector(sid)
        vad_detector.reset()
    emit('stream_started', {
        'message': 'Stream initialized',
        'vad_mode': 'silero'
    })

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    sid = request.sid
    try:
        audio_blob = data['audio']
        sample_rate = data.get('sampleRate', 48000)

        audio_array = np.frombuffer(audio_blob, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0

        if sample_rate != 16000:
            audio_float = _resample_linear(audio_float, sample_rate, 16000)

        # High-pass filter at ~80Hz: cuts HVAC rumble, mic handling thumps,
        # AC power hum — frequencies VAD/ASR get no signal from but that
        # inflate RMS and can trigger false speech detection.
        audio_float = _highpass_80hz(audio_float)

        # Decide action and copy audio out with minimal locking
        action = 'none'
        is_final = False
        audio_to_process = None
        chunk_has_speech = False

        with buffer_locks[sid]:
            # Get VAD detector for this session
            vad_detector = get_vad_detector(sid)

            # Always add to buffer first
            audio_buffers[sid].append(audio_float)
            combined_audio = np.concatenate(audio_buffers[sid]) if audio_buffers[sid] else np.array([], dtype=np.float32)

            # Check if current chunk has speech
            chunk_has_speech = vad_detector.has_speech(audio_float)

            # Manage full-utterance buffer lifecycle
            if chunk_has_speech and not utterance_active[sid]:
                # Start a new utterance
                utterance_active[sid] = True
                utterance_audio_full[sid] = []
            if utterance_active[sid]:
                # Keep collecting audio (including intervening short silences)
                utterance_audio_full[sid].append(audio_float)

            # Barge-in: if user starts speaking while assistant is active, cancel current generation
            if chunk_has_speech and generation_active.get(sid, False):
                # Increment generation id to cancel any in-flight streams
                current_generation_id[sid] += 1
                generation_active[sid] = False
                print(f"[BARGE-IN] Cancelling assistant for sid={sid}, new gen_id={current_generation_id[sid]}")
                socketio.emit('assistant_cancel', {'reason': 'barge_in'}, room=sid)

            # STAGE 1: VAD Detection
            if chunk_has_speech:
                print(f"[STAGE 1 - VAD] Speech detected in chunk (sid: {sid})")
                socketio.emit('pipeline_stage', {
                    'stage': 'vad',
                    'message': 'Speech detected'
                }, room=sid)

            # Update silence counter (Silero only)
            if chunk_has_speech:
                silence_counters[sid] = 0
            else:
                silence_counters[sid] += 1

            # Minimum buffer size check
            if len(combined_audio) >= MIN_BUFFER_SIZE:
                # Check if we have speech in the full buffer
                buffer_has_speech = vad_detector.has_speech(combined_audio)

                # If no speech detected in buffer, clear it and wait
                if not buffer_has_speech:
                    # Use detector's configured silence_threshold
                    if silence_counters[sid] > getattr(vad_detector, 'silence_threshold', 8):
                        audio_buffers[sid] = []
                        silence_counters[sid] = 0
                        last_partial_text[sid] = ""
                else:
                    # STREAMING MODE: Transcribe while speaking (every 2 seconds)
                    should_stream_partial = (
                        chunk_has_speech and
                        len(combined_audio) >= STREAMING_CHUNK_SIZE
                    )

                    # FINAL MODE: User stopped speaking
                    is_turn_complete = vad_detector.is_turn_complete(audio_float)
                    should_finalize = (
                        is_turn_complete and
                        len(combined_audio) >= MIN_BUFFER_SIZE
                    )

                    if should_stream_partial or should_finalize:
                        is_final = should_finalize
                        action = 'final' if is_final else 'partial'
                        # Copy audio out for processing without holding the lock
                        # For partials, use rolling window; for final, use the full utterance buffer
                        if is_final and utterance_audio_full[sid]:
                            try:
                                audio_to_process = np.concatenate(utterance_audio_full[sid]).copy()
                            except Exception:
                                audio_to_process = combined_audio.copy()
                        else:
                            audio_to_process = combined_audio.copy()
                        # For partials, keep overlap to improve continuity
                        if not is_final:
                            if len(combined_audio) > OVERLAP_SAMPLES:
                                audio_buffers[sid] = [combined_audio[-OVERLAP_SAMPLES:]]
                        else:
                            # For final, reset buffer and partial text for next turn
                            audio_buffers[sid] = []
                            last_partial_text[sid] = ""
                            utterance_active[sid] = False
                            utterance_audio_full[sid] = []

        # If nothing to do, return quickly
        if action == 'none' or audio_to_process is None:
            return

        # Write temp wav outside the lock
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                pcm16 = np.clip(audio_to_process, -1.0, 1.0)
                pcm16 = (pcm16 * 32767.0).astype(np.int16)
                wf.writeframes(pcm16.tobytes())

        try:
            # STAGE 2: Transcription
            print(f"[STAGE 2 - TRANSCRIPTION] Starting transcription (is_final={is_final}, sid={sid})")
            socketio.emit('pipeline_stage', {
                'stage': 'transcription_start',
                'message': 'Transcribing audio...',
                'is_final': is_final
            }, room=sid)
            socketio.sleep(0)

            # ── Session logger: start turn + ASR timing (final pass only) ──
            session_log = None
            if is_final:
                session_log = get_session_logger(sid)
                session_log.start_turn(audio_to_process, user_sample_rate=16000)
                session_log.mark('asr_start')

            result = generate_transcription(model=transcription_model, audio=tmp_path, format="txt", verbose=False)
            text = result.text

            if session_log is not None:
                session_log.mark('asr_end')

            if text and text.strip():
                print(f"[STAGE 2 - TRANSCRIPTION] Transcribed text: '{text}' (is_final={is_final})")
                socketio.emit('pipeline_stage', {
                    'stage': 'transcription_complete',
                    'message': f'Transcribed: {text[:50]}...',
                    'is_final': is_final
                }, room=sid)
                socketio.sleep(0)

                if is_final:
                    if session_log is not None:
                        session_log.set_user_text(text)

                    # Add user message to conversation history
                    try:
                        conversation_history[sid].append({"role": "user", "content": text})
                    except Exception:
                        pass

                    # Send final user message to frontend IMMEDIATELY
                    print(f"[STAGE 3 - USER MESSAGE] Displaying user message in UI")
                    emit('user_message', {'text': text, 'is_final': True})
                    socketio.sleep(0)

                    # STAGE 3: LLM Processing
                    print(f"[STAGE 4 - LLM] Starting LLM processing")
                    socketio.emit('pipeline_stage', {
                        'stage': 'llm_start',
                        'message': 'Processing with LLM...'
                    }, room=sid)
                    socketio.sleep(0)

                    # Signal start of assistant response
                    emit('assistant_start')
                    # Mark generation active and assign new generation id
                    current_generation_id[sid] += 1
                    gen_id = current_generation_id[sid]
                    generation_active[sid] = True
                    socketio.sleep(0)

                    # Offload LLM+TTS to a background task for live streaming
                    socketio.start_background_task(_run_llm_and_tts_for_sid, sid, gen_id)
                else:
                    # PARTIAL transcription - only emit if different from last partial
                    if text != last_partial_text[sid]:
                        emit('partial_transcription', {'text': text, 'is_final': False})
                        last_partial_text[sid] = text
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        # Log exceptions for visibility instead of swallowing silently
        print(f"Error in audio_chunk handler: {e}")
        import traceback
        traceback.print_exc()

@socketio.on('stop_stream')
def handle_stop_stream():
    sid = request.sid
    try:
        with buffer_locks[sid]:
            vad_detector = get_vad_detector(sid)

            # Prefer full utterance buffer if available
            candidate_audio = None
            if utterance_active.get(sid, False) and utterance_audio_full.get(sid):
                try:
                    candidate_audio = np.concatenate(utterance_audio_full[sid])
                except Exception:
                    candidate_audio = None
            if candidate_audio is None and sid in audio_buffers and audio_buffers[sid]:
                combined_audio = np.concatenate(audio_buffers[sid])
                if vad_detector.has_speech(combined_audio):
                    candidate_audio = combined_audio

            if candidate_audio is not None and candidate_audio.size > 0:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    with wave.open(tmp_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        pcm16 = np.clip(candidate_audio, -1.0, 1.0)
                        pcm16 = (pcm16 * 32767.0).astype(np.int16)
                        wf.writeframes(pcm16.tobytes())

                    try:
                        result = generate_transcription(model=transcription_model, audio=tmp_path, format="txt", verbose=False)
                        text = result.text

                        if text and text.strip():
                            if accumulated_text[sid]:
                                accumulated_text[sid] += " " + text
                            else:
                                accumulated_text[sid] = text

                    finally:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass

            final_text = accumulated_text.get(sid, "")
            emit('final_transcription', {'text': final_text, 'final': True})

            # Cancel any active assistant generation (stop audio & text streaming)
            current_generation_id[sid] += 1
            generation_active[sid] = False
            socketio.emit('assistant_cancel', {'reason': 'stop'}, room=sid)

            audio_buffers[sid] = []
            utterance_audio_full[sid] = []
            utterance_active[sid] = False
            accumulated_text[sid] = ""

    except Exception as e:
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3003))
    print(f"Starting server on port {port}...", flush=True)
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
