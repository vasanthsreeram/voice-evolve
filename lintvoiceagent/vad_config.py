#!/usr/bin/env python3
"""
VAD Configuration
Adjust these settings to control voice activity detection behavior
"""

# ============================================================================
# SILERO VAD SETTINGS
# ============================================================================

SILERO_CONFIG = {
    # Speech detection threshold (0.0-1.0)
    # Lower = more sensitive (detects quieter speech)
    # Higher = less sensitive (only detects clear speech)
    'threshold': 0.65,

    # Number of consecutive silent chunks before considering turn complete
    # Each chunk is ~256ms (4096 samples at 16kHz)
    # 10 chunks = ~2.5 seconds - responsive but patient enough for natural pauses
    # Lower = faster response but may cut off speech
    # Higher = more patient but slower response
    'silence_threshold': 10,

    # Minimum speech duration to detect (milliseconds)
    # Lower = detects very short utterances
    # Higher = ignores short sounds/noise — 400ms filters brief background pops/sounds
    'min_speech_duration_ms': 400,

    # Minimum silence duration to split speech segments (milliseconds)
    'min_silence_duration_ms': 120,

    # RMS energy threshold for quick detection (pre-filter before Silero runs)
    # 0.015 filters out quiet background hum/AC noise without cutting off soft speech
    'energy_threshold': 0.015,
}

# ============================================================================
# SMART TURN V3 SETTINGS
# ============================================================================

SMART_TURN_CONFIG = {
    # Turn completion probability threshold (0.0-1.0)
    # Lower = interrupts sooner (more aggressive)
    # Higher = waits longer (more patient)
    # 0.3 = very responsive, may interrupt
    # 0.5 = balanced (default)
    # 0.7 = patient, waits for clear completion
    'threshold': 0.5,

    # RMS energy threshold for quick detection
    'energy_threshold': 0.001,

    # Responsiveness multiplier (0.5-2.0)
    # Controls how aggressively Smart Turn responds
    # 0.5 = very patient, waits for very clear turn completion
    # 1.0 = balanced (default)
    # 1.5 = more responsive
    # 2.0 = very aggressive, responds to slight pauses
    'responsiveness': 1.0,
}

# ============================================================================
# STREAMING & CHUNKING SETTINGS
# ============================================================================

# Show partial transcription every N seconds while speaking
# 2.0 = every 2 seconds (default)
# 1.0 = every 1 second (more frequent updates)
# 3.0 = every 3 seconds (less frequent)
STREAMING_INTERVAL_SECONDS = 2.0

# Minimum audio buffer before transcription (seconds)
# 1.5 = won't transcribe unless at least 1.5 seconds of audio
MIN_BUFFER_SECONDS = 1.5

# Audio overlap to keep between partial transcriptions (seconds)
# Prevents cutting words at chunk boundaries
# 1.0 = keep last 1 second (default)
OVERLAP_SECONDS = 1.0

# ============================================================================
# PRESETS
# ============================================================================

# Quick presets for common use cases
PRESETS = {
    'default': {
        'silero': SILERO_CONFIG.copy(),
        'smart_turn': SMART_TURN_CONFIG.copy(),
    },

    'very_responsive': {
        'silero': {
            **SILERO_CONFIG,
            'silence_threshold': 4,  # 1 second of silence
            'threshold': 0.55,
        },
        'smart_turn': {
            **SMART_TURN_CONFIG,
            'threshold': 0.3,
            'responsiveness': 1.5,
        },
    },

    'patient': {
        'silero': {
            **SILERO_CONFIG,
            'silence_threshold': 12,  # 3 seconds of silence
            'threshold': 0.6,
        },
        'smart_turn': {
            **SMART_TURN_CONFIG,
            'threshold': 0.7,
            'responsiveness': 0.7,
        },
    },

    'balanced': {
        'silero': {
            **SILERO_CONFIG,
            'silence_threshold': 8,  # 2 seconds of silence
            'threshold': 0.6,
        },
        'smart_turn': {
            **SMART_TURN_CONFIG,
            'threshold': 0.5,
            'responsiveness': 1.0,
        },
    },
}

# ============================================================================
# ACTIVE PRESET
# ============================================================================

# Change this to switch presets easily
# Options: 'default', 'very_responsive', 'patient', 'balanced'
ACTIVE_PRESET = 'default'

def get_vad_config(vad_type='silero', preset=None):
    """
    Get VAD configuration

    Args:
        vad_type: 'silero' or 'smart_turn_v3'
        preset: Optional preset name (overrides ACTIVE_PRESET)

    Returns:
        dict: Configuration parameters
    """
    preset_name = preset or ACTIVE_PRESET
    preset_config = PRESETS.get(preset_name, PRESETS['default'])

    if vad_type == 'smart_turn_v3':
        return preset_config['smart_turn']
    elif vad_type == 'firered':
        # FireRedVAD: tuned for noise rejection over sensitivity.
        return {
            'silence_threshold': preset_config['silero'].get('silence_threshold', 8),
            # 0.4 → 0.6: must be more confidently speech (fewer false positives
            # from keyboard, fan, distant talkers).
            'speech_threshold': 0.6,
            # 8 → 15 frames ≈ 150ms of sustained speech required, so short
            # clicks/blips/cough syllables don't trigger.
            'min_speech_frame': 15,
            # 0.001 → 0.005: quiet room noise gated out before VAD even runs.
            'energy_threshold': 0.005,
        }
    else:
        return preset_config['silero']
