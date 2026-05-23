#!/usr/bin/env python3
"""
Voice Activity Detection module with support for multiple VAD methods:
- Silero VAD (energy-based + ML)
- Smart Turn v3 (semantic turn-completion detection)
"""

import numpy as np
import torch
from abc import ABC, abstractmethod


class VADDetector(ABC):
    """Abstract base class for VAD detectors"""

    @abstractmethod
    def has_speech(self, audio_data):
        """
        Detect if audio contains speech

        Args:
            audio_data: numpy array of audio samples (float32, -1 to 1)

        Returns:
            bool: True if speech is detected
        """
        pass

    @abstractmethod
    def is_turn_complete(self, audio_data):
        """
        Detect if speaker has completed their turn

        Args:
            audio_data: numpy array of audio samples (float32, -1 to 1)

        Returns:
            bool: True if turn is complete
        """
        pass


class SileroVAD(VADDetector):
    """Silero VAD implementation - energy-based speech detection"""

    def __init__(self, threshold=0.5, silence_threshold=8, min_speech_duration_ms=250,
                 min_silence_duration_ms=100, energy_threshold=0.001):
        """
        Initialize Silero VAD

        Args:
            threshold: confidence threshold for speech detection (0.0-1.0)
            silence_threshold: number of consecutive silent chunks before turn end (default: 8)
            min_speech_duration_ms: minimum speech duration to detect (default: 250ms)
            min_silence_duration_ms: minimum silence duration to split (default: 100ms)
            energy_threshold: RMS energy threshold for quick detection (default: 0.001)
        """
        print("Loading Silero VAD model...")
        self.vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        self.get_speech_timestamps = vad_utils[0]
        self.threshold = threshold
        self.silence_counter = 0
        self.silence_threshold = silence_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.energy_threshold = energy_threshold
        print(f"Silero VAD loaded successfully! (silence_threshold={silence_threshold}, threshold={threshold})")

    def has_speech(self, audio_data):
        """Detect if audio contains speech using Silero VAD"""
        if len(audio_data) == 0:
            return False

        # Quick energy check to avoid unnecessary VAD processing
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < self.energy_threshold:
            return False

        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_data).float()

            # Get speech timestamps from Silero VAD
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=16000,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                return_seconds=False
            )

            return len(speech_timestamps) > 0

        except Exception as e:
            print(f"Silero VAD error: {e}")
            # Fallback to energy-based detection
            return rms > 0.01

    def is_turn_complete(self, audio_data):
        """
        Detect turn completion using silence counting

        This is the original logic from app.py - counts consecutive
        silent chunks and considers turn complete after threshold.
        """
        has_speech_now = self.has_speech(audio_data)

        if has_speech_now:
            self.silence_counter = 0
            return False
        else:
            self.silence_counter += 1
            return self.silence_counter >= self.silence_threshold

    def reset(self):
        """Reset silence counter"""
        self.silence_counter = 0


class SmartTurnV3(VADDetector):
    """Smart Turn v3 - semantic turn-completion detection"""

    def __init__(self, threshold=0.5, energy_threshold=0.001, responsiveness=1.0):
        """
        Initialize Smart Turn v3

        Args:
            threshold: probability threshold for turn completion (0.0-1.0, default: 0.5)
                      Lower = more aggressive (interrupts sooner)
                      Higher = more patient (waits longer)
            energy_threshold: RMS energy threshold for quick detection (default: 0.001)
            responsiveness: multiplier for turn detection (0.5-2.0, default: 1.0)
                           Lower = more patient, waits for clearer turn completion
                           Higher = more aggressive, responds faster to pauses
        """
        print("Loading Smart Turn v3 model...")
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            import librosa

            # Download ONNX model from Hugging Face
            model_path = hf_hub_download(
                repo_id="pipecat-ai/smart-turn-v3",
                filename="smart-turn-v3.0.onnx"
            )

            # Load ONNX model
            self.session = ort.InferenceSession(model_path)
            self.threshold = threshold
            self.energy_threshold = energy_threshold
            self.responsiveness = responsiveness

            print(f"Smart Turn v3 loaded successfully! (threshold={threshold}, responsiveness={responsiveness})")

        except ImportError as e:
            print(f"Error loading Smart Turn v3: Missing dependencies")
            print("Install with: pip install onnxruntime huggingface-hub librosa")
            print("Falling back to Silero VAD")
            raise
        except Exception as e:
            print(f"Error loading Smart Turn v3: {e}")
            print("Falling back to Silero VAD")
            raise

    def has_speech(self, audio_data):
        """Quick energy-based speech detection"""
        if len(audio_data) == 0:
            return False

        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > self.energy_threshold

    def is_turn_complete(self, audio_data):
        """
        Detect turn completion using Smart Turn v3 semantic analysis

        Returns True when the model predicts the speaker has finished their turn
        """
        if len(audio_data) == 0:
            return False

        # Quick energy check first
        if not self.has_speech(audio_data):
            return True  # No speech = turn complete

        try:
            import librosa

            # Smart Turn v3 expects exactly 8 seconds of audio
            # With hop_length=160, we need exactly 800 frames
            # 800 frames * 160 hop = 128,000 samples, but we need to account for padding
            # Formula: n_frames = 1 + (n_samples - n_fft) / hop_length
            # Solving for n_samples: n_samples = (n_frames - 1) * hop_length + n_fft
            # n_samples = (800 - 1) * 160 + 400 = 127,840 + 400 = 128,240
            target_samples = 127840  # This produces exactly 800 frames

            # Pad or trim to exact length
            if len(audio_data) < target_samples:
                # Pad with zeros
                audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
            elif len(audio_data) > target_samples:
                # Take last portion
                audio_data = audio_data[-target_samples:]

            # Convert to mel spectrogram (80 mel bins, 800 time steps)
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=16000,
                n_mels=80,
                n_fft=400,
                hop_length=160,
                center=False  # Disable padding to get exact frame count
            )

            # Ensure exactly 800 frames
            if mel_spec.shape[1] > 800:
                mel_spec = mel_spec[:, :800]
            elif mel_spec.shape[1] < 800:
                # Pad if needed
                pad_width = 800 - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')

            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

            # Model expects (batch, mel_bins, time_steps) = (1, 80, 800)
            input_tensor = log_mel[np.newaxis, :, :].astype(np.float32)

            # Run ONNX inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name

            outputs = self.session.run([output_name], {input_name: input_tensor})
            logits = outputs[0]

            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # The model outputs a single probability value
            # Higher value = turn is complete
            # Check the shape and extract appropriately
            if probabilities.shape[-1] == 1:
                # Single probability output
                turn_complete_prob = probabilities[0][0]
            elif probabilities.shape[-1] == 2:
                # Binary classification (turn incomplete vs complete)
                turn_complete_prob = probabilities[0][1]
            else:
                # Fallback: use the maximum probability
                turn_complete_prob = np.max(probabilities)

            # Apply responsiveness multiplier
            # Higher responsiveness = lower effective threshold
            effective_threshold = self.threshold / self.responsiveness

            return turn_complete_prob >= effective_threshold

        except Exception as e:
            print(f"Smart Turn v3 error: {e}")
            # Fallback to energy-based detection
            return not self.has_speech(audio_data)

    def reset(self):
        """Reset (no internal state for Smart Turn v3)"""
        pass


class FireRedVAD(VADDetector):
    """FireRedVAD (DFSMN-based, FireRedTeam) — non-streaming speech detection.

    Uses FireRedVad (non-streaming) rather than FireRedStreamVad because our
    audio pipeline calls has_speech() on a rolling buffer of overlapping chunks.
    The streaming variant maintains DFSMN model_caches + AudioFeat state across
    calls, which gets scrambled by overlapping input; the non-streaming variant
    resets state per call (correct for chunk-based detection).
    """

    def __init__(self, speech_threshold=0.4, silence_threshold=8, energy_threshold=0.001,
                 min_speech_frame=8, model_dir=None, use_gpu=False):
        import os
        import sys

        repo_root = os.path.dirname(os.path.abspath(__file__))
        vendor_path = os.path.join(repo_root, "vendor", "FireRedVAD")
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

        from fireredvad import FireRedVad, FireRedVadConfig

        if model_dir is None:
            model_dir = os.path.join(repo_root, "pretrained_models", "FireRedVAD", "VAD")

        print(f"Loading FireRedVAD (non-streaming) from {model_dir}...")
        config = FireRedVadConfig(
            use_gpu=use_gpu,
            smooth_window_size=5,
            speech_threshold=speech_threshold,
            # min_speech_frame lowered from default 20 → 8 so short chunks
            # (~250ms) can still register as speech.
            min_speech_frame=min_speech_frame,
            max_speech_frame=2000,
            min_silence_frame=10,
            merge_silence_frame=0,
            extend_speech_frame=0,
            chunk_max_frame=30000,
        )
        self.vad = FireRedVad.from_pretrained(model_dir, config)
        self.silence_counter = 0
        self.silence_threshold = silence_threshold
        self.energy_threshold = energy_threshold
        self.speech_threshold = speech_threshold
        print(f"FireRedVAD loaded! (silence_threshold={silence_threshold}, speech_threshold={speech_threshold})")

    def has_speech(self, audio_data):
        if len(audio_data) == 0:
            return False
        # Quick RMS gate to skip inference on near-silent buffers.
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        if rms < self.energy_threshold:
            return False
        try:
            # FireRedVAD's AudioFeat reads files with dtype="int16", so when we
            # bypass file I/O and pass a numpy array directly, we MUST hand it
            # int16-scale samples — not the float32 [-1, 1] our pipeline uses.
            audio_int16 = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
            result, _probs = self.vad.detect(audio_int16)
            # result["timestamps"] is a list of (start_sec, end_sec) speech segments.
            return len(result.get("timestamps", [])) > 0
        except Exception as e:
            print(f"FireRedVAD error: {e}")
            return rms > 0.01

    def is_turn_complete(self, audio_data):
        if self.has_speech(audio_data):
            self.silence_counter = 0
            return False
        self.silence_counter += 1
        return self.silence_counter >= self.silence_threshold

    def reset(self):
        self.silence_counter = 0

    def set_barge_in_mode(self, active):
        """Tighten timing for fast barge-in detection during TTS playback.

        During TTS playback we want to detect a user interrupt FAST. Browser
        echoCancellation suppresses TTS through speakers → mic, so the risk of
        false-positive from echo is low. Drop min_speech_frame so ~50ms of
        sustained speech is enough to fire (vs 150ms normally).
        """
        try:
            if active:
                self.vad.vad_postprocessor.min_speech_frame = 4   # ~40ms
            else:
                self.vad.vad_postprocessor.min_speech_frame = 15  # ~150ms
        except Exception:
            pass


def create_vad_detector(vad_type="silero", threshold=0.5, **kwargs):
    """
    Factory function to create VAD detector

    Args:
        vad_type: "silero" or "smart_turn_v3"
        threshold: detection threshold (0.0-1.0)
        **kwargs: Additional parameters for specific VAD types

            For Silero VAD:
                - silence_threshold: consecutive silent chunks before turn end (default: 8)
                - min_speech_duration_ms: minimum speech duration (default: 250ms)
                - min_silence_duration_ms: minimum silence duration (default: 100ms)
                - energy_threshold: RMS energy threshold (default: 0.001)

            For Smart Turn v3:
                - energy_threshold: RMS energy threshold (default: 0.001)
                - responsiveness: turn detection multiplier 0.5-2.0 (default: 1.0)

    Returns:
        VADDetector instance
    """
    if vad_type == "smart_turn_v3":
        return SmartTurnV3(threshold=threshold, **kwargs)
    elif vad_type == "firered":
        return FireRedVAD(**kwargs)
    else:
        return SileroVAD(threshold=threshold, **kwargs)
