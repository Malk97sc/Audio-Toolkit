import math
from pathlib import Path

import numpy as np
import soundfile as sf


def _input_path(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("file") or str(value)
    return str(value)


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    return data.astype(np.float32, copy=False), int(sample_rate)


def _save_audio(path: str, waveform: np.ndarray, sample_rate: int) -> None:
    sf.write(path, waveform, sample_rate)


def _resample_linear(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or data.size == 0:
        return data.astype(np.float32, copy=False)
    duration = data.shape[0] / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, duration, num=data.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_size, endpoint=False)
    channels = [
        np.interp(target_positions, source_positions, data[:, channel])
        for channel in range(data.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    if waveform.shape[1] == 1:
        return waveform
    return waveform.mean(axis=1, keepdims=True, dtype=np.float32)


def _to_stereo(waveform: np.ndarray) -> np.ndarray:
    if waveform.shape[1] == 1:
        return np.repeat(waveform, 2, axis=1)
    if waveform.shape[1] > 2:
        return waveform[:, :2]
    return waveform


def _trim_silence(waveform: np.ndarray, sample_rate: int, threshold_db: float) -> tuple[np.ndarray, dict]:
    mono = np.abs(_to_mono(waveform)[:, 0])
    if mono.size == 0:
        return waveform, {"trimmed": False, "samples_removed": 0, "seconds_removed": 0.0}
    peak = float(np.max(mono))
    if peak <= 0:
        return waveform, {"trimmed": False, "samples_removed": 0, "seconds_removed": 0.0}

    threshold = peak * (10.0 ** (threshold_db / 20.0))
    active = np.flatnonzero(mono > threshold)
    if active.size == 0:
        return waveform, {"trimmed": False, "samples_removed": 0, "seconds_removed": 0.0}

    margin = int(sample_rate * 0.025)
    start = max(0, int(active[0]) - margin)
    end = min(waveform.shape[0], int(active[-1]) + 1 + margin)
    trimmed = waveform[start:end]
    removed = waveform.shape[0] - trimmed.shape[0]
    return trimmed, {
        "trimmed": removed > 0,
        "start_sample": start,
        "end_sample": end,
        "samples_removed": int(removed),
        "seconds_removed": round(removed / sample_rate, 4),
    }


def _stats(waveform: np.ndarray, sample_rate: int) -> dict:
    flat = waveform.reshape(-1).astype(np.float32, copy=False)
    duration = waveform.shape[0] / float(sample_rate)
    peak = float(np.max(np.abs(flat))) if flat.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(flat, dtype=np.float64)))) if flat.size else 0.0
    clipped = int(np.sum(np.abs(flat) >= 0.999)) if flat.size else 0
    silence_ratio = float(np.mean(np.abs(flat) < 1e-4)) if flat.size else 1.0
    return {
        "sample_rate": sample_rate,
        "channels": int(waveform.shape[1]),
        "samples": int(waveform.shape[0]),
        "duration_seconds": round(duration, 4),
        "peak_amplitude": round(peak, 6),
        "rms_amplitude": round(rms, 6),
        "rms_db": round(20.0 * math.log10(max(rms, 1e-12)), 2),
        "clipped_samples": clipped,
        "clipping_ratio": round(clipped / max(1, flat.size), 6),
        "silence_ratio": round(silence_ratio, 4),
    }


def _apply_rms_gain(waveform: np.ndarray, target_db: float) -> tuple[np.ndarray, float]:
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    if rms <= 0:
        return waveform, 0.0
    current_db = 20.0 * math.log10(max(rms, 1e-12))
    gain_db = target_db - current_db
    gain = 10.0 ** (gain_db / 20.0)
    return np.clip(waveform * gain, -1.0, 1.0).astype(np.float32), gain_db


def run(inputs: dict, params: dict, context) -> dict:
    audio_value = inputs.get("audio")
    if audio_value is None:
        raise ValueError("Audio Processor requires an Audio input.")

    source_path = _input_path(audio_value)
    waveform, original_rate = _load_audio(source_path)
    if waveform.size == 0:
        raise ValueError("Audio input contains no samples.")

    target_rate = int(params.get("target_sample_rate", 16000))
    if target_rate < 4000 or target_rate > 192000:
        raise ValueError("Target Sample Rate must be between 4000 and 192000 Hz.")

    channels = str(params.get("channels", "mono")).lower()
    if channels not in {"mono", "stereo", "keep"}:
        raise ValueError("Channels must be 'mono', 'stereo', or 'keep'.")

    output_format = str(params.get("output_format", "wav")).lower()
    if output_format not in {"wav", "flac"}:
        raise ValueError("Output Format must be 'wav' or 'flac'.")

    original_stats = _stats(waveform, original_rate)
    operations = []

    if original_rate != target_rate:
        context.log(f"Resampling from {original_rate} Hz to {target_rate} Hz")
        waveform = _resample_linear(waveform, original_rate, target_rate)
        operations.append({"operation": "resample", "from": original_rate, "to": target_rate})
    current_rate = target_rate

    before_channels = waveform.shape[1]
    if channels == "mono":
        waveform = _to_mono(waveform)
    elif channels == "stereo":
        waveform = _to_stereo(waveform)
    if waveform.shape[1] != before_channels:
        operations.append({"operation": "channel_conversion", "from": int(before_channels), "to": int(waveform.shape[1])})

    if bool(params.get("trim_silence", True)):
        threshold_db = float(params.get("silence_threshold_db", -45.0))
        waveform, trim_report = _trim_silence(waveform, current_rate, threshold_db)
        trim_report["threshold_db"] = threshold_db
        operations.append({"operation": "trim_silence", **trim_report})

    if bool(params.get("normalize_volume", False)):
        target_db = float(params.get("target_loudness_db", -20.0))
        waveform, gain_db = _apply_rms_gain(waveform, target_db)
        operations.append({"operation": "volume_normalization", "target_rms_db": target_db, "gain_db": round(gain_db, 2)})

    if bool(params.get("normalize_peak", True)):
        target_peak = float(params.get("target_peak", 0.95))
        if target_peak <= 0 or target_peak > 1:
            raise ValueError("Peak Target must be greater than 0 and no more than 1.")
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 0:
            waveform = (waveform / peak * target_peak).astype(np.float32)
        operations.append({"operation": "peak_normalization", "target_peak": target_peak, "previous_peak": round(peak, 6)})

    processed_stats = _stats(waveform, current_rate)
    output_path = context.output_path(f"processed_waveform.{output_format}", port_id="processed_waveform")
    _save_audio(output_path, waveform, current_rate)

    report = {
        "title": "Audio Processing Report",
        "source_file": Path(source_path).name,
        "original": original_stats,
        "processed": processed_stats,
        "operations": operations,
        "quality_flags": {
            "clipping_detected": processed_stats["clipped_samples"] > 0,
            "mostly_silent": processed_stats["silence_ratio"] > 0.85,
            "very_short": processed_stats["duration_seconds"] < 1.0,
        },
        "next_step": "Connect Processed Waveform to Feature Extractor or Spectrogram Viewer.",
    }

    for key in ["duration_seconds", "sample_rate", "channels", "peak_amplitude", "rms_db", "clipping_ratio", "silence_ratio"]:
        context.emit_metric(key, processed_stats[key], port_id="audio_statistics")

    return {
        "processed_waveform": output_path,
        "processing_report": report,
        "audio_statistics": processed_stats,
    }
