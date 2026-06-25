import math
import os
from pathlib import Path

import soundfile as sf
import torch


def _input_path(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("file") or str(value)
    return str(value)


def _safe_float(value: float, digits: int = 4):
    if math.isfinite(value):
        return round(float(value), digits)
    return None


def _channel_summary(waveform: torch.Tensor) -> list[dict]:
    result = []
    for index, channel in enumerate(waveform.float()):
        peak = channel.abs().max().item() if channel.numel() else 0.0
        rms = torch.sqrt(torch.mean(channel.square())).item() if channel.numel() else 0.0
        result.append(
            {
                "channel": index + 1,
                "peak_amplitude": _safe_float(peak, 6),
                "rms_amplitude": _safe_float(rms, 6),
                "rms_db": _safe_float(20.0 * math.log10(max(rms, 1e-12)), 2),
            }
        )
    return result


def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data).transpose(0, 1).contiguous()
    return waveform, int(sample_rate)


def _save_audio(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    data = waveform.detach().cpu().transpose(0, 1).numpy()
    sf.write(path, data, sample_rate)


def _metadata(path: str, waveform: torch.Tensor, sample_rate: int) -> dict:
    file_path = Path(path)
    file_size = file_path.stat().st_size
    duration = waveform.shape[-1] / float(sample_rate)
    info = None
    try:
        info = sf.info(path)
    except Exception:
        info = None

    bitrate = None
    if duration > 0:
        bitrate = int(round((file_size * 8) / duration))

    subtype_info = getattr(info, "subtype_info", None) if info else None
    encoding = getattr(info, "format", None) if info else None
    metadata = {
        "source_path": str(file_path),
        "file_name": file_path.name,
        "format": file_path.suffix.lstrip(".").lower() or "unknown",
        "file_size_bytes": file_size,
        "sample_rate": sample_rate,
        "channels": int(waveform.shape[0]),
        "samples": int(waveform.shape[-1]),
        "duration_seconds": _safe_float(duration, 4),
        "estimated_bitrate_bps": bitrate,
        "encoding": encoding,
        "subtype": subtype_info,
        "soundfile_supported": True,
    }
    metadata["channel_summary"] = _channel_summary(waveform)
    return metadata


def _report(metadata: dict, preview_seconds: float) -> dict:
    bitrate_kbps = None
    if metadata.get("estimated_bitrate_bps"):
        bitrate_kbps = round(metadata["estimated_bitrate_bps"] / 1000, 1)
    return {
        "title": "Audio Load Report",
        "summary": (
            f"Loaded {metadata['file_name']} with {metadata['channels']} channel(s), "
            f"{metadata['sample_rate']} Hz sample rate, and "
            f"{metadata['duration_seconds']} seconds duration."
        ),
        "duration_seconds": metadata["duration_seconds"],
        "channels": metadata["channels"],
        "sample_rate": metadata["sample_rate"],
        "bitrate_kbps": bitrate_kbps,
        "file_size_mb": round(metadata["file_size_bytes"] / (1024 * 1024), 3),
        "preview_window_seconds": preview_seconds,
        "notes": [
            "The waveform output is a workflow-ready audio copy for downstream Audio Toolkit nodes.",
            "Use Audio Processor next when sample-rate, channel, loudness, or silence cleanup is needed.",
        ],
    }


def run(inputs: dict, params: dict, context) -> dict:
    audio_value = inputs.get("audio_file")
    if audio_value is None:
        raise ValueError("Audio Loader requires an Audio File input.")

    source_path = _input_path(audio_value)
    if not os.path.exists(source_path):
        raise ValueError(f"Audio file does not exist: {source_path}")

    output_format = str(params.get("output_format", "wav")).lower()
    if output_format not in {"wav", "flac"}:
        raise ValueError("Working Audio Format must be 'wav' or 'flac'.")

    preview_seconds = float(params.get("preview_seconds", 3.0))
    if preview_seconds <= 0:
        raise ValueError("Preview Seconds must be greater than zero.")

    context.log(f"Loading audio from {source_path}")
    waveform, sample_rate = _load_audio(source_path)
    if waveform.numel() == 0:
        raise ValueError("Audio file contains no samples.")

    metadata = _metadata(source_path, waveform, sample_rate)
    report = _report(metadata, preview_seconds)

    output_path = context.output_path(f"waveform.{output_format}", port_id="waveform")
    _save_audio(output_path, waveform, sample_rate)

    context.emit_metric("sample_rate", sample_rate, port_id="sample_rate")
    context.emit_metric("duration_seconds", metadata["duration_seconds"], port_id="sample_rate")
    context.emit_metric("channels", metadata["channels"], port_id="sample_rate")

    return {
        "waveform": output_path,
        "sample_rate": {
            "sample_rate": sample_rate,
            "duration_seconds": metadata["duration_seconds"],
            "channels": metadata["channels"],
        },
        "metadata": metadata,
        "audio_report": report,
    }
