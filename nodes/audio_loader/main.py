import math
import os
from pathlib import Path

import soundfile as sf


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


def _empty_channel_stats(channels: int) -> list[dict]:
    return [{"peak": 0.0, "sum_squares": 0.0, "samples": 0} for _ in range(channels)]


def _update_channel_stats(stats: list[dict], block) -> None:
    for channel_index in range(block.shape[1]):
        channel = block[:, channel_index]
        current = stats[channel_index]
        current["peak"] = max(current["peak"], float(abs(channel).max(initial=0.0)))
        current["sum_squares"] += float((channel * channel).sum())
        current["samples"] += int(channel.shape[0])


def _channel_summary(stats: list[dict]) -> list[dict]:
    result = []
    for index, item in enumerate(stats):
        samples = max(1, int(item["samples"]))
        rms = math.sqrt(float(item["sum_squares"]) / samples)
        result.append(
            {
                "channel": index + 1,
                "peak_amplitude": _safe_float(float(item["peak"]), 6),
                "rms_amplitude": _safe_float(rms, 6),
                "rms_db": _safe_float(20.0 * math.log10(max(rms, 1e-12)), 2),
            }
        )
    return result


def _audio_info(path: str):
    try:
        return sf.info(path)
    except Exception as exc:
        raise ValueError(f"Audio file is not readable by libsndfile: {exc}") from exc


def _metadata(path: str, info, channel_stats: list[dict]) -> dict:
    file_path = Path(path)
    file_size = file_path.stat().st_size
    sample_rate = int(info.samplerate)
    frames = int(info.frames)
    duration = frames / float(sample_rate) if sample_rate > 0 else 0.0

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
        "channels": int(info.channels),
        "samples": frames,
        "duration_seconds": _safe_float(duration, 4),
        "estimated_bitrate_bps": bitrate,
        "encoding": encoding,
        "subtype": subtype_info,
        "soundfile_supported": True,
    }
    metadata["channel_summary"] = _channel_summary(channel_stats)
    return metadata


def _output_subtype(source, output_format: str) -> str:
    if source.format.upper() == output_format.upper() and source.subtype:
        return source.subtype
    return "PCM_16"


def _copy_audio_stream(source_path: str, output_path: str, output_format: str, block_size: int = 65536) -> tuple[object, list[dict]]:
    info = _audio_info(source_path)
    if info.frames <= 0:
        raise ValueError("Audio file contains no samples.")

    stats = _empty_channel_stats(int(info.channels))
    with sf.SoundFile(source_path, mode="r") as source:
        with sf.SoundFile(
            output_path,
            mode="w",
            samplerate=source.samplerate,
            channels=source.channels,
            format=output_format.upper(),
            subtype=_output_subtype(source, output_format),
        ) as target:
            while True:
                block = source.read(block_size, dtype="float32", always_2d=True)
                if block.size == 0:
                    break
                _update_channel_stats(stats, block)
                target.write(block)
    return info, stats


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
    output_path = context.output_path(f"waveform.{output_format}", port_id="waveform")
    info, channel_stats = _copy_audio_stream(source_path, output_path, output_format)

    metadata = _metadata(source_path, info, channel_stats)
    report = _report(metadata, preview_seconds)

    context.emit_metric("sample_rate", metadata["sample_rate"], port_id="sample_rate")
    context.emit_metric("duration_seconds", metadata["duration_seconds"], port_id="sample_rate")
    context.emit_metric("channels", metadata["channels"], port_id="sample_rate")

    return {
        "waveform": output_path,
        "sample_rate": {
            "sample_rate": metadata["sample_rate"],
            "duration_seconds": metadata["duration_seconds"],
            "channels": metadata["channels"],
        },
        "metadata": metadata,
        "audio_report": report,
    }
