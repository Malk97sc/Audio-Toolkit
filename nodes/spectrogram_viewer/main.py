from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def _input_path(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("file") or str(value)
    return str(value)


def _mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.float32, copy=False)
    return data.mean(axis=1, dtype=np.float32)


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.size / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _frame_signal(samples: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))
    frame_count = 1 + max(0, (samples.size - n_fft) // hop_length)
    shape = (frame_count, n_fft)
    strides = (samples.strides[0] * hop_length, samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides).copy()


def _power_spectrogram(samples: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    frames = _frame_signal(samples, n_fft, hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=1)
    return (np.abs(spectrum) ** 2).T.astype(np.float32)


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    frequencies = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    mel_points = np.linspace(_hz_to_mel(0), _hz_to_mel(sample_rate / 2), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    filters = np.zeros((n_mels, frequencies.size), dtype=np.float32)

    for index in range(n_mels):
        left, center, right = hz_points[index:index + 3]
        rising = (frequencies - left) / max(center - left, 1e-12)
        falling = (right - frequencies) / max(right - center, 1e-12)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
    return filters


def _amplitude_to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def _theme(name: str) -> dict:
    if name == "dark":
        return {"face": "#111827", "text": "#f8fafc", "line": "#38bdf8", "cmap": "magma"}
    if name == "print":
        return {"face": "#ffffff", "text": "#111827", "line": "#111827", "cmap": "gray_r"}
    return {"face": "#ffffff", "text": "#172554", "line": "#2563eb", "cmap": "viridis"}


def _plot_waveform(samples: np.ndarray, sample_rate: int, path: str, theme: dict) -> None:
    times = np.arange(samples.size, dtype=np.float32) / float(sample_rate)
    figure, axis = plt.subplots(figsize=(11, 3.8), facecolor=theme["face"])
    axis.set_facecolor(theme["face"])
    axis.plot(times, samples, color=theme["line"], linewidth=0.9)
    axis.set_title("Waveform", color=theme["text"])
    axis.set_xlabel("Time (seconds)", color=theme["text"])
    axis.set_ylabel("Amplitude", color=theme["text"])
    axis.tick_params(colors=theme["text"])
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_spectrogram(matrix: np.ndarray, path: str, title: str, y_label: str, theme: dict) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.5), facecolor=theme["face"])
    axis.set_facecolor(theme["face"])
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap=theme["cmap"])
    axis.set_title(title, color=theme["text"])
    axis.set_xlabel("Frame", color=theme["text"])
    axis.set_ylabel(y_label, color=theme["text"])
    axis.tick_params(colors=theme["text"])
    figure.colorbar(image, ax=axis, label="dB")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run(inputs: dict, params: dict, context) -> dict:
    audio_value = inputs.get("audio")
    if audio_value is None:
        raise ValueError("Spectrogram Viewer requires an Audio input.")

    sample_rate = int(params.get("sample_rate", 16000))
    max_seconds = float(params.get("max_seconds", 30.0))
    theme_name = str(params.get("theme", "accessible")).lower()
    if theme_name not in {"accessible", "dark", "print"}:
        raise ValueError("Plot Theme must be accessible, dark, or print.")
    if max_seconds <= 0:
        raise ValueError("Max Seconds To Plot must be greater than zero.")

    source_path = _input_path(audio_value)
    data, original_rate = sf.read(source_path, always_2d=True, dtype="float32")
    samples = _mono(data)
    if samples.size == 0:
        raise ValueError("Audio input contains no samples.")

    samples = _resample_linear(samples, int(original_rate), sample_rate)
    max_samples = int(max_seconds * sample_rate)
    if samples.size > max_samples:
        samples = samples[:max_samples]

    display = _theme(theme_name)
    n_fft = 1024
    hop_length = 256
    spectrogram = _power_spectrogram(samples, n_fft, hop_length)
    spectrogram_db = _amplitude_to_db(spectrogram)
    mel = _mel_filterbank(sample_rate, n_fft, 64) @ spectrogram
    mel_db = _amplitude_to_db(mel)

    waveform_path = context.output_path("waveform.png", port_id="waveform_plot")
    spectrogram_path = context.output_path("spectrogram.png", port_id="spectrogram_plot")
    mel_path = context.output_path("mel_spectrogram.png", port_id="mel_spectrogram_plot")
    _plot_waveform(samples, sample_rate, waveform_path, display)
    _plot_spectrogram(spectrogram_db, spectrogram_path, "Spectrogram", "Frequency bin", display)
    _plot_spectrogram(mel_db, mel_path, "Mel Spectrogram", "Mel band", display)

    report = {
        "title": "Audio Visualization Report",
        "source_file": Path(source_path).name,
        "original_sample_rate": int(original_rate),
        "visualization_sample_rate": sample_rate,
        "duration_plotted_seconds": round(samples.size / sample_rate, 4),
        "theme": theme_name,
        "artifacts": {
            "waveform_plot": waveform_path,
            "spectrogram_plot": spectrogram_path,
            "mel_spectrogram_plot": mel_path,
        },
    }
    return {
        "waveform_plot": waveform_path,
        "spectrogram_plot": spectrogram_path,
        "mel_spectrogram_plot": mel_path,
        "visualization_report": report,
    }
