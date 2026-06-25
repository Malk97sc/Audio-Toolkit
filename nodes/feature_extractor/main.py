import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _load_audio(path: str, target_rate: int) -> tuple[np.ndarray, int, int]:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    samples = _mono(data)
    if samples.size == 0:
        raise ValueError("Audio input contains no samples.")
    original_rate = int(sample_rate)
    samples = _resample_linear(samples, original_rate, target_rate)
    return samples, target_rate, original_rate


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


def _dct_matrix(output_count: int, input_count: int) -> np.ndarray:
    rows = np.arange(output_count, dtype=np.float32)[:, None]
    cols = np.arange(input_count, dtype=np.float32)[None, :]
    basis = np.cos(np.pi / input_count * (cols + 0.5) * rows)
    basis[0] *= math.sqrt(1.0 / input_count)
    if output_count > 1:
        basis[1:] *= math.sqrt(2.0 / input_count)
    return basis.astype(np.float32)


def _power_to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def _spectral_features(power: np.ndarray, sample_rate: int, rolloff_percent: float) -> dict:
    frequencies = np.linspace(0.0, sample_rate / 2, power.shape[0], dtype=np.float32)[:, None]
    energy = np.maximum(power.sum(axis=0), 1e-12)
    centroid = (frequencies * power).sum(axis=0) / energy
    bandwidth = np.sqrt((((frequencies - centroid[None, :]) ** 2) * power).sum(axis=0) / energy)
    cumulative = np.cumsum(power, axis=0)
    threshold = energy * rolloff_percent
    rolloff_indices = np.argmax(cumulative >= threshold[None, :], axis=0)
    rolloff = rolloff_indices / max(1, power.shape[0] - 1) * (sample_rate / 2)
    safe_power = np.maximum(power, 1e-12)
    flatness = np.exp(np.mean(np.log(safe_power), axis=0)) / np.mean(safe_power, axis=0)
    mean_spectrum = power.mean(axis=1)
    spectral_flux = np.mean(np.abs(np.diff(power, axis=1))) if power.shape[1] > 1 else 0.0
    return {
        "spectral_centroid_hz": float(np.mean(centroid)),
        "spectral_bandwidth_hz": float(np.mean(bandwidth)),
        "spectral_rolloff_hz": float(np.mean(rolloff)),
        "spectral_flatness": float(np.mean(flatness)),
        "dominant_frequency_hz": float(frequencies[int(np.argmax(mean_spectrum)), 0]),
        "spectral_flux": float(spectral_flux),
    }


def _zero_crossing_rate(samples: np.ndarray) -> float:
    if samples.size < 2:
        return 0.0
    return float(np.mean(samples[1:] * samples[:-1] < 0))


def _rms_energy(samples: np.ndarray) -> tuple[float, float]:
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return rms, 20.0 * math.log10(max(rms, 1e-12))


def _plot_matrix(matrix: np.ndarray, path: str, title: str, x_label: str, y_label: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="magma")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    figure.colorbar(image, ax=axis, label="dB")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run(inputs: dict, params: dict, context) -> dict:
    audio_value = inputs.get("audio")
    if audio_value is None:
        raise ValueError("Feature Extractor requires an Audio input.")

    sample_rate = int(params.get("sample_rate", 16000))
    n_fft = int(params.get("n_fft", 1024))
    hop_length = int(params.get("hop_length", 256))
    mel_bands = int(params.get("mel_bands", 64))
    mfcc_count = int(params.get("mfcc_count", 13))
    rolloff_percent = float(params.get("rolloff_percent", 0.85))

    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("FFT Window Size and Hop Length must be positive.")
    if not 0.5 <= rolloff_percent <= 0.99:
        raise ValueError("Spectral Rolloff Percent must be between 0.50 and 0.99.")

    source_path = _input_path(audio_value)
    samples, current_rate, original_rate = _load_audio(source_path, sample_rate)
    duration = samples.size / float(current_rate)
    context.log(f"Extracting audio features from {Path(source_path).name}")

    spectrogram = _power_spectrogram(samples, n_fft, hop_length)
    spectrogram_db = _power_to_db(spectrogram)
    mel = _mel_filterbank(current_rate, n_fft, mel_bands) @ spectrogram
    mel_db = _power_to_db(mel)
    mfcc = _dct_matrix(mfcc_count, mel_bands) @ mel_db

    spectral = _spectral_features(spectrogram, current_rate, rolloff_percent)
    rms, rms_db = _rms_energy(samples)
    features = {
        "duration_seconds": duration,
        "sample_rate": current_rate,
        "zero_crossing_rate": _zero_crossing_rate(samples),
        "rms_energy": rms,
        "rms_db": rms_db,
        "mel_mean_db": float(np.mean(mel_db)),
        "mel_std_db": float(np.std(mel_db)),
        "mfcc_0_mean": float(np.mean(mfcc[0])) if mfcc.shape[0] else 0.0,
        "mfcc_mean_abs": float(np.mean(np.abs(mfcc))),
        **spectral,
    }

    rows = [{"feature": name, "value": round(float(value), 6)} for name, value in sorted(features.items())]
    for index in range(min(mfcc_count, mfcc.shape[0])):
        rows.append({"feature": f"mfcc_{index + 1}_mean", "value": round(float(np.mean(mfcc[index])), 6)})
    feature_frame = pd.DataFrame(rows)

    spectrogram_path = context.output_path("spectrogram.png", port_id="spectrogram_image")
    mel_path = context.output_path("mel_spectrogram.png", port_id="mel_spectrogram_image")
    _plot_matrix(spectrogram_db, spectrogram_path, "Spectrogram", "Frame", "Frequency bin")
    _plot_matrix(mel_db, mel_path, "Mel Spectrogram", "Frame", "Mel band")

    summary = {
        "title": "Audio Feature Summary",
        "source_file": Path(source_path).name,
        "duration_seconds": round(duration, 4),
        "original_sample_rate": original_rate,
        "analysis_sample_rate": current_rate,
        "frame_count": int(spectrogram.shape[-1]),
        "features": {name: round(float(value), 6) for name, value in features.items()},
        "dominant_frequency_range": {
            "center_hz": round(float(spectral["dominant_frequency_hz"]), 2),
            "centroid_hz": round(float(spectral["spectral_centroid_hz"]), 2),
            "bandwidth_hz": round(float(spectral["spectral_bandwidth_hz"]), 2),
        },
        "interpretation": [
            "Higher zero crossing rate usually indicates noisier or brighter audio.",
            "Higher spectral centroid indicates more high-frequency content.",
            "RMS dB summarizes average loudness and is useful for quality checks.",
        ],
    }
    metadata = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "mel_bands": mel_bands,
        "mfcc_count": mfcc_count,
        "rolloff_percent": rolloff_percent,
        "spectrogram_shape": list(spectrogram.shape),
        "mel_spectrogram_shape": list(mel.shape),
        "mfcc_shape": list(mfcc.shape),
    }

    return {
        "feature_vectors": feature_frame,
        "feature_summary": summary,
        "spectrogram_image": spectrogram_path,
        "mel_spectrogram_image": mel_path,
        "feature_metadata": metadata,
    }
