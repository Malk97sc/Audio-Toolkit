import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F


def _input_path(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("file") or str(value)
    return str(value)


def _mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def _load_audio(path: str, target_rate: int) -> tuple[torch.Tensor, int, int]:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data).transpose(0, 1).contiguous()
    if waveform.numel() == 0:
        raise ValueError("Audio input contains no samples.")
    original_rate = sample_rate
    waveform = _mono(waveform).float()
    if sample_rate != target_rate:
        waveform = F.resample(waveform, sample_rate, target_rate)
        sample_rate = target_rate
    return waveform, sample_rate, original_rate


def _spectral_features(power: torch.Tensor, sample_rate: int, n_fft: int, rolloff_percent: float) -> dict:
    power = power.squeeze(0).float()
    freqs = torch.linspace(0, sample_rate / 2, power.shape[0], device=power.device).unsqueeze(1)
    energy = power.sum(dim=0).clamp_min(1e-12)
    centroid = (freqs * power).sum(dim=0) / energy
    bandwidth = torch.sqrt((((freqs - centroid.unsqueeze(0)) ** 2) * power).sum(dim=0) / energy)
    cumulative = torch.cumsum(power, dim=0)
    threshold = energy * rolloff_percent
    rolloff_indices = (cumulative >= threshold.unsqueeze(0)).float().argmax(dim=0)
    rolloff = rolloff_indices / max(1, power.shape[0] - 1) * (sample_rate / 2)
    flatness = torch.exp(torch.mean(torch.log(power.clamp_min(1e-12)), dim=0)) / torch.mean(power.clamp_min(1e-12), dim=0)
    return {
        "spectral_centroid_hz": centroid.mean().item(),
        "spectral_bandwidth_hz": bandwidth.mean().item(),
        "spectral_rolloff_hz": rolloff.mean().item(),
        "spectral_flatness": flatness.mean().item(),
        "dominant_frequency_hz": freqs[power.mean(dim=1).argmax()].item(),
        "spectral_flux": torch.mean(torch.abs(power[:, 1:] - power[:, :-1])).item() if power.shape[1] > 1 else 0.0,
    }


def _zero_crossing_rate(waveform: torch.Tensor) -> float:
    samples = waveform.squeeze(0)
    if samples.numel() < 2:
        return 0.0
    crossings = (samples[1:] * samples[:-1] < 0).float().mean()
    return crossings.item()


def _rms_energy(waveform: torch.Tensor) -> tuple[float, float]:
    rms = torch.sqrt(torch.mean(waveform.float().square())).item()
    return rms, 20.0 * math.log10(max(rms, 1e-12))


def _plot_matrix(matrix: torch.Tensor, path: str, title: str, x_label: str, y_label: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    data = matrix.detach().cpu().squeeze(0).numpy()
    image = axis.imshow(data, origin="lower", aspect="auto", cmap="magma")
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
    waveform, current_rate, original_rate = _load_audio(source_path, sample_rate)
    duration = waveform.shape[-1] / float(current_rate)
    context.log(f"Extracting audio features from {Path(source_path).name}")

    spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)(waveform)
    spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=current_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=mel_bands,
        power=2.0,
    )(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=current_rate,
        n_mfcc=mfcc_count,
        melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": mel_bands},
    )(waveform)

    spectral = _spectral_features(spectrogram, current_rate, n_fft, rolloff_percent)
    rms, rms_db = _rms_energy(waveform)
    features = {
        "duration_seconds": duration,
        "sample_rate": current_rate,
        "zero_crossing_rate": _zero_crossing_rate(waveform),
        "rms_energy": rms,
        "rms_db": rms_db,
        "mel_mean_db": mel_db.mean().item(),
        "mel_std_db": mel_db.std().item(),
        "mfcc_0_mean": mfcc[:, 0, :].mean().item() if mfcc.shape[1] else 0.0,
        "mfcc_mean_abs": mfcc.abs().mean().item(),
        **spectral,
    }

    rows = [{"feature": name, "value": round(float(value), 6)} for name, value in sorted(features.items())]
    for index in range(min(mfcc_count, mfcc.shape[1])):
        rows.append({"feature": f"mfcc_{index + 1}_mean", "value": round(float(mfcc[:, index, :].mean().item()), 6)})
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
