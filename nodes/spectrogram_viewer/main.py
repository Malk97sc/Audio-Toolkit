from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _theme(name: str) -> dict:
    if name == "dark":
        return {"face": "#111827", "text": "#f8fafc", "line": "#38bdf8", "cmap": "magma"}
    if name == "print":
        return {"face": "#ffffff", "text": "#111827", "line": "#111827", "cmap": "gray_r"}
    return {"face": "#ffffff", "text": "#172554", "line": "#2563eb", "cmap": "viridis"}


def _plot_waveform(waveform: torch.Tensor, sample_rate: int, path: str, theme: dict) -> None:
    samples = _mono(waveform).squeeze(0).detach().cpu()
    times = torch.arange(samples.numel()) / sample_rate
    figure, axis = plt.subplots(figsize=(11, 3.8), facecolor=theme["face"])
    axis.set_facecolor(theme["face"])
    axis.plot(times.numpy(), samples.numpy(), color=theme["line"], linewidth=0.9)
    axis.set_title("Waveform", color=theme["text"])
    axis.set_xlabel("Time (seconds)", color=theme["text"])
    axis.set_ylabel("Amplitude", color=theme["text"])
    axis.tick_params(colors=theme["text"])
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_spectrogram(matrix: torch.Tensor, path: str, title: str, y_label: str, theme: dict) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.5), facecolor=theme["face"])
    axis.set_facecolor(theme["face"])
    data = matrix.detach().cpu().squeeze(0).numpy()
    image = axis.imshow(data, origin="lower", aspect="auto", cmap=theme["cmap"])
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
    waveform = torch.from_numpy(data).transpose(0, 1).contiguous()
    if waveform.numel() == 0:
        raise ValueError("Audio input contains no samples.")
    if original_rate != sample_rate:
        waveform = F.resample(waveform, original_rate, sample_rate)
    max_samples = int(max_seconds * sample_rate)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]

    display = _theme(theme_name)
    n_fft = 1024
    hop_length = 256
    mono = _mono(waveform).float()
    spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)(mono)
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=64)(mono)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

    waveform_path = context.output_path("waveform.png", port_id="waveform_plot")
    spectrogram_path = context.output_path("spectrogram.png", port_id="spectrogram_plot")
    mel_path = context.output_path("mel_spectrogram.png", port_id="mel_spectrogram_plot")
    _plot_waveform(waveform, sample_rate, waveform_path, display)
    _plot_spectrogram(spec_db, spectrogram_path, "Spectrogram", "Frequency bin", display)
    _plot_spectrogram(mel_db, mel_path, "Mel Spectrogram", "Mel band", display)

    report = {
        "title": "Audio Visualization Report",
        "source_file": Path(source_path).name,
        "original_sample_rate": original_rate,
        "visualization_sample_rate": sample_rate,
        "duration_plotted_seconds": round(waveform.shape[-1] / sample_rate, 4),
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
