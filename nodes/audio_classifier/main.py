import math
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F


CLASSES = [
    "silence",
    "speech_like",
    "music_like",
    "environmental_noise",
    "percussive_impact",
    "tonal_signal",
]


WEIGHTS = torch.tensor(
    [
        [-4.0, -1.0, -1.0, -1.0, -1.0, 5.5, -0.5, -0.5, -0.5],
        [1.6, 1.2, 0.2, 0.1, 0.2, -1.2, -0.8, 0.5, -0.8],
        [1.1, -0.7, 1.3, 1.0, 1.1, -1.1, -0.7, 0.4, 0.5],
        [0.6, 1.4, 1.0, 1.2, 1.1, -0.2, 1.7, 0.0, -0.5],
        [1.4, 1.8, 0.8, 1.7, 1.5, -0.4, 0.5, -0.8, -0.4],
        [0.8, -1.2, -0.6, -0.8, -0.8, -0.2, -1.5, 0.7, 3.4],
    ],
    dtype=torch.float32,
)

BIASES = torch.tensor([-1.1, 0.2, 0.1, -0.1, -0.4, 0.1], dtype=torch.float32)


def _input_path(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("file") or str(value)
    return str(value)


def _feature_map_from_table(frame: pd.DataFrame) -> dict:
    if not {"feature", "value"}.issubset(set(frame.columns)):
        raise ValueError("Feature Vectors must contain 'feature' and 'value' columns.")
    return {str(row["feature"]): float(row["value"]) for _, row in frame.iterrows()}


def _extract_from_audio(path: str) -> dict:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data).transpose(0, 1).contiguous()
    if waveform.numel() == 0:
        raise ValueError("Audio input contains no samples.")
    waveform = waveform.mean(dim=0, keepdim=True).float()
    target_rate = 16000
    if sample_rate != target_rate:
        waveform = F.resample(waveform, sample_rate, target_rate)
        sample_rate = target_rate
    spec = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=2.0)(waveform)
    freqs = torch.linspace(0, sample_rate / 2, spec.shape[1]).unsqueeze(1)
    energy = spec.squeeze(0).sum(dim=0).clamp_min(1e-12)
    centroid = (freqs * spec.squeeze(0)).sum(dim=0) / energy
    bandwidth = torch.sqrt((((freqs - centroid.unsqueeze(0)) ** 2) * spec.squeeze(0)).sum(dim=0) / energy)
    flat = waveform.reshape(-1)
    rms = torch.sqrt(torch.mean(flat.square())).item()
    zcr = (flat[1:] * flat[:-1] < 0).float().mean().item() if flat.numel() > 1 else 0.0
    flatness = torch.exp(torch.mean(torch.log(spec.clamp_min(1e-12)))) / torch.mean(spec.clamp_min(1e-12))
    return {
        "duration_seconds": flat.numel() / sample_rate,
        "zero_crossing_rate": zcr,
        "rms_energy": rms,
        "rms_db": 20.0 * math.log10(max(rms, 1e-12)),
        "spectral_centroid_hz": centroid.mean().item(),
        "spectral_bandwidth_hz": bandwidth.mean().item(),
        "spectral_rolloff_hz": centroid.quantile(0.85).item(),
        "spectral_flatness": flatness.item(),
        "dominant_frequency_hz": freqs[spec.squeeze(0).mean(dim=1).argmax()].item(),
    }


def _summary_features(inputs: dict) -> dict:
    if isinstance(inputs.get("feature_vectors"), pd.DataFrame):
        return _feature_map_from_table(inputs["feature_vectors"])
    if isinstance(inputs.get("feature_summary"), dict):
        summary = inputs["feature_summary"]
        features = summary.get("features")
        if isinstance(features, dict):
            return {str(key): float(value) for key, value in features.items() if isinstance(value, (int, float))}
    if inputs.get("audio") is not None:
        return _extract_from_audio(_input_path(inputs["audio"]))
    raise ValueError("Audio Classifier requires Feature Vectors, Feature Summary, or Audio.")


def _scale(features: dict) -> torch.Tensor:
    rms_db = float(features.get("rms_db", -80.0))
    zcr = float(features.get("zero_crossing_rate", 0.0))
    centroid = float(features.get("spectral_centroid_hz", 0.0))
    bandwidth = float(features.get("spectral_bandwidth_hz", 0.0))
    rolloff = float(features.get("spectral_rolloff_hz", 0.0))
    flatness = float(features.get("spectral_flatness", 0.0))
    duration = float(features.get("duration_seconds", 0.0))
    dominant = float(features.get("dominant_frequency_hz", 0.0))
    silence_ratio = float(features.get("silence_ratio", 0.0))
    if rms_db < -70:
        silence_ratio = max(silence_ratio, 0.95)
    vector = [
        max(-2.0, min(2.0, (rms_db + 35.0) / 18.0)),
        max(0.0, min(2.0, zcr / 0.12)),
        max(0.0, min(2.0, centroid / 3500.0)),
        max(0.0, min(2.0, bandwidth / 3500.0)),
        max(0.0, min(2.0, rolloff / 6500.0)),
        max(0.0, min(1.5, silence_ratio)),
        max(0.0, min(2.0, flatness / 0.35)),
        max(0.0, min(2.0, duration / 20.0)),
        max(0.0, min(2.0, 1.0 - abs(dominant - 440.0) / 1200.0)),
    ]
    return torch.tensor(vector, dtype=torch.float32)


def _explanations(features: dict, predicted: str) -> list[str]:
    reasons = []
    rms_db = float(features.get("rms_db", -80.0))
    zcr = float(features.get("zero_crossing_rate", 0.0))
    centroid = float(features.get("spectral_centroid_hz", 0.0))
    flatness = float(features.get("spectral_flatness", 0.0))
    if rms_db < -55:
        reasons.append("The recording has very low average loudness.")
    if zcr > 0.12:
        reasons.append("The signal crosses zero frequently, which often indicates noise or fricative speech.")
    if centroid > 3000:
        reasons.append("Energy is concentrated toward higher frequencies.")
    if flatness > 0.25:
        reasons.append("The spectrum is relatively flat, which is common in noisy audio.")
    if predicted == "music_like":
        reasons.append("The spectral spread and sustained energy look more music-like than speech-like.")
    if predicted == "speech_like":
        reasons.append("The feature pattern is consistent with moderate-bandwidth voiced or spoken content.")
    return reasons or ["The classifier combined loudness, spectral shape, and temporal features to select this class."]


def run(inputs: dict, params: dict, context) -> dict:
    threshold = float(params.get("confidence_threshold", 0.55))
    if threshold <= 0 or threshold >= 1:
        raise ValueError("Confidence Threshold must be between 0 and 1.")
    include_explanations = bool(params.get("include_explanations", True))

    features = _summary_features(inputs)
    vector = _scale(features)
    logits = WEIGHTS.matmul(vector) + BIASES
    probabilities = torch.softmax(logits, dim=0)
    best_index = int(torch.argmax(probabilities).item())
    best_class = CLASSES[best_index]
    confidence = float(probabilities[best_index].item())
    display_class = best_class if confidence >= threshold else "uncertain"

    probability_rows = [
        {
            "class": label,
            "probability": round(float(probabilities[index].item()), 6),
            "rank": int((probabilities > probabilities[index]).sum().item()) + 1,
        }
        for index, label in enumerate(CLASSES)
    ]
    probability_frame = pd.DataFrame(sorted(probability_rows, key=lambda row: row["probability"], reverse=True))

    report = {
        "title": "Audio Classification Report",
        "model": "Audio Toolkit lightweight PyTorch DSP classifier v1",
        "predicted_class": display_class,
        "raw_predicted_class": best_class,
        "confidence": round(confidence, 6),
        "confidence_threshold": threshold,
        "top_classes": probability_frame.head(3).to_dict(orient="records"),
        "input_features": {key: round(float(value), 6) for key, value in features.items() if isinstance(value, (int, float))},
        "explanations": _explanations(features, best_class) if include_explanations else [],
        "limitations": [
            "This is an offline lightweight classifier intended for workflow triage, not a substitute for a domain-specific model.",
            "For production taxonomies, validate against labeled examples from the target recording environment.",
        ],
    }
    predicted = {
        "class": display_class,
        "raw_class": best_class,
        "confidence": round(confidence, 6),
        "class_index": best_index,
    }
    confidence_scores = {row["class"]: row["probability"] for row in probability_rows}
    confidence_scores["top_confidence"] = round(confidence, 6)

    context.emit_metric("confidence", confidence, port_id="confidence_scores")
    context.emit_metric("class_index", best_index, port_id="confidence_scores")

    return {
        "predicted_class": predicted,
        "confidence_scores": confidence_scores,
        "probability_distribution": probability_frame,
        "classification_report": report,
    }
