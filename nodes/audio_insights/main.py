import math


def _get_number(*sources, key: str, default=None):
    for source in sources:
        if isinstance(source, dict) and isinstance(source.get(key), (int, float)):
            return source[key]
    return default


def _classification(report: dict | None) -> tuple[str | None, float | None]:
    if not isinstance(report, dict):
        return None, None
    label = report.get("predicted_class") or report.get("raw_predicted_class")
    confidence = report.get("confidence")
    return (str(label) if label else None, float(confidence) if isinstance(confidence, (int, float)) else None)


def _frequency_text(feature_summary: dict | None) -> str | None:
    if not isinstance(feature_summary, dict):
        return None
    range_info = feature_summary.get("dominant_frequency_range")
    if not isinstance(range_info, dict):
        return None
    centroid = range_info.get("centroid_hz")
    bandwidth = range_info.get("bandwidth_hz")
    if not isinstance(centroid, (int, float)) or not isinstance(bandwidth, (int, float)):
        return None
    low = max(0, centroid - bandwidth / 2)
    high = centroid + bandwidth / 2
    return f"{low:.0f}Hz-{high:.0f}Hz"


def _quality_notes(duration, loudness_db, clipping_ratio, silence_ratio) -> list[str]:
    notes = []
    if isinstance(duration, (int, float)) and duration < 1.0:
        notes.append("The recording is very short; classification confidence may be limited.")
    if isinstance(loudness_db, (int, float)) and loudness_db < -45:
        notes.append("The recording is quiet. Consider volume normalization before downstream analysis.")
    if isinstance(clipping_ratio, (int, float)) and clipping_ratio > 0.001:
        notes.append("Clipping was detected. Results may be distorted by overloaded samples.")
    if isinstance(silence_ratio, (int, float)) and silence_ratio > 0.7:
        notes.append("A large portion of the recording is near-silent.")
    return notes or ["Audio quality looks usable for the selected analysis path."]


def _recommendations(class_label, confidence, notes) -> list[str]:
    items = []
    if confidence is not None and confidence < 0.55:
        items.append("Review the spectrogram and consider a domain-specific classifier for this recording type.")
    if class_label == "speech_like":
        items.append("For speech workflows, keep a 16 kHz mono version and consider adding transcription or diarization downstream.")
    elif class_label == "music_like":
        items.append("For music workflows, use a higher sample rate if timbre or instrument detail matters.")
    elif class_label == "environmental_noise":
        items.append("For sound-event workflows, collect labeled examples from the target environment before automating decisions.")
    if any("quiet" in note.lower() for note in notes):
        items.append("Run Audio Processor with Normalize Volume enabled.")
    return items or ["Use the generated plots and feature summary to compare this recording with similar examples."]


def _markdown(report: dict, audience: str) -> str:
    lines = [
        "# Audio Insights",
        "",
        f"- Duration: {report.get('duration_seconds', 'unknown')} seconds",
        f"- Average loudness: {report.get('average_loudness_db', 'unknown')} dB",
        f"- Dominant frequencies: {report.get('dominant_frequency_range', 'unknown')}",
        f"- Detected class: {report.get('detected_class', 'unknown')}",
        f"- Confidence: {report.get('confidence_percent', 'unknown')}%",
        "",
        "## Quality Notes",
    ]
    lines.extend(f"- {item}" for item in report["quality_notes"])
    if audience != "executive":
        lines.extend(["", "## Recommendations"])
        lines.extend(f"- {item}" for item in report["recommendations"])
    return "\n".join(lines)


def run(inputs: dict, params: dict, context) -> dict:
    audience = str(params.get("audience", "general")).lower()
    if audience not in {"general", "technical", "executive"}:
        raise ValueError("Audience must be general, technical, or executive.")
    include_recommendations = bool(params.get("include_recommendations", True))

    metadata = inputs.get("metadata") if isinstance(inputs.get("metadata"), dict) else {}
    processing = inputs.get("processing_report") if isinstance(inputs.get("processing_report"), dict) else {}
    statistics = inputs.get("audio_statistics") if isinstance(inputs.get("audio_statistics"), dict) else {}
    feature_summary = inputs.get("feature_summary") if isinstance(inputs.get("feature_summary"), dict) else {}
    classification_report = inputs.get("classification_report") if isinstance(inputs.get("classification_report"), dict) else {}
    visualization_report = inputs.get("visualization_report") if isinstance(inputs.get("visualization_report"), dict) else {}

    processed = processing.get("processed") if isinstance(processing.get("processed"), dict) else {}
    features = feature_summary.get("features") if isinstance(feature_summary.get("features"), dict) else {}
    duration = _get_number(statistics, processed, metadata, features, key="duration_seconds")
    loudness_db = _get_number(statistics, processed, features, key="rms_db")
    clipping_ratio = _get_number(statistics, processed, key="clipping_ratio", default=0.0)
    silence_ratio = _get_number(statistics, processed, key="silence_ratio", default=0.0)
    class_label, confidence = _classification(classification_report)
    frequency_range = _frequency_text(feature_summary)
    notes = _quality_notes(duration, loudness_db, clipping_ratio, silence_ratio)
    recommendations = _recommendations(class_label, confidence, notes) if include_recommendations else []

    structured = {
        "title": "Audio Insights Report",
        "audience": audience,
        "duration_seconds": round(float(duration), 4) if isinstance(duration, (int, float)) else None,
        "average_loudness_db": round(float(loudness_db), 2) if isinstance(loudness_db, (int, float)) and math.isfinite(loudness_db) else None,
        "dominant_frequency_range": frequency_range,
        "detected_class": class_label,
        "confidence": round(float(confidence), 6) if isinstance(confidence, (int, float)) else None,
        "confidence_percent": round(float(confidence) * 100, 1) if isinstance(confidence, (int, float)) else None,
        "quality_notes": notes,
        "recommendations": recommendations,
        "available_visualizations": visualization_report.get("artifacts", {}) if isinstance(visualization_report, dict) else {},
        "inputs_used": {
            "metadata": bool(metadata),
            "processing_report": bool(processing),
            "audio_statistics": bool(statistics),
            "feature_summary": bool(feature_summary),
            "classification_report": bool(classification_report),
            "visualization_report": bool(visualization_report),
        },
    }
    markdown = _markdown(structured, audience)

    if isinstance(duration, (int, float)):
        context.emit_metric("duration_seconds", float(duration), port_id="metrics")
    if isinstance(loudness_db, (int, float)) and math.isfinite(loudness_db):
        context.emit_metric("average_loudness_db", float(loudness_db), port_id="metrics")
    if isinstance(confidence, (int, float)):
        context.emit_metric("classification_confidence", float(confidence), port_id="metrics")
    context.emit_metric("quality_note_count", len(notes), port_id="metrics")

    return {
        "structured_report": structured,
        "markdown_summary": markdown,
        "metrics": {
            "duration_seconds": structured["duration_seconds"],
            "average_loudness_db": structured["average_loudness_db"],
            "classification_confidence": structured["confidence"],
            "quality_note_count": len(notes),
        },
    }
