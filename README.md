# Audio Toolkit for Data2Flow

Audio Toolkit is a production-ready Data2Flow plugin for offline audio analysis workflows. It provides six interoperable nodes for loading recordings, preparing signals, extracting audio features, classifying content, rendering spectrograms, and generating a final analyst-facing summary.

## Nodes

| Node | Purpose | Typical upstream input | Main outputs |
| --- | --- | --- | --- |
| Audio Loader | Reads WAV/FLAC/other libsndfile-supported audio and emits workflow-ready audio plus metadata. | File upload | waveform, metadata, audio_report |
| Audio Processor | Resamples, converts channels, trims silence, normalizes peak/RMS, and reports quality flags. | Audio Loader | processed_waveform, processing_report, audio_statistics |
| Feature Extractor | Computes DSP features, MFCCs, spectrograms, mel spectrograms, and feature tables. | Audio Processor | feature_vectors, feature_summary, spectrogram images |
| Audio Classifier | Uses a lightweight NumPy DSP-feature classifier for offline triage. | Feature Extractor | predicted_class, probability_distribution, classification_report |
| Spectrogram Viewer | Generates waveform, spectrogram, and mel spectrogram plots. | Audio Processor | waveform_plot, spectrogram_plot, visualization_report |
| Audio Insights | Combines upstream reports into a structured report, markdown summary, and metrics. | All upstream reports | structured_report, markdown_summary, metrics |

## Installation

From the Data2Flow UI:

1. Open Plugins.
2. Choose Clone from GitHub.
3. Use owner `Malk97sc` and repository `Audio-Toolkit`.
4. Install the package.

From the Data2Flow backend container:

```bash
uv run python -m app.cli plugins validate --path /tmp/Audio-Toolkit
uv run python -m app.cli plugins install --path /tmp/Audio-Toolkit
```

The plugin requires a dependency image with:

```text
numpy>=1.26.0
pandas>=2.0.0
matplotlib>=3.5.0
soundfile>=0.12.1
```

## Design Notes

The classifier is intentionally offline and deterministic. Public pretrained audio taggers such as YAMNet and PANNs provide broader taxonomies, but they add external model weights and larger runtime assumptions. Data2Flow plugin execution copies node entrypoints into isolated runtime scripts, so every node is implemented as a self-contained `main.py` with no shared package imports.

Audio I/O uses `soundfile`, and DSP operations use NumPy for resampling, spectrograms, mel spectrograms, MFCC extraction, and classification. This keeps the runtime lightweight and avoids mapping large PyTorch shared libraries in constrained node runners.

## Recommended Workflow

```text
Audio Loader
  -> Audio Processor
    -> Feature Extractor
      -> Audio Classifier
    -> Spectrogram Viewer
  -> Audio Insights
```

Audio Insights accepts reports from every upstream node and degrades gracefully when optional inputs are omitted.

## Limits

The built-in classifier is intended for workflow triage, not audited domain classification. For regulated or high-stakes use, validate the outputs against labeled data from the target recording environment or replace the classifier with a domain-specific model.
