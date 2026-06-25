# Architecture

## Package Layout

```text
data2flow.plugin.json
nodes/
  audio_loader/
  audio_processor/
  feature_extractor/
  audio_classifier/
  spectrogram_viewer/
  audio_insights/
```

Each node contains a `metadata.json` manifest and a self-contained `main.py`. This matches Data2Flow's plugin runtime, where node scripts execute independently and should not depend on sibling Python modules.

## Data Flow

Audio Toolkit is built around three artifact families:

| Artifact family | Producers | Consumers |
| --- | --- | --- |
| Audio files | Audio Loader, Audio Processor | Audio Processor, Feature Extractor, Spectrogram Viewer, Audio Classifier |
| Structured reports and metrics | Every analytical node | Audio Insights, user-facing previews |
| Visual files | Feature Extractor, Spectrogram Viewer | Audio Insights, user review |

The normal path is:

```text
audio_file -> waveform -> processed_waveform -> feature_summary -> classification_report -> structured_report
                                      \-> visualization_report -----------/
```

## Runtime Boundary

Audio files are read and written with `soundfile`, which returns NumPy arrays. Nodes keep processing in NumPy so runtime images do not need PyTorch shared libraries.

NumPy is used for:

| Operation | Node |
| --- | --- |
| Resampling | Audio Processor, Feature Extractor, Audio Classifier, Spectrogram Viewer |
| Spectrograms | Feature Extractor, Audio Classifier, Spectrogram Viewer |
| Mel spectrograms | Feature Extractor, Spectrogram Viewer |
| MFCCs | Feature Extractor |

## Classifier

The classifier is a lightweight NumPy linear model over normalized DSP features:

```text
rms_db
zero_crossing_rate
spectral_centroid_hz
spectral_bandwidth_hz
spectral_rolloff_hz
silence_ratio
spectral_flatness
duration_seconds
dominant_frequency_hz proximity to 440 Hz
```

It emits probabilities for:

```text
silence
speech_like
music_like
environmental_noise
percussive_impact
tonal_signal
```

This design keeps the plugin offline, transparent, and installable without external model downloads.

## Failure Handling

Nodes validate required inputs and parameter ranges before processing. Empty audio files, missing files, invalid channels, invalid output formats, invalid confidence thresholds, and invalid plot themes raise explicit `ValueError` exceptions.
