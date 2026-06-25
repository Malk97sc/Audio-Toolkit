# Example Workflows

## Full Audio Triage

Use this for first-pass inspection of an uploaded recording.

```text
Audio Loader
Audio Processor
Feature Extractor
Audio Classifier
Spectrogram Viewer
Audio Insights
```

Recommended parameters:

| Node | Parameter | Value |
| --- | --- | --- |
| Audio Processor | target_sample_rate | 16000 |
| Audio Processor | channels | mono |
| Audio Processor | trim_silence | true |
| Audio Processor | normalize_peak | true |
| Feature Extractor | n_fft | 1024 |
| Feature Extractor | hop_length | 256 |
| Audio Classifier | confidence_threshold | 0.55 |
| Spectrogram Viewer | theme | accessible |
| Audio Insights | audience | technical |

## Quick Visualization

Use this when the user only needs plots and basic metadata.

```text
Audio Loader -> Audio Processor -> Spectrogram Viewer
```

Outputs:

```text
waveform_plot
spectrogram_plot
mel_spectrogram_plot
visualization_report
```

## Feature Export

Use this when the downstream workflow needs numeric features.

```text
Audio Loader -> Audio Processor -> Feature Extractor
```

The `feature_vectors` table can feed scoring, export, comparison, or later model-training nodes.

## Classification Review

Use this when a human reviewer needs transparent model evidence.

```text
Audio Loader -> Audio Processor -> Feature Extractor -> Audio Classifier
```

Review these outputs together:

```text
probability_distribution
classification_report.explanations
feature_summary.features
```

## Executive Summary

Use this when the final result should be readable by a nontechnical stakeholder.

```text
Audio Loader
Audio Processor
Feature Extractor
Audio Classifier
Spectrogram Viewer
Audio Insights
```

Set `Audio Insights` audience to `executive`.
