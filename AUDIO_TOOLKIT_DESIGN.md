# Audio Toolkit Design

## Goals

Audio Toolkit is designed for Data2Flow users who need a complete audio inspection workflow without leaving the canvas. The plugin should work from a file upload, produce useful intermediate artifacts, and end with a clear summary that explains what happened to the audio.

## Non-Goals

The first release does not ship a large pretrained sound-event taxonomy, speech recognition, source separation, diarization, or music transcription. Those features require larger model assets, more specialized validation, and clearer resource controls.

## Classifier Tradeoff

Three classifier options were evaluated:

| Option | Strength | Cost |
| --- | --- | --- |
| YAMNet | Broad AudioSet taxonomy and small model footprint. | TensorFlow Hub dependency and external model asset flow. |
| PANNs | Strong AudioSet tagging performance in PyTorch. | External weights, larger model footprint, and download/cache management. |
| Embedded DSP classifier | Offline, transparent, deterministic, and easy to validate in Data2Flow. | Coarser taxonomy and lower ceiling than labeled domain models. |

The plugin uses the embedded DSP classifier because it satisfies Data2Flow's current plugin isolation model and can be validated without external downloads.

## I/O Decision

TorchAudio 2.9 and newer route `torchaudio.save()` through TorchCodec. During runtime validation, this failed without an explicit TorchCodec dependency. Importing PyTorch also adds a large shared-library footprint in constrained node runners. Audio Toolkit therefore uses `soundfile` for file I/O and NumPy for DSP.

This split keeps audio file handling simple:

```text
soundfile: read/write WAV and FLAC
numpy: resampling, spectral transforms, MFCCs, and classifier math
matplotlib: plots
pandas: feature and probability tables
```

## Extensibility

The node chain leaves obvious extension points:

| Extension | Integration point |
| --- | --- |
| Speech transcription | After Audio Processor |
| Domain classifier | Replace or follow Audio Classifier |
| Batch comparison | After Feature Extractor |
| QA thresholds | After Audio Processor or Audio Insights |
| Report export | After Audio Insights |

Future model-backed classifiers should preserve the current `probability_distribution` and `classification_report` outputs so existing workflows continue to work.
