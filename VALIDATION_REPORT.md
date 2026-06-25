# Validation Report

## Static Checks

```bash
python3 -m py_compile nodes/*/main.py
python3 -m json.tool data2flow.plugin.json >/dev/null
for f in nodes/*/metadata.json; do python3 -m json.tool "$f" >/dev/null; done
```

Result: passed.

## Data2Flow Package Validation

```bash
docker compose exec -T backend uv run python -m app.cli plugins validate --path /tmp/Audio-Toolkit
```

Result: passed with no diagnostics after uninstalling the previous local build.

## Data2Flow Install Validation

```bash
docker compose exec -T backend uv run python -m app.cli plugins install --path /tmp/Audio-Toolkit
```

Result: passed.

Runtime image:

```text
data2flow-plugin-env:873b848154c02022
data2flow-plugin-env:community.audio_toolkit-873b848154c0
```

Dependency hash:

```text
873b848154c020224127c2ea0631d361abb22517da37de14f4a0ad71e7ba3236
```

## Runtime Smoke Test

The smoke test generated a synthetic 440 Hz audio file, then executed:

```text
Audio Loader
Audio Processor
Feature Extractor
Audio Classifier
Spectrogram Viewer
Audio Insights
```

Result:

```json
{
  "artifact_count": 7,
  "feature_rows": 28,
  "predicted_class": {
    "class": "tonal_signal",
    "raw_class": "tonal_signal",
    "confidence": 0.602797,
    "class_index": 5
  },
  "emitted_metric_count": 16
}
```

The smoke test verified that generated audio files, plot files, feature tables, probability tables, and markdown insights were all present and nonempty.

## Notes

The package hash changes when documentation changes. Use the Data2Flow validator output as the source of truth for the current package hash.
