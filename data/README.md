# Local data layout

Do not commit datasets, extracted features, or generated audio to Git.

The intended local layout is:

```text
data/
  raw/          Original downloaded datasets
  processed/    Mel-spectrograms and other derived features
  splits/       Reproducible train, validation, and test manifests
```

Download and preprocessing instructions will be documented after the primary dataset is finalized.
