# Audio Deepfake Detection Using Audio Spectrogram Transformers

Course project for 18-813: Foundations of Software Engineering.

## Team

- Andrew Chen
- Anirudh Dash
- Cassie Chang

## Overview

This project studies audio deepfake detection: given an input waveform, predict whether it is bona fide human speech or AI-generated speech. We compare an Audio Spectrogram Transformer (AST) with a convolutional neural network baseline using Mel-spectrogram inputs.

## Research questions

1. Can AST improve audio deepfake detection over a conventional CNN baseline?
2. How well do the models generalize to unseen spoofing attacks?
3. How robust are the models when background noise is added to the input audio?

## Planned methodology

- **Dataset:** ASVspoof 2019 Logical Access, with WaveFake as an alternative.
- **Features:** 2D Mel-spectrograms extracted from raw audio.
- **Models:** Audio Spectrogram Transformer and a lightweight CNN baseline.
- **Training:** Binary cross-entropy loss with AdamW and learning-rate scheduling.
- **Evaluation:** Equal Error Rate (EER), accuracy, confidence intervals, and results across five random seeds.
- **Robustness:** Evaluation on unseen attack types and multiple background-noise levels.

## Repository structure

```text
configs/          Version-controlled experiment configuration files
data/             Dataset setup instructions (local audio is not committed)
notebooks/        Exploratory notebooks
reports/figures/  Figures used in reports
src/audio_deepfake/
  data/            Protocol parser, validation, and shared log-Mel frontend
  models/          CNN baseline and the model extension point
  training/        Model-independent training and prediction utilities
tests/            Automated tests
```

## Setup

Python 3.10 or newer is required. Create an environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[dev]'
```

PyTorch and TorchAudio must be compatible with the machine's CUDA setup. The
command above installs their default wheels; use the
[PyTorch installer](https://pytorch.org/get-started/locally/) first on machines
that need a specific CUDA build.

## Data

Datasets and generated audio features are intentionally excluded from Git because of their size and licensing requirements. See `data/README.md` for the expected local layout.

After following those instructions, verify the complete extraction:

```bash
add-validate-data --la-root data/raw/asvspoof2019/LA
```

This checks official protocol counts and verifies that every listed FLAC file
exists. Do not start training unless it reports `Dataset validation PASSED`.

## CNN baseline

The checked-in baseline config documents every preprocessing and training
choice. Train it with:

```bash
add-train --config configs/cnn_baseline.yaml
```

Run the planned five seeds without editing the shared config (each run stores
its fully resolved config):

```bash
for seed in 42 43 44 45 46; do
  add-train \
    --config configs/cnn_baseline.yaml \
    --seed "$seed" \
    --output-dir "outputs/cnn_baseline/seed_${seed}"
done
```

The score convention is fixed across the project:

```text
target = 1 and high score -> bona fide
target = 0 and low score  -> spoof
```

The best development-loss checkpoint, copied config, loss history, and
development predictions are written beneath `outputs/`, which Git ignores.
Threshold selection and test metrics must use the development predictions, not
evaluation labels.

For a quick sanity check, every epoch prints and records development accuracy,
balanced accuracy, bona-fide recall, and spoof recall at a fixed score threshold
of 0.5. Prefer balanced accuracy for monitoring because the dataset contains far
more spoof than bona-fide samples. These monitoring values do not replace the
planned EER, confidence-interval, and threshold-tuned evaluation.

## Contributor interfaces

The shared model contract is implemented in
`audio_deepfake.models.factory.build_model`: a model receives a float tensor of
shape `[batch, 1, n_mels, frames]` and returns raw bona-fide logits of shape
`[batch]`. Cassie's AST implementation can add an `ast` factory branch without
changing the dataset or training loop.

`audio_deepfake.training.engine.predict` emits rows with this stable schema:

```text
utterance_id,label,target,score,attack_id
```

Anirudh's evaluation code can consume this schema for EER, threshold selection,
attack-wise results, confidence intervals, and noise robustness experiments.

Run the lightweight test suite before opening a pull request:

```bash
pytest
```
