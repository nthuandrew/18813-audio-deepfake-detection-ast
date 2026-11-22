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
configs/          Experiment configuration files
data/             Local dataset instructions (audio data is not committed)
notebooks/        Exploratory notebooks
reports/figures/  Figures used in reports
src/              Source code
tests/            Automated tests
```

## Setup

Environment setup and reproducible experiment commands will be added as the implementation is developed.

## Data

Datasets and generated audio features are intentionally excluded from Git because of their size and licensing requirements. See `data/README.md` for the expected local layout.
