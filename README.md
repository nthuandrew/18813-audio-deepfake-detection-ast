# Audio Deepfake Detection Using Audio Spectrogram Transformers

Course project for 18-813: Foundations of Responsible AI

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

## Models

CNN and AST use the same dataset loader, binary labels, prediction schema,
training loop, and evaluation CLI. Architecture-specific preprocessing is
stored in each checkpoint and restored automatically during prediction.

| Setting | CNN baseline | AST |
| --- | --- | --- |
| Config | `configs/cnn_baseline.yaml` | `configs/ast.yaml` |
| Model name in config | `cnn` | `ast` |
| Output namespace | `cnn_baseline` | `ast` |
| Input clip | 4.0 seconds | 10.24 seconds |
| Mel bands | 128 | 128 |
| Initialization | Random | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Published checkpoint | `artifacts/cnn_baseline/seed_<N>/best.pt` | `artifacts/ast/seed_<N>/best.pt` |
| Published metrics | `artifacts/cnn_baseline/seed_<N>/eval_metrics.json` | `artifacts/ast/seed_<N>/eval_metrics.json` |

The different clip durations mean the current results are preliminary rather
than a controlled architecture-only comparison. Do not change preprocessing at
evaluation time: `add-predict` reads the resolved data and model configuration
embedded in the selected checkpoint.

### Score and label convention

```text
target = 1 and high score -> bona fide
target = 0 and low score  -> spoof
```

Both models return one raw bona-fide logit per sample. Prediction applies a
sigmoid and writes the same CSV columns for CNN and AST:

```text
utterance_id,label,target,score,attack_id
```

## Training

Train one CNN seed:

```bash
add-train \
  --config configs/cnn_baseline.yaml \
  --seed 42 \
  --output-dir outputs/cnn_baseline/seed_42
```

Train one AST seed:

```bash
add-train \
  --config configs/ast.yaml \
  --seed 42 \
  --output-dir outputs/ast/seed_42
```

The same pattern applies to the planned five seeds:

```bash
for seed in 42 43 44 45 46; do
  add-train \
    --config configs/cnn_baseline.yaml \
    --seed "$seed" \
    --output-dir "outputs/cnn_baseline/seed_${seed}"

  add-train \
    --config configs/ast.yaml \
    --seed "$seed" \
    --output-dir "outputs/ast/seed_${seed}"
done
```

Each run writes its resolved `config.yaml`, `history.json`, best-development-loss
`best.pt`, `dev_predictions.csv`, and `dev_metrics.json` beneath `outputs/`.
Generated outputs are ignored by Git.

For a quick CNN pipeline check, deterministic subsets can be selected with
`--max-train-samples` and `--max-dev-samples`. Subset results are smoke tests
only and must not be reported as final results.

## Published checkpoints

CNN checkpoints are stored directly in this repository. Fine-tuned AST
checkpoints can be downloaded from
[Google Drive](https://drive.google.com/drive/folders/1PVUkPOT6YelrAOSSj7bwpgfqOhnmArI6?usp=sharing).
Place the AST files alongside their checked-in configs and metrics:

```text
artifacts/
├── cnn_baseline/
│   ├── seed_42/
│   │   ├── best.pt
│   │   ├── config.yaml
│   │   └── eval_metrics.json
│   └── seed_43 ... seed_46/
└── ast/
    ├── seed_42/
    │   ├── best.pt             # downloaded separately
    │   ├── config.yaml
    │   └── eval_metrics.json
    └── seed_43 ... seed_46/
```

The checked-in `eval_metrics.json` files allow result inspection without the
dataset or checkpoint. Reproducing predictions requires the ASVspoof data and
the corresponding `best.pt`.

## Evaluation workflow for CNN and AST

The commands below are deliberately model-agnostic. Set `MODEL` to either
`cnn_baseline` or `ast`; all remaining steps are identical.

### 1. Generate development predictions

Development predictions are used for sanity checks and any threshold selection.
Never select a threshold using evaluation labels.

```bash
MODEL=cnn_baseline  # change to: MODEL=ast
SEED=42

add-predict \
  --checkpoint "artifacts/${MODEL}/seed_${SEED}/best.pt" \
  --split dev \
  --output "outputs/${MODEL}/seed_${SEED}/dev_predictions.csv"
```

### 2. Generate evaluation predictions

```bash
add-predict \
  --checkpoint "artifacts/${MODEL}/seed_${SEED}/best.pt" \
  --split eval \
  --output "outputs/${MODEL}/seed_${SEED}/eval_predictions.csv"
```

`add-predict` reconstructs CNN or AST from the checkpoint, loads the checkpoint's
preprocessing settings, and preserves the official split order and attack IDs.

### 3. Compute metrics

Use the default threshold of 0.5, or pass a threshold selected from development
predictions:

```bash
add-evaluate \
  --predictions "outputs/${MODEL}/seed_${SEED}/eval_predictions.csv" \
  --threshold 0.5 \
  --output "outputs/${MODEL}/seed_${SEED}/eval_metrics.json"
```

The JSON contains overall EER, accuracy, balanced accuracy, precision, recall,
F1, bona-fide recall, spoof recall, and confusion counts. `by_attack` contains
the same metrics for A07-A19, where each attack-specific subset uses all
evaluation bona-fide samples plus the spoof samples from that attack.

### 4. Evaluate all five seeds

```bash
MODEL=cnn_baseline  # change to: MODEL=ast

for seed in 42 43 44 45 46; do
  add-predict \
    --checkpoint "artifacts/${MODEL}/seed_${seed}/best.pt" \
    --split eval \
    --output "outputs/${MODEL}/seed_${seed}/eval_predictions.csv"

  add-evaluate \
    --predictions "outputs/${MODEL}/seed_${seed}/eval_predictions.csv" \
    --threshold 0.5 \
    --output "outputs/${MODEL}/seed_${seed}/eval_metrics.json"
done
```

Keep per-seed results separate. Aggregate means and confidence intervals only
after all five runs use the same split, metric definitions, and threshold
selection procedure.

## Single-audio CNN inference

The tracked example accepts WAV, FLAC, and other SoundFile-supported audio:

```bash
python3 baseline_inference.py \
  --checkpoint artifacts/cnn_baseline/seed_42/best.pt \
  --audio path/to/audio.flac
```

It restores preprocessing from the checkpoint, converts audio to mono,
resamples when necessary, and prints the bona-fide score and thresholded label.
This script is for the CNN baseline; dataset-level CNN and AST evaluation should
use the shared `add-predict` workflow above.

## Reports

- Initial CNN train/development run: [`reports/baseline_report.md`](reports/baseline_report.md)
- CNN evaluation and A07-A19 analysis: [`reports/evaluation_report.md`](reports/evaluation_report.md)
- AST evaluation and A07-A19 analysis: [`reports/ast_evaluation_report.md`](reports/ast_evaluation_report.md)

## Contributor interfaces

`audio_deepfake.models.factory.build_model` is the shared model boundary. Both
CNN and AST accept `[batch, 1, n_mels, frames]` tensors and return raw bona-fide
logits shaped `[batch]`. Evaluators should consume prediction CSVs rather than
calling architecture-specific model internals.

Run the test suite before opening a pull request:

```bash
pytest
ruff check src tests data/example.py baseline_inference.py
```
