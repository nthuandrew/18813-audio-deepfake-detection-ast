# CNN Baseline Report

## Summary

This report documents the first end-to-end CNN baseline run on the ASVspoof
2019 Logical Access (LA) dataset. The purpose of the run was to verify the
shared data, feature, training, checkpoint, prediction, and metric pipelines on
the complete official training and development partitions.

The best checkpoint was obtained after three epochs. On the official
development partition it achieved **0.0392% EER**, **99.9638% accuracy**, and
**99.8234% F1**. These are preliminary known-attack results: both training and
development contain attacks A01-A06. They must not be interpreted as evidence
of generalization to the unseen A07-A19 attacks in the evaluation partition.

## Run information

| Item | Value |
| --- | --- |
| Date | 2026-11-22 |
| Dataset | ASVspoof 2019 LA |
| Training partition | 25,380 samples: 2,580 bona fide, 22,800 spoof |
| Development partition | 24,844 samples: 2,548 bona fide, 22,296 spoof |
| Attack IDs | A01-A06 in both train and development |
| Random seed | 42 |
| Epochs | 3 |
| Device | NVIDIA A10G |
| PyTorch | 2.15.0+cu132 |
| TorchAudio | 2.11.1+cu132 |
| CUDA runtime | 13.2 |

The dataset archive matched the documented MD5 checksum, and the dataset
validator confirmed the official row counts and the existence of every listed
FLAC file in train, development, and evaluation.

## Preprocessing

All audio is converted to mono and resampled to 16 kHz when necessary. Each
sample is cropped or zero-padded to four seconds, then converted to a log-Mel
spectrogram with the following settings:

| Setting | Value |
| --- | ---: |
| Sample rate | 16,000 Hz |
| Clip duration | 4.0 seconds |
| FFT size | 1,024 |
| Window length | 400 samples / 25 ms |
| Hop length | 160 samples / 10 ms |
| Mel bands | 128 |
| Frequency range | 0-8,000 Hz |
| Normalization | Per-sample mean and standard deviation |
| Model input shape | `[batch, 1, 128, 401]` |

The same frontend is shared with the AST extension point so that the CNN and
AST comparison can use consistent inputs.

## Model and training

The baseline is a compact 2D CNN with channel widths 32, 64, 128, and 256.
Each block contains a 3x3 convolution, batch normalization, ReLU, and 2x2 max
pooling. Adaptive average pooling, dropout 0.3, and a linear layer produce one
raw bona-fide logit per sample. The model has **389,057 trainable parameters**.

Training used:

- weighted binary cross-entropy, with bona fide encoded as 1 and spoof as 0;
- AdamW with learning rate 0.0003 and weight decay 0.0001;
- batch size 32;
- `ReduceLROnPlateau` with factor 0.5 and patience 2;
- checkpoint selection by lowest development loss;
- deterministic seed configuration.

The positive-class weight was computed from the actual training entries to
account for the substantial class imbalance.

## Results by epoch

Precision, recall, and F1 treat bona fide as the positive class. All
threshold-dependent metrics use a fixed bona-fide score threshold of 0.5. EER
is computed from the score ranking and does not depend on that fixed threshold.

| Epoch | Train loss | Dev loss | EER | Accuracy | Balanced accuracy | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.429670 | 0.357104 | 2.1193% | 84.9662% | 91.6067% | 40.5509% | 99.9608% | 57.6962% |
| 2 | 0.082490 | 0.060245 | 0.2691% | 99.1185% | 99.4915% | 92.1157% | 99.9608% | 95.8780% |
| 3 | 0.039973 | 0.027252 | **0.0392%** | **99.9638%** | **99.8929%** | **99.8430%** | **99.8038%** | **99.8234%** |

### Best-checkpoint confusion matrix

| Actual / predicted | Bona fide | Spoof |
| --- | ---: | ---: |
| Bona fide | 2,543 | 5 |
| Spoof | 4 | 22,292 |

The epoch-1 accuracy alone understates the model's ranking quality and obscures
threshold calibration: EER was already 2.12%, while the fixed 0.5 threshold
produced low precision. By epoch 3, both ranking and fixed-threshold metrics had
converged on the development set.

## Reproduction

Install the environment and validate the local dataset:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
add-validate-data --la-root data/raw/asvspoof2019/LA
```

Reproduce this exact three-epoch run:

```bash
add-train \
  --config configs/cnn_baseline.yaml \
  --epochs 3 \
  --seed 42 \
  --output-dir outputs/cnn_baseline/initial_full_seed_42
```

Recompute metrics from the saved prediction file:

```bash
add-evaluate \
  --predictions outputs/cnn_baseline/initial_full_seed_42/dev_predictions.csv \
  --output outputs/cnn_baseline/initial_full_seed_42/dev_metrics.json
```

The local run directory contains the resolved configuration, best checkpoint,
loss history, development predictions, and metric summary. These generated
artifacts are intentionally excluded from Git.

## Limitations and next steps

This run is a pipeline-validating baseline, not the final experiment:

- It uses one seed and three epochs rather than five seeds and the planned
  full training schedule.
- It reports development results for known attacks A01-A06 only.
- It does not evaluate unseen A07-A19 attacks.
- It does not include background-noise robustness experiments.
- The 0.5 classification threshold is fixed rather than selected on a separate
  development procedure for final reporting.
- Confidence intervals have not yet been computed.

The next evaluation should load this checkpoint, select any operating threshold
without using evaluation labels, and report full and attack-wise results on the
official evaluation partition. The final CNN/AST comparison should then repeat
the agreed protocol across five random seeds.
