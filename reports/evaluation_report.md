# CNN Evaluation Report

## Summary

This report evaluates the trained CNN baseline on the complete ASVspoof 2019
Logical Access evaluation partition. The checkpoint was trained for three
epochs on the official training partition and selected using development loss.
Training and development contain attacks A01-A06; evaluation contains attacks
A07-A19.

Overall evaluation performance is substantially worse than development
performance: EER increases from **0.0392%** on development to **12.7923%** on
evaluation. Performance also varies sharply by attack. A07-A09 are detected
perfectly at the fixed threshold, while only 1.22% of A17 spoof samples are
correctly rejected.

## Evaluation setup

| Item | Value |
| --- | --- |
| Evaluation date | 2026-11-23 |
| Checkpoint | `outputs/cnn_baseline/initial_full_seed_42/best.pt` |
| Selected epoch | 3 |
| Training attacks | A01-A06 |
| Evaluation attacks | A07-A19 |
| Evaluation bona fide | 7,355 |
| Evaluation spoof | 63,882 |
| Total evaluation samples | 71,237 |
| Spoof samples per attack | 4,914 |
| Classification threshold | 0.5 |
| Positive class | Bona fide |

The checkpoint produces a bona-fide score between zero and one. Scores at or
above 0.5 are classified as bona fide; lower scores are classified as spoof.
The evaluation labels were used only after inference to calculate metrics. They
were not used to update the model or tune the threshold.

## Metric definitions

- **EER:** interpolated operating point where spoof false-accept and bona-fide
  false-reject rates are equal. EER is calculated from score ranking and does
  not depend on the fixed 0.5 threshold.
- **Accuracy:** proportion of all samples classified correctly.
- **Balanced accuracy:** average of bona-fide recall and spoof recall.
- **Precision:** among samples classified as bona fide, the proportion that are
  actually bona fide.
- **Recall / bona-fide recall:** proportion of bona-fide samples classified as
  bona fide.
- **F1:** harmonic mean of bona-fide precision and recall.
- **Spoof recall:** proportion of spoof samples correctly classified as spoof.
- **TP/TN/FP/FN:** confusion counts with bona fide as the positive class. A
  false positive is therefore a spoof sample incorrectly accepted as bona fide.

## Overall evaluation results

| Metric | Value |
| --- | ---: |
| EER | 12.7923% |
| Accuracy | 65.9362% |
| Balanced accuracy | 80.9109% |
| Precision | 23.2328% |
| Recall | 99.7825% |
| F1 | 37.6900% |
| Bona-fide recall | 99.7825% |
| Spoof recall | 62.0394% |
| True positive | 7,339 |
| True negative | 39,632 |
| False positive | 24,250 |
| False negative | 16 |

### Overall confusion matrix

| Actual / predicted | Bona fide | Spoof |
| --- | ---: | ---: |
| Bona fide | 7,339 | 16 |
| Spoof | 24,250 | 39,632 |

The model preserves nearly all bona-fide samples, but incorrectly accepts
24,250 spoof samples. This explains why recall is very high while precision and
F1 are much lower. Accuracy alone is not sufficient for interpreting this
imbalanced evaluation set.

## Attack-specific results

Each attack-specific set contains all 7,355 evaluation bona-fide samples and
the 4,914 spoof samples from exactly one attack. This construction provides
both classes for EER and other binary metrics. Because the same bona-fide set
and threshold are used in every row, bona-fide recall and false negatives are
identical across attacks; differences are driven by spoof detection.

| Attack | EER | Accuracy | Balanced accuracy | Precision | Recall | F1 | Bona-fide recall | Spoof recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A07 | 0.0000% | 99.8696% | 99.8912% | 100.0000% | 99.7825% | 99.8911% | 99.7825% | 100.0000% |
| A08 | 0.0000% | 99.8696% | 99.8912% | 100.0000% | 99.7825% | 99.8911% | 99.7825% | 100.0000% |
| A09 | 0.0000% | 99.8696% | 99.8912% | 100.0000% | 99.7825% | 99.8911% | 99.7825% | 100.0000% |
| A10 | 15.8934% | 75.5726% | 69.5595% | 71.1143% | 99.7825% | 83.0438% | 99.7825% | 39.3366% |
| A11 | 11.1315% | 82.4762% | 78.1778% | 77.4728% | 99.7825% | 87.2237% | 99.7825% | 56.5731% |
| A12 | 21.9170% | 69.1417% | 61.5314% | 66.0636% | 99.7825% | 79.4952% | 99.7825% | 23.2804% |
| A13 | 17.0940% | 68.3919% | 60.5953% | 65.5209% | 99.7825% | 79.1011% | 99.7825% | 21.4082% |
| A14 | 0.5303% | 99.4458% | 99.3621% | 99.2964% | 99.7825% | 99.5389% | 99.7825% | 98.9418% |
| A15 | 16.4650% | 76.5833% | 70.8212% | 71.9792% | 99.7825% | 83.6306% | 99.7825% | 41.8600% |
| A16 | 0.0204% | 99.8696% | 99.8912% | 100.0000% | 99.7825% | 99.8911% | 99.7825% | 100.0000% |
| A17 | 35.2685% | 60.3065% | 50.5017% | 60.1903% | 99.7825% | 75.0870% | 99.7825% | 1.2210% |
| A18 | 7.4507% | 69.6389% | 62.1521% | 66.4283% | 99.7825% | 79.7587% | 99.7825% | 24.5218% |
| A19 | 0.3127% | 99.6169% | 99.5758% | 99.5794% | 99.7825% | 99.6808% | 99.7825% | 99.3691% |

### Attack-specific confusion counts

Each row contains 7,355 bona-fide and 4,914 spoof samples.

| Attack | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| A07 | 7,339 | 4,914 | 0 | 16 |
| A08 | 7,339 | 4,914 | 0 | 16 |
| A09 | 7,339 | 4,914 | 0 | 16 |
| A10 | 7,339 | 1,933 | 2,981 | 16 |
| A11 | 7,339 | 2,780 | 2,134 | 16 |
| A12 | 7,339 | 1,144 | 3,770 | 16 |
| A13 | 7,339 | 1,052 | 3,862 | 16 |
| A14 | 7,339 | 4,862 | 52 | 16 |
| A15 | 7,339 | 2,057 | 2,857 | 16 |
| A16 | 7,339 | 4,914 | 0 | 16 |
| A17 | 7,339 | 60 | 4,854 | 16 |
| A18 | 7,339 | 1,205 | 3,709 | 16 |
| A19 | 7,339 | 4,883 | 31 | 16 |

## Interpretation

The attacks fall into three broad groups for this checkpoint:

- **Strong detection:** A07-A09, A14, A16, and A19 have EER below 1% and spoof
  recall above 98.9%.
- **Weak generalization:** A10-A13, A15, and A18 have substantial EER and low
  spoof recall at the fixed threshold.
- **Primary failure mode:** A17 has 35.27% EER and only 1.22% spoof recall. The
  model accepts 4,854 of its 4,914 spoof samples as bona fide.

The large development-to-evaluation gap indicates that the CNN learned cues
that transfer well to some attack systems but poorly to others. The development
set uses the same A01-A06 attack families as training, whereas evaluation uses
A07-A19. The high development score therefore does not mean that audio
deepfake detection is solved; it primarily measures known-attack performance.

## Artifacts and reproduction

The local evaluation artifacts are:

```text
outputs/cnn_baseline/initial_full_seed_42/
├── best.pt
├── eval_predictions.csv
└── eval_metrics.json
```

They are intentionally excluded from Git. Recompute the overall metrics from
the saved predictions with:

```bash
add-evaluate \
  --predictions outputs/cnn_baseline/initial_full_seed_42/eval_predictions.csv \
  --output outputs/cnn_baseline/initial_full_seed_42/eval_metrics.json
```

The next formal comparison should repeat training across five seeds, report
confidence intervals, and apply the same evaluation protocol to the AST model.
