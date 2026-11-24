# AST Evaluation Report

## Summary

This report evaluates the AST model on the complete ASVspoof 2019 Logical
Access evaluation partition. The model was initialized from
`MIT/ast-finetuned-audioset-10-10-0.4593`, fine-tuned on the official training
partition, and selected using development loss. Training and development use
attacks A01-A06; evaluation contains unseen attacks A07-A19.

The AST model achieves an evaluation EER of **10.6714%**, accuracy of
**84.3508%**, and spoof recall of **82.6258%**. It substantially reduces false
positives compared with the CNN evaluation result, but its performance is very
uneven across attacks. A17 and A18 remain difficult, with spoof recall of only
2.5234% and 1.3228%, respectively.

## Evaluation setup

| Item | Value |
| --- | --- |
| Evaluation date | 2026-11-23 |
| Checkpoint | `outputs/ast/seed_42/best.pt` |
| Selected epoch | 6 |
| Early stopping | After epoch 8 |
| Training attacks | A01-A06 |
| Evaluation attacks | A07-A19 |
| Evaluation bona fide | 7,355 |
| Evaluation spoof | 63,882 |
| Total evaluation samples | 71,237 |
| Spoof samples per attack | 4,914 |
| Classification threshold | 0.5 |
| Positive class | Bona fide |
| Clip duration | 10.24 seconds |
| Sample rate | 16,000 Hz |
| FFT size | 1,024 |
| Mel bins | 128 |
| AST pretrained checkpoint | `MIT/ast-finetuned-audioset-10-10-0.4593` |

The checkpoint produces a bona-fide score between zero and one. Scores at or
above 0.5 are classified as bona fide; lower scores are classified as spoof.
The evaluation labels were used only after inference to calculate metrics.

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
- **TP/TN/FP/FN:** confusion counts with bona fide as the positive class.

## Overall evaluation results

| Metric | Value |
| --- | ---: |
| EER | 10.6714% |
| Accuracy | 84.3508% |
| Balanced accuracy | 90.9798% |
| Precision | 39.6957% |
| Recall | 99.3338% |
| F1 | 56.7236% |
| Bona-fide recall | 99.3338% |
| Spoof recall | 82.6258% |
| True positive | 7,306 |
| True negative | 52,783 |
| False positive | 11,099 |
| False negative | 49 |

### Overall confusion matrix

| Actual / predicted | Bona fide | Spoof |
| --- | ---: | ---: |
| Bona fide | 7,306 | 49 |
| Spoof | 11,099 | 52,783 |

The model correctly detects almost all bona-fide samples, but still accepts
11,099 spoof samples as bona fide. This produces high bona-fide recall while
limiting precision and F1. Because the evaluation set is highly imbalanced,
accuracy alone is not sufficient for interpreting performance.

## Attack-specific results

Each attack-specific set contains all 7,355 evaluation bona-fide samples and
the 4,914 spoof samples from exactly one attack. Therefore, bona-fide recall
and false negatives are identical across rows; differences are driven by spoof
detection.

| Attack | EER | Accuracy | Balanced accuracy | Precision | Recall | F1 | Bona-fide recall | Spoof recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A07 | 0.2175% | 99.6006% | 99.6669% | 100.0000% | 99.3338% | 99.6658% | 99.3338% | 100.0000% |
| A08 | 2.0666% | 97.1799% | 96.6449% | 96.0936% | 99.3338% | 97.6869% | 99.3338% | 93.9560% |
| A09 | 0.0272% | 99.6006% | 99.6669% | 100.0000% | 99.3338% | 99.6658% | 99.3338% | 100.0000% |
| A10 | 0.3867% | 99.5028% | 99.5448% | 99.8360% | 99.3338% | 99.5843% | 99.3338% | 99.7558% |
| A11 | 0.3256% | 99.5273% | 99.5753% | 99.8770% | 99.3338% | 99.6046% | 99.3338% | 99.8168% |
| A12 | 0.0952% | 99.6006% | 99.6669% | 100.0000% | 99.3338% | 99.6658% | 99.3338% | 100.0000% |
| A13 | 0.1224% | 99.6006% | 99.6669% | 100.0000% | 99.3338% | 99.6658% | 99.3338% | 100.0000% |
| A14 | 0.4215% | 99.4947% | 99.5346% | 99.8224% | 99.3338% | 99.5775% | 99.3338% | 99.7354% |
| A15 | 0.0814% | 99.5925% | 99.6567% | 99.9863% | 99.3338% | 99.6590% | 99.3338% | 99.9796% |
| A16 | 0.2442% | 99.5517% | 99.6058% | 99.9179% | 99.3338% | 99.6250% | 99.3338% | 99.8779% |
| A17 | 32.8093% | 60.5591% | 50.9286% | 60.4001% | 99.3338% | 75.1221% | 99.3338% | 2.5234% |
| A18 | 26.6019% | 60.0782% | 50.3283% | 60.1069% | 99.3338% | 74.8949% | 99.3338% | 1.3228% |
| A19 | 4.1921% | 90.4556% | 88.2505% | 86.6872% | 99.3338% | 92.5806% | 99.3338% | 77.1673% |

### Attack-specific confusion counts

Each row contains 7,355 bona-fide samples and 4,914 spoof samples.

| Attack | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| A07 | 7,306 | 4,914 | 0 | 49 |
| A08 | 7,306 | 4,617 | 297 | 49 |
| A09 | 7,306 | 4,914 | 0 | 49 |
| A10 | 7,306 | 4,902 | 12 | 49 |
| A11 | 7,306 | 4,905 | 9 | 49 |
| A12 | 7,306 | 4,914 | 0 | 49 |
| A13 | 7,306 | 4,914 | 0 | 49 |
| A14 | 7,306 | 4,901 | 13 | 49 |
| A15 | 7,306 | 4,913 | 1 | 49 |
| A16 | 7,306 | 4,908 | 6 | 49 |
| A17 | 7,306 | 124 | 4,790 | 49 |
| A18 | 7,306 | 65 | 4,849 | 49 |
| A19 | 7,306 | 3,792 | 1,122 | 49 |

<!-- ## Development versus evaluation

| Metric | Development | Evaluation |
| --- | ---: | ---: |
| EER | 0.4081% | 10.6714% |
| Accuracy | 99.7907% | 84.3508% |
| Balanced accuracy | 99.2055% | 90.9798% |
| F1 | 98.9744% | 56.7236% |
| Bona-fide recall | 98.4694% | 99.3338% |
| Spoof recall | 99.9417% | 82.6258% |

The large development-to-evaluation gap is expected to some extent because the
development set uses attacks A01-A06, while the evaluation set uses unseen
attacks A07-A19. The result shows that the AST model performs strongly on the
development distribution but does not generalize equally well to every unseen
attack family. -->

## Interpretation

- **Strong detection:** A07 and A09-A16 have EER below 0.5% and spoof recall
  above 99.7%.
- **Moderate generalization:** A08 and A19 are weaker, with spoof recall of
  93.9560% and 77.1673%.
- **Primary failure modes:** A17 and A18 have EER of 32.8093% and 26.6019%,
  and spoof recall of only 2.5234% and 1.3228%.
- **Overall trade-off:** Compared with the CNN evaluation result, AST reduces
  false positives from 24,250 to 11,099 and improves spoof recall from 62.0394%
  to 82.6258%, while bona-fide recall decreases slightly from 99.7825% to
  99.3338%.

The attack-specific results are more informative than the overall accuracy:
the AST model is effective for most evaluation attacks, but it accepts nearly
all A17 and A18 spoof samples as bona fide. Additional training seeds and a
matched CNN/AST input configuration are needed for a fully controlled model
comparison.

## Artifacts and reproduction

The AST evaluation artifacts are:

```text
outputs/ast/seed_42/
├── best.pt
├── config.yaml
├── dev_metrics.json
├── dev_predictions.csv
├── eval_metrics.json
├── eval_predictions.csv
└── history.json
```

Recompute evaluation predictions and metrics with:

```bash
add-predict \
  --checkpoint outputs/ast/seed_42/best.pt \
  --split eval \
  --output outputs/ast/seed_42/eval_predictions.csv

add-evaluate \
  --predictions outputs/ast/seed_42/eval_predictions.csv \
  --output outputs/ast/seed_42/eval_metrics.json
```

This report covers seed 42 only. A formal comparison should repeat the same
protocol across multiple seeds and report variation or confidence intervals.
