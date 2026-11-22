# Audio Deepfake Detection Using Audio Spectrogram Transformers

## Project information

- **Project type:** Technical project
- **Team members:** Andrew Chen, Anirudh Dash, and Cassie Chang
- **Primary task:** Binary classification of bona fide and AI-generated speech

## 1. Problem setting

This project focuses on audio deepfake detection (ADD). Given an input audio waveform, the goal is to determine whether the audio is:

- **Bona fide:** Human-generated speech
- **Spoofed:** AI-generated speech

Audio spoofing attacks generally fall into four categories:

1. Text-to-speech (TTS)
2. Voice conversion (VC)
3. Replay attacks
4. Mimicry

This project focuses on detecting TTS- and VC-generated audio. Speaker verification is outside the project scope; the objective is to determine whether an audio sample is AI-generated, regardless of the claimed speaker identity.

More formally, given an input waveform $x$, the project aims to learn a detector $f_\theta$ such that:

- $f_\theta(x) > \tau$ when $x$ is human-generated
- $f_\theta(x) \le \tau$ when $x$ is AI-generated

Here, $\tau$ is the classification threshold and $\theta$ represents the detector parameters.

## 2. Motivation

Recent speech foundation models and open-source systems such as F5-TTS, CosyVoice, and LLaSA have made high-quality synthetic speech easy to generate. Voice cloning may require only 3-10 seconds of reference audio, lowering the barrier to audio impersonation and fraud.

Audio deepfakes have already been used in harmful real-world scenarios, including:

- Impersonating a family member to demand ransom
- Impersonating executives or legal professionals to authorize fraudulent transfers
- Generating deceptive political or diplomatic messages

One reported attack used a fake CEO message and a cloned lawyer's voice to deceive a private bank into authorizing offshore transfers totaling $95 million. These risks motivate the development of strong, reliable, and robust detectors for AI-generated audio.

## 3. Proposal summary

The proposed pipeline uses the Audio Spectrogram Transformer (AST) to distinguish spoofed audio from bona fide speech. AST self-attention can capture relationships across time-frequency regions in a spectrogram, which may provide an advantage over conventional CNN-based detectors.

The project will compare AST with a standard CNN baseline under multiple spoofing attacks and noisy conditions. The intended contribution is an empirical evaluation of whether AST offers better detection performance and robustness for audio deepfake detection.

## 4. Research questions

1. Does AST outperform a conventional CNN baseline for audio deepfake detection?
2. How well do AST and CNN models generalize to unseen spoofing attacks?
3. How much does background noise degrade each model's performance?
4. Are the results stable across multiple random seeds?

## 5. Project plan

### 5.1 Dataset

Use the **ASVspoof 2019 Logical Access** dataset as the primary dataset. **WaveFake** may be used as an alternative if needed. The selected dataset must provide bona fide speech and TTS/VC-generated spoofed audio with standardized training, validation, and evaluation splits.

Data-related tasks:

- [ ] Download the selected dataset locally
- [ ] Review its license and usage requirements
- [ ] Preserve the official train, validation, and evaluation splits
- [ ] Document class counts and attack types for each split
- [ ] Ensure that raw audio and generated features are not committed to Git

### 5.2 Feature extraction

Convert raw audio waveforms into two-dimensional Mel-spectrograms for both model families.

Tasks:

- [ ] Standardize sampling rate and channel format
- [ ] Define clip length and padding/cropping behavior
- [ ] Select Mel-spectrogram parameters
- [ ] Normalize features consistently
- [ ] Save all preprocessing choices in configuration files
- [ ] Verify that train/test preprocessing does not leak information

### 5.3 Model architectures

#### Proposed model: Audio Spectrogram Transformer

Fine-tune an AST model on the extracted Mel-spectrograms.

- [ ] Select and document the pretrained AST checkpoint
- [ ] Define input dimensions and positional-embedding handling
- [ ] Replace or configure the final layer for binary classification
- [ ] Document which layers are frozen or fine-tuned

#### Baseline model: CNN

Implement a lightweight CNN baseline commonly used in audio deepfake detection, such as LCNN or ResNet-18.

- [ ] Select one primary CNN baseline
- [ ] Keep input preprocessing comparable with AST
- [ ] Document parameter counts and training settings

### 5.4 Training strategy

Train both models using binary cross-entropy loss. Optimize AST with AdamW and a learning-rate scheduler. The model output will pass through a sigmoid function to produce a score between 0 and 1 representing the likelihood that an input is bona fide.

Tasks:

- [ ] Define label encoding clearly
- [ ] Select batch size, learning rate, weight decay, and epoch count
- [ ] Configure the learning-rate scheduler
- [ ] Add model checkpointing based on validation performance
- [ ] Set and log random seeds
- [ ] Train each model using five different random seeds
- [ ] Record all experiment configurations and results

### 5.5 Evaluation

The primary evaluation metrics are:

- **Equal Error Rate (EER):** The operating point at which false-positive and false-negative rates are equal
- **Accuracy:** Overall proportion of correctly classified samples

All reported results should include performance across five random seeds and confidence intervals.

Evaluation tasks:

- [ ] Implement EER calculation
- [ ] Implement accuracy calculation
- [ ] Select the classification threshold without using test labels
- [ ] Report the mean and confidence interval across five seeds
- [ ] Compare AST and CNN using the same data splits
- [ ] Save per-run and aggregate results in machine-readable files

Additional metrics such as precision, recall, F1 score, and ROC-AUC may be reported if useful for understanding class imbalance and threshold behavior.

### 5.6 Robustness and ablation studies

Evaluate real-world robustness by measuring both models under conditions not represented during training.

#### Unseen attacks

- [ ] Define which spoofing attack types are withheld from training
- [ ] Ensure that the split prevents attack-type leakage
- [ ] Evaluate AST and CNN on the same unseen attacks
- [ ] Compare performance degradation against the standard test set

#### Background noise

- [ ] Select one or more synthetic background-noise sources
- [ ] Define noise levels or signal-to-noise ratios
- [ ] Apply noise only to evaluation inputs unless otherwise specified
- [ ] Measure performance degradation at each noise level

If model components or preprocessing choices are removed or changed, report those experiments separately as true architectural or preprocessing ablations.

## 6. End-to-end pipeline

```text
ASVspoof / WaveFake dataset
            |
            v
Mel-spectrogram feature extraction
            |
        +---+---+
        |       |
        v       v
       AST     CNN baseline
        |       |
        +---+---+
            |
            v
BCE training with AdamW / configured optimizer
            |
            v
Evaluation across five seeds
  - EER
  - Accuracy
  - Confidence intervals
            |
            v
Robustness evaluation
  - Unseen spoofing attacks
  - Background noise
```

## 7. Milestones and timeline

| Dates | Milestone | Deliverables |
| --- | --- | --- |
| Nov 11-15 | Data preprocessing and setup | Prepare ASVspoof 2019 LA, extract Mel-spectrograms, and establish training and evaluation pipelines |
| Nov 14-22 | Model fine-tuning and evaluation | Fine-tune AST and CNN across five random seeds; evaluate using EER and accuracy |
| Nov 20-24 | Robustness evaluation | Test unseen spoofing attacks and multiple background-noise levels; compute confidence intervals |
| Nov 23-27 | Results analysis and final report | Compare results, analyze findings, create figures, and complete the final report |

## 8. Expected outputs

- Reproducible preprocessing pipeline
- Trained AST and CNN baseline models
- Results from five random seeds
- EER, accuracy, and confidence intervals
- Unseen-attack robustness results
- Background-noise robustness results
- Tables and figures comparing both model families
- Final technical report describing methods, results, limitations, and conclusions

## 9. Reproducibility requirements

- Do not commit datasets, extracted features, model checkpoints, credentials, or private keys
- Keep experiment settings in version-controlled configuration files
- Record package versions and environment setup instructions
- Fix and log random seeds
- Preserve official dataset splits and document any custom splits
- Provide commands for preprocessing, training, evaluation, and robustness testing
- Store summary results in CSV, JSON, or another machine-readable format

## 10. Decisions still to finalize

- [ ] Confirm ASVspoof 2019 LA or WaveFake as the primary dataset
- [ ] Select the CNN baseline: LCNN or ResNet-18
- [ ] Select the pretrained AST checkpoint
- [ ] Define the audio sampling rate and clip duration
- [ ] Define Mel-spectrogram parameters
- [ ] Define the unseen-attack split
- [ ] Select noise sources and signal-to-noise ratios
- [ ] Define the confidence-interval calculation method
- [ ] Assign implementation responsibilities among team members
