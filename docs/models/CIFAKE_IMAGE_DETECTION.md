# Phase 3 — CIFAKE Image Detection

**Status:** implementation, training, isolated test evaluation, and validation-sample explainability completed on 2026-11-25.

This branch detects **AI-generated images versus CIFAKE real images**. It does not detect face swaps, facial reenactment, or video deepfakes. Results describe performance on the supplied CIFAKE v3 split only; they are not evidence of general deepfake detection.

## Dataset and split control

The loader consumes the final approved duplicate-safe JSONL manifests and validates CIFAKE version, task, split, class label, 32×32 dimensions, paths, and `preprocessing_status`. It rejects path traversal, duplicate manifest paths or sample IDs, missing images, and cross-split group or file-hash overlap between train and validation. The final leakage report covers all three splits.

| Split | Samples | REAL | FAKE | Use |
| --- | ---: | ---: | ---: | --- |
| Train | 96,000 | 48,000 | 48,000 | Gradient updates and train-only normalization statistics |
| Validation | 12,000 | 6,000 | 6,000 | Model selection, early stopping, and validation-sample explanations |
| Test | 12,000 | 6,000 | 6,000 | One final evaluation per locked model |

Manifest identifiers (SHA-256):

- Train: `4f503083419cc48ad6ff8da7d2740a5931d2b38ae3d11710dfddde502d62fbef`
- Validation: `9af4310d4b1bf41bac6610696415aecfc9abea2888341c5a8158968ad4454937`
- Test: `66a7a079df08e38a4a5985acaac6a10292bb1c01ba467ac31d7bfff5ebd057a8`
- Leakage report: `234269083e0980df4097716cd62008ee65cf83481ae42f5538b8348d001c4651`

These hashes match the approved dataset records at Phase 3 completion. The manifests, raw images, annotations, split outputs, and leakage reports were read-only throughout implementation and training.

## Architectures

All models were initialized from scratch. No pretrained weights or additional datasets were downloaded.

| Model | CIFAKE adaptation |
| --- | --- |
| ResNet50 baseline | Replaced the first 7×7 stride-2 convolution with a 3×3 stride-1 convolution, removed the initial max-pooling layer, and replaced the classifier with two logits. |
| EfficientNet-B0 | Changed the stem convolution to stride 1 and replaced the classifier with two logits. |
| Small Vision Transformer | 32×32 input, 4×4 patches (8×8 grid, 64 patches), 6 transformer layers, 3 heads, 192-wide embeddings, and 768-wide MLP. |

## Training and reproducibility

- Configuration: batch size 128; seed 42; deterministic algorithms enabled; two data-loader workers; AdamW (`lr=0.0003`, weight decay `0.0001`); cosine annealing; gradient clipping at 1.0; no mixed precision; 30-epoch maximum.
- Train augmentation: random horizontal flip with probability 0.5 and padded random crop with 2-pixel padding. Validation and test use deterministic, non-augmented transforms.
- Channel means and standard deviations are calculated from train images only and saved per run.
- Validation ROC-AUC controls best-checkpoint selection and early stopping (`patience=5`, minimum improvement `0.0005`). Test data is not loaded during training or model selection.
- Each epoch is written to a distinct checkpoint file. Checkpoints include optimizer, scheduler, scaler, RNG, software, configuration, and manifest identifiers. Resume is restricted to the exact run configuration.
- Pilot: each architecture completed a 3-epoch pilot, then resumed from its own checkpoint. Final test evaluation used the recorded best validation checkpoint exactly once per model.
- Verification: all 35 backend and image-branch tests passed; 19 Python files passed in-memory syntax compilation; `pip check` reported no broken requirements.

Immutable experiment configurations:

- [ResNet50](../../configs/training/image/cifake-resnet50.json)
- [EfficientNet-B0](../../configs/training/image/cifake-efficientnet-b0.json)
- [Small ViT](../../configs/training/image/cifake-vit-small.json)

## Final held-out test results

The fixed decision threshold was 0.5, with `FAKE` as the positive class. Each model was evaluated once on the same 12,000 test samples.

| Model | Selected epoch | Train epochs | Accuracy | Precision (FAKE) | Recall (FAKE) | F1 (FAKE) | Macro-F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | 16 | 21 | 0.9723 | 0.9672 | 0.9778 | 0.9725 | 0.9723 | 0.9963 |
| EfficientNet-B0 | 15 | 20 | 0.9661 | 0.9665 | 0.9657 | 0.9661 | 0.9661 | 0.9949 |
| Small ViT | 25 | 30 | 0.9628 | 0.9737 | 0.9513 | 0.9624 | 0.9628 | 0.9946 |

Confusion matrices use rows=true `[REAL, FAKE]`, columns=predicted `[REAL, FAKE]`:

- ResNet50: `[[5801, 199], [133, 5867]]`
- EfficientNet-B0: `[[5799, 201], [206, 5794]]`
- Small ViT: `[[5846, 154], [292, 5708]]`

The results make ResNet50 the strongest of these three CIFAKE runs on accuracy, macro-F1, and ROC-AUC. This is a comparison on this fixed split, not a broader claim about image generators or deepfakes.

Per-run reports also include weighted F1, specificity, sensitivity, false-positive and false-negative rates, cross-entropy loss, full-precision metrics, per-sample predictions, confusion-matrix plots, and ROC plots. Test evaluation events record the single test-manifest hash and that test data was not used for selection.

## Explainability and inference

Grad-CAM was generated for two REAL and two FAKE **validation** examples for each CNN. ViT uses patch-grid gradient×input attribution and is not labeled Grad-CAM. The artifacts and sample metadata live in each run's `explainability/` directory. Attribution is coarse at 32×32 and is not forensic proof.

- [ResNet50 validation explanations](../../experiments/runs/image_detection/cifake-v3/resnet50-20261124T152708Z-d924cf0d/explainability/)
- [EfficientNet-B0 validation explanations](../../experiments/runs/image_detection/cifake-v3/efficientnet-b0-20261124T164138Z-3fb11f07/explainability/)
- [ViT patch attributions](../../experiments/runs/image_detection/cifake-v3/vit-small-20261125T052835Z-66479d18/explainability/)

Single-image inference accepts an image path or PIL image and returns REAL/FAKE probabilities, the raw model label, model identifier, and measured inference duration. A shared preparation path applies EXIF orientation, preserves aspect ratio, and uses the run's training-mean padding when converting non-native dimensions to 32×32. Native 32×32 input is passed through unchanged. Non-32×32 source images receive an `INCONCLUSIVE` assessment; the raw score is exploratory and outside the validated input domain. The API marks `confidence_calibrated` as false.

## GPU and software environment

The run used an NVIDIA GeForce RTX 5070 Laptop GPU (compute capability 12.0, driver 616.56, 8,123 MiB reported memory) through the project's Windows `.venv`. CUDA was visible to PyTorch. One-batch CUDA forward/backward smoke checks passed for all models, and checkpoint save/load passed. Peak training allocation was approximately 3,513 MiB for ResNet50, 985 MiB for EfficientNet-B0, and 802 MiB for ViT.

Recorded versions: Python 3.12.4; torch 2.15.0+cu132; torchvision 0.29.2+cu132; CUDA runtime 13.2; NumPy 2.5.5; scikit-learn 1.9.1; Pillow 13.0.0; Matplotlib 3.11.2; OpenCV 4.14.0.94; tqdm 4.70.1. `pip check` reported no broken requirements. The complete package inventory and host details are in [the environment snapshot](../../experiments/runs/image_detection/cifake-v3/environment/phase3-environment-20261125T060548Z-504483.json).

## Run records and generated artifacts

The unique run directories contain configuration snapshots, normalization values, software/GPU metadata, append-only JSONL and CSV histories, training curves, validation-selected checkpoint history, final test metrics, test predictions, confusion-matrix and ROC plots, and validation explainability images. The three training runs produced 7,877,870,811 bytes (about 7.34 GiB) of immutable epoch checkpoints. Two standalone one-batch smoke checkpoint files add 188,666,174 bytes, bringing the checkpoint tree to 8,066,536,985 bytes (about 7.51 GiB). Two corresponding uniquely named smoke run directories are empty. At measurement time the project volume had over 237 GB free.

Implementation and test files created for this branch:

- `scripts/training/image/{__init__.py,utils.py,transforms.py,data.py,models.py,metrics.py,checkpoint.py,experiment.py,train.py,evaluate.py,explain.py,inference.py}`
- `configs/training/image/{cifake-resnet50.json,cifake-efficientnet-b0.json,cifake-vit-small.json}`
- `backend/tests/{test_image_manifest_loader.py,test_image_models.py,test_image_training.py,test_image_metrics.py,test_image_checkpoint.py,test_image_explainability.py,test_image_inference.py}`
- `docs/models/CIFAKE_IMAGE_DETECTION.md`
- `README.md` received an appended Phase 3 section; earlier content was retained.

## Limitations and execution notes

CIFAKE images are only 32×32 pixels. This discards fine texture and facial detail before classification and limits what any model or heatmap can explain. The branch therefore addresses low-resolution synthetic-image classification, not facial manipulation, video deepfake detection, high-resolution forensic analysis, or generalization to unseen generators. High test scores on CIFAKE do not remove those limits.

The first resume attempt exposed a CUDA RNG-state device-mapping issue. The restore code now moves saved RNG byte tensors back to CPU before restoring CUDA generators; the CUDA checkpoint regression test passes, and all runs resumed from their recorded checkpoints. The application now checks that the registry checkpoint, held-out evaluation record, selected epoch, run snapshot, and architecture agree before loading the registered model. No external-generator evaluation is recorded for these runs.
