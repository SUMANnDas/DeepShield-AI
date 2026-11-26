# Phase 4 — Video facial manipulation detection

## Scope and data status

This branch detects potential facial manipulation in video and is separate from Phase 3 CIFAKE AI-generated still-image classification. Predictions are model-based forensic indicators, never proof of authenticity.

On 2026-11-26, a 1,200-video double-disjoint subset of `RealTimeFaceSwap-10k` (`stplusplus/rtfs-10k`, `CC BY-SA 4.0`) was acquired via the official Kaggle API into `C:\Users\User\DeepShield-Data\VideoDetection\raw\RealTimeFaceSwap-10k` (`840` train / `180` validation / `180` test; `0` decode errors; `0` cross-split YouTube ID or source-photo overlap). Two scratch-initialized models (`weights=None`, BF16 on RTX 5070 Laptop GPU) were trained and evaluated once on the locked 179-usable-clip held-out test set (`1` real clip had no Haar face track):

- **EfficientNet-B0 Frame Baseline** (`efficientnet-b0-frame-20261126T101514Z-b58150ca`, epoch 13): Accuracy `0.8045`, Precision `0.7311`, Recall `0.9667`, F1 `0.8325`, ROC-AUC `0.8720`, PR-AUC `0.8373`, confusion matrix `[[57, 32], [3, 87]]`; strata detection `100.0%` (`45/45`, AUC `0.9184`) on `inswapper` and `93.33%` (`42/45`, AUC `0.8257`) on `uniface`.
- **EfficientNet-B0 + BiGRU (Attention)** (`efficientnet-b0-bigru-attention-20261126T094052Z-02645f13`, epoch 13): Accuracy `0.7709`, Precision `0.7248`, Recall `0.8778`, F1 `0.7940`, ROC-AUC `0.8280`, PR-AUC `0.7923`, confusion matrix `[[59, 30], [11, 79]]`; strata detection `91.11%` (`41/45`, AUC `0.8675`) on `inswapper` and `84.44%` (`38/45`, AUC `0.7884`) on `uniface`.

FaceForensics++ c23 remains unacquired pending an authorized download script; its official sources remain documented below.

Official sources to recheck before acquisition: [FaceForensics++ repository](https://github.com/ondyari/FaceForensics), [dataset instructions](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md), and [terms of service](https://kaldir.vc.in.tum.de/faceforensics/webpage/FaceForensics_TOS.pdf). The user reports provider access is granted and confirmed non-commercial research use. Verify the authorized c23 listing, current terms, exact sizes, destination and free space; obtain separate download approval before downloading. No mirror, scraping, substitute dataset, or external upload is permitted.

## Processing and architecture

`video_metadata.py` reports hash, duration, FPS, dimensions, frame count, codec and decode status. OpenCV/FFmpeg is used locally; it does not reliably expose audio stream presence, so that field remains unknown. Invalid and truncated inputs report errors. `frame_sampling.py` selects deterministic, unique, uniformly spaced frame indices (default cap 12) with timestamps. `face_detection.py` provides a replaceable adapter and local OpenCV Haar cascade. Haar detection is a low-accuracy fallback, especially under pose, occlusion, compression, and small faces. `face_tracking.py` greedily associates boxes by IoU within a clip; track IDs do not identify people. `face_cropping.py` validates, pads, clips, converts/resizes face crops; source media is not modified.

The frame baseline is EfficientNet-B0 initialized with `weights=None` and a two-class head. The primary temporal model encodes valid frames through EfficientNet-B0, then applies a bidirectional two-layer GRU (hidden size 256), masked attention pooling, and a REAL/MANIPULATED head. Padding is excluded from frame encoding and attention. A lightweight optional Transformer accepts spatial features with learned positions and key padding mask. Default configs cap the pilot at 100 videos, 12 sampled frames and 12 crops per video, c23 only. No all-frame extraction is performed.

## Data integrity and experiments

JSONL records include video/source IDs, relative path, binary label, manipulation method, compression, file hash, metadata and decode status. `grouped_split()` assigns all records sharing a source video to one split; `check_leakage()` checks video ID, source ID and hash across splits. Synthetic fixture tests exercise sampling, metadata, cropping, tracking, splits, model tensor shapes, masked padding, backward pass and reports. For actual use, manifests must preserve original-to-manipulated source relationships and include all manipulation categories. The local JSON/CSV-style experiment approach records a unique run ID and configuration; users should add manifest hashes and runtime metadata when launching real experiments.

Training utilities provide finite-loss checks, gradient clipping, validation-loss checkpoint selection, exclusive checkpoint creation, and deterministic seeds. The current 8 GB RTX 5070 Laptop GPU profiles use batch size 1 and BF16; measure memory before increasing batch size. Checkpoints never overwrite. `metrics.py` reports accuracy, precision, recall, F1, ROC-AUC and confusion matrix. Report manipulation-method strata separately; source-video grouped test results are required. Frame-level and video-level metrics are distinct.

Inference returns per-frame scores/timestamps and a structured JSON report with metadata and preprocessing details. The mounted backend upload endpoint currently performs metadata/frame inspection only and explicitly reports prediction as unavailable. It enforces upload/storage limits and does not retain uploads. The current crop score heatmap is a visualization only; temporal attention is not causal explanation.

## Reproduction and limitations

Use Python 3.12 project `.venv`, installed PyTorch/Torchvision/OpenCV, configuration under `configs/training/video/`, and run CPU tests with `python -m unittest discover -s backend/tests`. Real training remains blocked on the authorized c23 file listing, measured-size preflight, and separate download approval. FaceForensics++ is an older benchmark; c23 compression, dataset bias, face-detector failures and source leakage can affect results. Performance on FF++ does not establish performance on unseen generators or datasets. Do not market this as a universal deepfake detector.

## Acquisition checklist

Before download: (1) verify the user's reported provider access; (2) recheck official source, current terms and attribution; (3) inspect the authorized c23-only file listing and exact byte sizes; (4) verify destination and free space; (5) estimate retained archive, extracted media, samples, manifests, and checkpoint storage against the 100 GB hard cap; (6) present the measured plan and obtain separate approval. Preserve archives and originals through training; do not silently expand the 100-video pilot cap or automate cleanup afterward.
