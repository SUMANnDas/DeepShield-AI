# Phase 4 — Video facial manipulation detection

## Scope and data status

This branch detects potential facial manipulation in video and is separate from Phase 3 CIFAKE AI-generated still-image classification. FaceForensics++ c23 has **not** been downloaded or processed, and no real-data training or evaluation has occurred. The branch was tested with locally generated synthetic AVI fixtures only. Predictions are model-based forensic indicators, never proof of authenticity.

Official sources to recheck before acquisition: [FaceForensics++ repository](https://github.com/ondyari/FaceForensics), [dataset instructions](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md), and [terms of service](https://kaldir.vc.in.tum.de/faceforensics/webpage/FaceForensics_TOS.pdf). Access must be granted by the provider. The terms restrict use to non-commercial research/education; verify current terms and obtain explicit acquisition approval before downloading. No mirror, scraping, substitute dataset, or external upload is permitted.

## Processing and architecture

`video_metadata.py` reports hash, duration, FPS, dimensions, frame count, codec and decode status. OpenCV/FFmpeg is used locally; it does not reliably expose audio stream presence, so that field remains unknown. Invalid and truncated inputs report errors. `frame_sampling.py` selects deterministic, unique, uniformly spaced frame indices (default cap 12) with timestamps. `face_detection.py` provides a replaceable adapter and local OpenCV Haar cascade. Haar detection is a low-accuracy fallback, especially under pose, occlusion, compression, and small faces. `face_tracking.py` greedily associates boxes by IoU within a clip; track IDs do not identify people. `face_cropping.py` validates, pads, clips, converts/resizes face crops; source media is not modified.

The frame baseline is EfficientNet-B0 initialized with `weights=None` and a two-class head. The primary temporal model encodes valid frames through EfficientNet-B0, then applies a bidirectional two-layer GRU (hidden size 256), masked attention pooling, and a REAL/MANIPULATED head. Padding is excluded from frame encoding and attention. A lightweight optional Transformer accepts spatial features with learned positions and key padding mask. Default configs cap the pilot at 100 videos, 12 sampled frames and 12 crops per video, c23 only. No all-frame extraction is performed.

## Data integrity and experiments

JSONL records include video/source IDs, relative path, binary label, manipulation method, compression, file hash, metadata and decode status. `grouped_split()` assigns all records sharing a source video to one split; `check_leakage()` checks video ID, source ID and hash across splits. Synthetic fixture tests exercise sampling, metadata, cropping, tracking, splits, model tensor shapes, masked padding, backward pass and reports. For actual use, manifests must preserve original-to-manipulated source relationships and include all manipulation categories. The local JSON/CSV-style experiment approach records a unique run ID and configuration; users should add manifest hashes and runtime metadata when launching real experiments.

Training utilities provide finite-loss checks, gradient clipping, validation-loss checkpoint selection, exclusive checkpoint creation, and deterministic seeds. Use conservative sequence batch size 2 on the 8 GB RTX 5070 Laptop GPU; benchmark memory before increasing it. AMP is off by default. Checkpoints never overwrite. `metrics.py` reports accuracy, precision, recall, F1, ROC-AUC and confusion matrix. Report manipulation-method strata separately; source-video grouped test results are required. Frame-level and video-level metrics are distinct.

Inference returns per-frame scores/timestamps and a structured JSON report with metadata and preprocessing details. The backend service is an integration boundary; it accepts a caller-provided temporary file and model, enforces a byte limit, and does not retain uploads. The current crop score heatmap is a visualization only; temporal attention is not causal explanation. No endpoint is mounted into the existing API yet.

## Reproduction and limitations

Use Python 3.12 project `.venv`, installed PyTorch/Torchvision/OpenCV, configuration under `configs/training/video/`, and run CPU tests with `python -m unittest discover -s backend/tests`. Real training remains blocked on provider access and separate user authorization. FaceForensics++ is an older benchmark; c23 compression, dataset bias, face-detector failures and source leakage can affect results. Performance on FF++ does not establish performance on unseen generators or datasets. Do not market this as a universal deepfake detector.

## Acquisition checklist

Before requesting download authorization: (1) user confirms provider access was granted; (2) recheck official source, current terms and attribution; (3) record selected version/category and c23-only archive sizes; (4) verify destination and free space; (5) estimate archive, extracted, sampled-crop and manifest storage; (6) create a unique destination and verify no conflicts. Request separate approval before each acquisition. Preserve archives and originals; do not silently expand the 100-video cap.
