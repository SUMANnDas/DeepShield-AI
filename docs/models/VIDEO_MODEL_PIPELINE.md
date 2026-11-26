# Video facial manipulation model pipeline

## Scope and readiness

The first video task is visual facial manipulation detection in videos. It is distinct from CIFAKE's 32×32 AI-generated image classification and does not claim to detect arbitrary AI-generated video or audio deepfakes.

The repository has model and preprocessing code, but no real video dataset has been acquired and no trained video checkpoint is registered. The application therefore continues to return metadata-only inspection for uploads; it must not emit a deepfake prediction until an approved real-data checkpoint is trained and validated.

## Manifest to training contract

`VideoSequenceDataset` reads JSONL records containing `relative_path` and a binary `label`. It accepts numeric labels (`0` real, `1` manipulated) or common text labels (`REAL`, `FAKE`, `MANIPULATED`, and related aliases). It reads source identifiers from `source_video_id` or `original_video_id` and preserves them for group audits. Unknown labels, missing videos, and paths outside the dataset root fail explicitly.

Training and inference both use `preprocess_video`: deterministic uniform sampling, one temporally persistent face track, padded face sequences, and the same crop size. The default sample is capped at 12 frames. Frames above 16 megapixels are rejected from metadata before selected-frame decoding. Empty face sequences are excluded from training and evaluation; evaluation reports them as skipped and returns no metrics if none are usable.

The current Haar detector is a local fallback with limited pose, lighting, occlusion, and small-face robustness. Its confidence is unavailable, so results must be interpreted cautiously. No external detector weights are downloaded.

## Model options

- `EfficientNetFrameClassifier`: per-frame baseline; video probability is the mean over valid sampled-frame probabilities.
- `EfficientNetBiGRU`: EfficientNet-B0 frame features, masked bidirectional GRU, temporal attention, and a video-level classifier.
- `EfficientNetTransformerVideoClassifier`: the same spatial encoder with an end-to-end masked temporal Transformer.

All backbones currently use `weights=None`; this work has not downloaded pretrained weights. The 8 GB GPU profiles use batch size 1 and BF16: [efficientnet-bigru-8gb.json](../../configs/training/video/efficientnet-bigru-8gb.json) and [efficientnet-transformer.json](../../configs/training/video/efficientnet-transformer.json). The local RTX 5070 Laptop GPU is visible with an 8,151 MiB total VRAM report; PyTorch 2.15.0+cu132 / Torchvision 0.29.2+cu132 reports CUDA available and BF16 supported. A synthetic 12-frame, 224×224 Transformer forward/backward smoke test produced finite loss and peaked at about 781 MiB allocated VRAM. It did not run an optimizer step or train on data. No real training run or performance claim is recorded here.

## Storage and experiments

Video inputs, manifests, sampled-frame outputs, annotation indexes, run configurations, and checkpoints must remain under `DEEPSHIELD_VIDEO_DATA_ROOT` (default `D:\DeepShield-Data\VideoDetection`). Persistent frame output is disabled by default. The budget target is 30 GB, hard limit is 40 GB, and at least 5 GB must remain free on the target drive. Retain archives and originals; a quota failure can leave a partial generated artifact, which is preserved and is not reusable as a checkpoint.

See [VIDEO_STORAGE_BUDGET.md](../datasets/VIDEO_STORAGE_BUDGET.md) for the acquisition envelope and approval gate.
