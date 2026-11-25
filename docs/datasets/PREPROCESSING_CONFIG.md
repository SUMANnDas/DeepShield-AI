# Preprocessing Configuration

Versioned JSON defaults are stored in configs/datasets/:

- image.json: target dimensions, accepted extensions, face detection/crop controls, perceptual-hash threshold, JPEG quality, and normalization convention.
- video.json: allowed extensions, sampling strategy, frame cap, probe cadence, scene threshold, face detector choice, and output quality.
- audio.json: target rate/channel policy, segmentation duration/overlap, last-segment policy, log-mel parameters, and pre-emphasis.
- splits.json: seed, ratios, group field, and label field.

Input/output paths are CLI arguments rather than hardcoded config values. Each manifest stores a hash of the effective configuration. Store a copy of the config alongside each experiment or manifest; do not mutate an old config in place when reproducing prior outputs.
