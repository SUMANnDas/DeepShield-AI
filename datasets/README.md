# Local Dataset Layout

Dataset content is kept inside this project. CIFAKE v3 is the only third-party dataset currently acquired; other sources remain pending approval.

```text
datasets/
  raw/
    images/       # CIFAKE-v3 archive and extracted train/test images; other data after approval
    video/        # source videos, after acquisition approval
    audio/        # source audio, after acquisition approval
  interim/
    images/       # bounded copied CIFAKE pilot input
    frames/       # sampled video frames
    annotations/  # provider label sidecars copied or normalized for processing
  processed/
    images/       # normalized CIFAKE pilot outputs
    video/        # sampled frames and derived video outputs
    audio/        # mono, resampled audio segments and log-mel arrays
  manifests/      # JSONL source, frame, segment, and split manifests
```

Preprocessing commands refuse to replace existing manifests, SQLite indexes, or media outputs. Use a new output path for each run. Keep the provider's archive and license/terms alongside acquisition records; the research and license notes in `docs/datasets/` are not a substitute for checking the selected release's current terms.
