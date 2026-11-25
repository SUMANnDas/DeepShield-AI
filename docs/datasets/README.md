# Dataset Research and Tooling

## Research

- [Recommended staged combination and storage](DATASET_SELECTION.md)
- [Image datasets](IMAGE_DATASETS.md)
- [Video datasets](VIDEO_DATASETS.md)
- [Audio datasets](AUDIO_DATASETS.md)
- [Access and license register](DATASET_LICENSES.md)
- [Controlled pilot acquisition plan](PILOT_ACQUISITION_PLAN.md)
- [Dataset environment and dependency verification](ENVIRONMENT_SETUP.md)

After CIFAKE v3 acquisition and validation, about 235.16 GiB remained free on C: and 66.93 GiB on D:. No approved project dataset destination exists on D:, so data remains inside the project on C:. CIFAKE v3 is acquired; FaceForensics++ c23 and ASVspoof 2019 LA remain pending separate approval. Their estimated source payloads plus CIFAKE's actual ZIP size total about 17.23 GB; reserve about 40 GB for extracted copies, bounded pilot outputs, and headroom. See the acquisition plan for exact paths, results, and licensing/access caveats.

## Processing and integrity

- [Preprocessing methodology](PREPROCESSING_METHODOLOGY.md)
- [Manifest schema](MANIFEST_SCHEMA.md)
- [Preprocessing configuration](PREPROCESSING_CONFIG.md)
- [Leakage prevention](DATA_LEAKAGE_PREVENTION.md)
- Local data structure: [datasets/README.md](../../datasets/README.md)

Main modules are `scripts/datasets/image_pipeline.py`, `video_pipeline.py`, `audio_pipeline.py`, `split_pipeline.py`, `leakage_pipeline.py`, and `report_pipeline.py`. Command-line entry points include `inspect_datasets.py`, `validate_images.py`, `preprocess_images.py`, `preprocess_videos.py`, `extract_video_frames.py`, `preprocess_audio.py`, `generate_spectrograms.py`, `generate_manifests.py`, `create_splits.py`, `check_data_leakage.py`, and `generate_dataset_report.py`.

From the project root, use for example:

```powershell
python scripts/datasets/inspect_datasets.py datasets/raw
python scripts/datasets/preprocess_images.py --input datasets/raw/images --output datasets/processed/images/run-001 --manifest datasets/manifests/images-run-001.jsonl --dataset-name CIFAKE --annotations path/to/labels.csv --config configs/datasets/image.json
python scripts/datasets/preprocess_videos.py --input datasets/raw/video --output datasets/processed/video/run-001 --manifest datasets/manifests/video-run-001.jsonl --dataset-name FaceForensicsPP --config configs/datasets/video.json
python scripts/datasets/preprocess_audio.py --input datasets/raw/audio --output datasets/processed/audio/run-001 --manifest datasets/manifests/audio-run-001.jsonl --dataset-name ASVspoof2019-LA --config configs/datasets/audio.json
python scripts/datasets/create_splits.py --manifest datasets/manifests/audio-run-001.jsonl --output-prefix datasets/manifests/audio-run-001 --config configs/datasets/splits.json
python scripts/datasets/check_data_leakage.py --manifests datasets/manifests/audio-run-001_train.jsonl datasets/manifests/audio-run-001_validation.jsonl datasets/manifests/audio-run-001_test.jsonl --report datasets/reports/generated/audio-leakage.json
```

Do not use the placeholder `path/to/labels.csv` literally. Supply a source annotation file in the project tree. Commands refuse to replace an existing manifest, SQLite index, or media output; choose a new run name for each attempt.

`requirements-datasets.txt` declares optional dataset tooling. The approved packages have been installed into the project `.venv` only; see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for versions and verification. FLAC decoding requires SoundFile; PCM WAV does not.

## Tests

Run `.venv\Scripts\python.exe -m unittest discover -s backend/tests -v` from the project root on Windows. Tests create small synthetic fixtures under `backend/tests/generated/`; they do not download or modify real datasets.
