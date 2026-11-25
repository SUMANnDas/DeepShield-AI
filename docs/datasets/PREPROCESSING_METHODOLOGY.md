# Preprocessing Methodology

## Invariants

1. Raw source media is read-only input. All outputs go to a caller-selected separate directory.
2. Every run requires a new manifest and output paths. Existing targets cause a conflict error; no output is silently replaced.
3. Every sample retains dataset name/version, source-relative path, SHA-256, original label, standardized label, task, and processing status.
4. REAL, FAKE, and UNKNOWN are stored separately from the provider's original label. Dataset-specific method/attack labels are never collapsed.
5. Missing faces, silent segments, decode errors, and corrupt files are recorded; they are not silently removed.
6. The same explicit JSON configuration should be recorded and reused for training-time preparation and inference.

## Images

OpenCV validates decoding and channel count, extracts dimensions, hashes bytes, and computes a 64-bit perceptual hash. A SQLite-backed index checks exact SHA-256 duplicates and approximate perceptual duplicates without retaining the dataset in RAM. When enabled, OpenCV Haar cascades record face detections. Optional crops use an eye-pair rotation when two eyes are found; otherwise a crop-only result is explicitly marked. If no face is found, the resized full image is saved and face_detected=false.

Output is an RGB JPEG at configured dimensions. A utility provides float32 RGB values in [0,1] for downstream preprocessing. No train-time augmentation is applied here. The JSONL manifest stores dimensions, channels, face status/boxes, hashes, duplicate links, output path, and errors.

## Video

OpenCV reads one frame at a time. Uniform sampling is the default and is bounded by max_frames. Scene-change and face-presence modes probe at a configurable stride and fall back to uniform sampling when no candidates are found. Each extracted JPEG has a frame-manifest record with source video ID, original frame number, timestamp, inherited video label, face status and simple IoU-linked track IDs. Decode failures are written as frame error records.

Tracking is a lightweight sampling aid; it is not a validated identity tracker. FFmpeg is not invoked. The installed OpenCV build reports FFmpeg support, but the standalone ffmpeg command is not currently on PATH.

## Audio

PCM WAV files can be decoded with Python's standard library. FLAC requires the optional SoundFile package; without it, the manifest records an actionable dependency error. Channels are mixed to mono when configured; resampling uses SciPy polyphase filtering when available, with linear interpolation as a dependency-free fallback. Segments preserve silence and the last partial segment by default.

For each segment, the pipeline can save a mono PCM16 WAV and a log-mel spectrogram as a NumPy `.npy` array. MFCC extraction is not implemented in this phase. Feature parameters, source/target sampling rates, source channels, segment offsets, duration, RMS/silent flag, and status are stored in JSONL. There is no volume normalization or silence removal by default.

## Scaling

Directory scans and media reads process one file at a time. Video/audio arrays are bounded to one source clip/segment. CSV/JSONL annotations and duplicate/split indexes use SQLite; small JSON-array annotation files are accepted but loaded into memory. Output generation is single-process by default to avoid uncontrolled CPU and disk pressure.
