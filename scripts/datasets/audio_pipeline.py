"""Streaming audio inspection, mono/resample/segment processing, and log-mel extraction."""

from __future__ import annotations

import logging
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from scripts.datasets.manifest_utils import (
    annotation_for,
    append_jsonl,
    config_hash,
    exclusive_jsonl,
    iter_media_files,
    json_config,
    load_annotations,
    log_file_error,
    relative_posix,
    refuse_existing,
    sample_id,
    sha256_file,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG: dict[str, Any] = {
    "audio_extensions": [".wav", ".flac"],
    "target_sample_rate": 16000,
    "channel_policy": "mono_average",
    "segment_seconds": 4.0,
    "overlap_seconds": 0.5,
    "keep_short_final_segment": True,
    "window_ms": 25.0,
    "hop_ms": 10.0,
    "fft_size": 512,
    "mel_bins": 64,
    "min_frequency": 20.0,
    "max_frequency": 7600.0,
    "pre_emphasis": 0.97,
    "silence_rms_threshold": 0.001,
    "skip_silence": False,
    "write_audio_segments": True,
    "write_log_mel": True,
}


def _decode_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        width = handle.getsampwidth()
        frame_count = handle.getnframes()
        compression = handle.getcomptype()
        if compression != "NONE":
            raise ValueError("unsupported_wave_compression")
        raw = handle.readframes(frame_count)
    if width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        integers = packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8) | (packed[:, 2].astype(np.int32) << 16)
        integers = (integers ^ 0x800000) - 0x800000
        values = integers.astype(np.float32) / 8388608.0
    elif width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported_wave_sample_width")
    if channels <= 0 or sample_rate <= 0 or values.size != frame_count * channels:
        raise ValueError("invalid_wave_metadata")
    return values.reshape(-1, channels), int(sample_rate)


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    """Decode PCM WAV with the standard library or FLAC/other supported files with SoundFile."""
    try:
        import soundfile as sf
    except ImportError:
        if Path(path).suffix.lower() != ".wav":
            raise RuntimeError("soundfile_required_for_non_wav_audio")
        return _decode_wave(Path(path))
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.size == 0 or sample_rate <= 0:
        raise ValueError("empty_or_invalid_audio")
    return np.asarray(samples, dtype=np.float32), int(sample_rate)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample_rate_must_be_positive")
    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(source_rate, target_rate)
        return resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(np.float32)
    except ImportError:
        target_length = max(1, round(len(samples) * target_rate / source_rate))
        source_positions = np.arange(len(samples), dtype=np.float64)
        target_positions = np.arange(target_length, dtype=np.float64) * source_rate / target_rate
        return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _hz_to_mel(value: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def log_mel_spectrogram(samples: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
    window_size = max(16, int(round(sample_rate * float(config["window_ms"]) / 1000.0)))
    hop_size = max(1, int(round(sample_rate * float(config["hop_ms"]) / 1000.0)))
    fft_size = max(window_size, int(config["fft_size"]))
    mel_bins = max(1, int(config["mel_bins"]))
    emphasis = float(config.get("pre_emphasis", 0.0))
    signal = samples.astype(np.float32, copy=False)
    if signal.size == 0:
        return np.zeros((mel_bins, 0), dtype=np.float32)
    if emphasis:
        signal = np.concatenate((signal[:1], signal[1:] - emphasis * signal[:-1]))
    if signal.size < window_size:
        signal = np.pad(signal, (0, window_size - signal.size))
    frame_count = 1 + max(0, (signal.size - window_size) // hop_size)
    shape = (frame_count, window_size)
    strides = (signal.strides[0] * hop_size, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()
    frames *= np.hanning(window_size).astype(np.float32)[None, :]
    spectrum = np.abs(np.fft.rfft(frames, n=fft_size, axis=1)).astype(np.float32) ** 2

    min_hz = max(0.0, float(config["min_frequency"]))
    max_hz = min(float(config["max_frequency"]), sample_rate / 2.0)
    mel_points = np.linspace(_hz_to_mel(min_hz), _hz_to_mel(max_hz), mel_bins + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    filterbank = np.zeros((mel_bins, fft_size // 2 + 1), dtype=np.float32)
    for band in range(mel_bins):
        left, center, right = bins[band:band + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for index in range(max(0, left), min(center, filterbank.shape[1])):
            filterbank[band, index] = (index - left) / max(1, center - left)
        for index in range(max(0, center), min(right, filterbank.shape[1])):
            filterbank[band, index] = (right - index) / max(1, right - center)
    mel = spectrum @ filterbank.T
    return np.log(np.maximum(mel, 1e-10)).T.astype(np.float32)


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    refuse_existing(path)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    with path.open("xb") as raw:
        with wave.open(raw, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())


def process_audio_directory(
    input_root: Path,
    output_dir: Path,
    manifest_path: Path,
    *,
    dataset_name: str,
    dataset_version: str | None = None,
    source_url: str | None = None,
    dataset_license: str | None = None,
    citation: str | None = None,
    default_label: str = "UNKNOWN",
    annotations_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, int]:
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Audio input directory does not exist: {root}")
    output_root = Path(output_dir)
    manifest = Path(manifest_path)
    refuse_existing(manifest)
    config = json_config(config_path, DEFAULT_CONFIG)
    digest = config_hash(config)
    annotation_path = manifest.with_suffix(".annotations.sqlite3")
    annotations = load_annotations(annotations_path, annotation_path if annotations_path else None)
    output_root.mkdir(parents=True, exist_ok=True)
    stats = {"audio_files": 0, "segments": 0, "silent_segments": 0, "errors": 0}
    target_rate = int(config["target_sample_rate"])
    segment_length = max(1, int(round(float(config["segment_seconds"]) * target_rate)))
    overlap = int(round(float(config["overlap_seconds"]) * target_rate))
    step = segment_length - overlap
    if step <= 0:
        if hasattr(annotations, "close"):
            annotations.close()
        raise ValueError("overlap_seconds must be shorter than segment_seconds")
    with exclusive_jsonl(manifest) as out:
        for path in iter_media_files(root, config["audio_extensions"]):
            relative = relative_posix(path, root)
            audio_id = sample_id(dataset_name, relative)
            label = annotation_for(annotations, relative, default_label)
            file_hash: str | None = None
            try:
                file_hash = sha256_file(path)
                samples, original_rate = decode_audio(path)
                channels = int(samples.shape[1]) if samples.ndim == 2 else 1
                if samples.ndim == 2:
                    if config.get("channel_policy") != "mono_average":
                        raise ValueError("only_mono_average_channel_policy_is_supported")
                    samples = samples.mean(axis=1, dtype=np.float32)
                samples = resample_audio(samples, original_rate, target_rate)
                starts = range(0, len(samples), step) if len(samples) else range(0)
                for segment_index, start in enumerate(starts):
                    end = min(start + segment_length, len(samples))
                    is_short_final = end - start < segment_length
                    if is_short_final and not config.get("keep_short_final_segment", True):
                        continue
                    segment = samples[start:end].copy()
                    rms = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64)))) if segment.size else 0.0
                    silent = rms < float(config.get("silence_rms_threshold", 0.001))
                    sample_id_value = f"{audio_id}_s{segment_index:06d}"
                    record: dict[str, Any] = {
                        "sample_id": sample_id_value,
                        "audio_sample_id": audio_id,
                        "dataset_name": dataset_name,
                        "dataset_version": dataset_version,
                        "source_url": source_url,
                        "dataset_license": dataset_license,
                        "citation": citation,
                        "task": label.get("task") or "synthetic_speech",
                        "relative_path": relative,
                        "file_hash": file_hash,
                        "label": label["label"],
                        "original_label": label["original_label"],
                        "source_split": label.get("source_split"),
                        "manipulation_type": label.get("manipulation_type"),
                        "speaker_id": label.get("speaker_id") or label.get("identity_id"),
                        "sample_rate": original_rate,
                        "target_sample_rate": target_rate,
                        "channels": channels,
                        "duration_seconds": len(samples) / target_rate,
                        "segment_index": segment_index,
                        "segment_start_seconds": start / target_rate,
                        "segment_end_seconds": end / target_rate,
                        "rms": rms,
                        "silent": silent,
                        "audio_path": None,
                        "feature_path": None,
                        "preprocessing_config_hash": digest,
                        "preprocessing_status": "pending",
                        "error_code": None,
                    }
                    if silent and config.get("skip_silence", False):
                        record["preprocessing_status"] = "skipped_silence"
                        stats["silent_segments"] += 1
                    else:
                        if config.get("write_audio_segments", True):
                            audio_output = output_root / "audio" / f"{sample_id_value}.wav"
                            audio_output.parent.mkdir(parents=True, exist_ok=True)
                            _write_pcm16(audio_output, segment, target_rate)
                            record["audio_path"] = audio_output.relative_to(output_root).as_posix()
                        if config.get("write_log_mel", True):
                            feature_output = output_root / "features" / f"{sample_id_value}.npy"
                            feature_output.parent.mkdir(parents=True, exist_ok=True)
                            refuse_existing(feature_output)
                            with feature_output.open("xb") as handle:
                                np.save(handle, log_mel_spectrogram(segment, target_rate, config), allow_pickle=False)
                            record["feature_path"] = feature_output.relative_to(output_root).as_posix()
                        record["preprocessing_status"] = "processed"
                        stats["segments"] += 1
                    append_jsonl(out, record)
                stats["audio_files"] += 1
            except FileExistsError:
                raise
            except Exception as exc:
                stats["errors"] += 1
                log_file_error(LOGGER, "audio", relative, exc)
                append_jsonl(out, {
                    "sample_id": audio_id,
                    "audio_sample_id": audio_id,
                    "dataset_name": dataset_name,
                    "relative_path": relative,
                    "file_hash": file_hash,
                    "label": label["label"],
                    "original_label": label["original_label"],
                    "source_split": label.get("source_split"),
                    "preprocessing_config_hash": digest,
                    "preprocessing_status": "corrupt" if isinstance(exc, (ValueError, wave.Error)) else "error",
                    "error_code": str(exc)[:160],
                })
    if hasattr(annotations, "close"):
        annotations.close()
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version")
    parser.add_argument("--source-url")
    parser.add_argument("--dataset-license")
    parser.add_argument("--citation")
    parser.add_argument("--label", default="UNKNOWN")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(process_audio_directory(
        args.input, args.output, args.manifest, dataset_name=args.dataset_name,
        dataset_version=args.dataset_version, source_url=args.source_url,
        dataset_license=args.dataset_license, citation=args.citation,
        default_label=args.label, annotations_path=args.annotations, config_path=args.config,
    ))


if __name__ == "__main__":
    main()
