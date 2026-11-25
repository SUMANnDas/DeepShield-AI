# Dataset Manifest Schema

Manifests are UTF-8 JSON Lines files written incrementally. Null is used for unavailable metadata; unknown values are not guessed.

## Common fields

- sample_id: deterministic ID derived from dataset name and source-relative path.
- dataset_name, dataset_version, source_url, dataset_license, citation.
- relative_path and file_hash (SHA-256 of original bytes).
- label: REAL, FAKE, or UNKNOWN.
- original_label and manipulation_type: provider values preserved verbatim.
- task: for example ai_generated_image, face_manipulation, or synthetic_speech.
- source_split: optional provider-published partition such as train, dev, eval, or test, retained verbatim for audit and to preserve official partitions.
- split and split_group_id: absent until split generation; both record the assignment and the selected grouping key. If provider partitions are present and mapped in the split configuration, those assignments take precedence over new random allocation.
- preprocessing_status and error_code; errors omit absolute paths and secrets.
- preprocessing_config_hash: SHA-256 of canonical JSON configuration.

## Image fields

image_width, image_height, color_channels, face_detected, face_detection_status, face_boxes, alignment_status, perceptual_hash, duplicate_of, output_files.

## Video and extracted-frame fields

video_sample_id, original_video_id, source_identity_id if available, duration_seconds, frame_rate, frame_count, width, height, fourcc, extracted_frame_count. Each frame row also contains source_relative_path, video_sample_id, frame_number, timestamp_seconds, face_detected, face_boxes, track_ids, and output_path.

## Audio and segment fields

audio_sample_id, speaker_id if available, sample_rate, target_sample_rate, channels, duration_seconds, segment_index, segment_start_seconds, segment_end_seconds, rms, silent, audio_path, feature_path.

Never put names, email addresses, credentials, or unrelated personal details into manifests. Provider identifiers required for split integrity should be retained only as the provider's pseudonymous/source ID.
