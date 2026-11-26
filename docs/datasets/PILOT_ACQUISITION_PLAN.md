# Controlled Pilot Acquisition Plan

**Status:** CIFAKE v3 was downloaded and validated after the user's explicit approval. FaceForensics++ c23 has not been downloaded; the user reports provider access is granted and confirmed non-commercial research use, but exact package sizing and separate download approval remain pending. ASVspoof 2019 LA remains unacquired and unauthorized.

## Capacity and destination

The project is located on C:. A recent preflight measured about 220.96 GiB free on C:; recheck before any acquisition. Video data will use the dedicated root `C:\Users\User\DeepShield-Data\VideoDetection`, outside the repository, with a 90 GB target, 100 GB hard cap, and 10 GB minimum free-space reserve. That root does not exist yet. The user plans to delete the video dataset after training; no automatic cleanup is configured. CIFAKE remains in its existing project location and is untouched. CIFAKE's archive plus extracted source uses 220,348,705 bytes including its acquisition metadata and Kaggle snapshot; all retained CIFAKE manifests, indexes, samples, and outputs currently use about 2.32 GB. The latter includes preserved exploratory split runs.

| Dataset | Proposed exact raw destination | Download/extracted storage estimate | Proposed pilot output |
|---|---|---|---|
| CIFAKE v3 (downloaded 2026-11-24) | `C:\Users\User\Desktop\DeepShield-AI\datasets\raw\images\CIFAKE-v3` (`archive.zip` retained; `train/` and `test/` extracted beneath it) | Actual ZIP: 109,625,224 bytes; extracted source: 110,717,853 bytes; raw archive + source + metadata: 220,344,730 bytes. | 400-image pilot: 100 per class in each published train/test partition; 32x32 RGB; 400 processed outputs, all readable, 0 corrupt. |
| FaceForensics++ c23 | `C:\Users\User\DeepShield-Data\VideoDetection\raw\FaceForensics++-c23` | Official README estimates about 10 GB for c23 videos, but exact authorized package/archive and extracted sizes are unknown until its file listing is inspected. Preflight all retained copies and outputs against the 100 GB hard cap; stop if over the 90 GB target until a smaller c23-only subset is approved. | Stage at most 100 videos only after acquisition approval; uniformly sample at most 12 frames/video, capped at 1,200 frames. Persistent full-frame extraction is disabled. |
| ASVspoof 2019 LA | `C:\Users\User\Desktop\DeepShield-AI\datasets\raw\audio\ASVspoof2019-LA` (retain `LA.zip` here; extract to `release\`) | Official record lists `LA.zip` at 7.12 GB; extracted FLAC is estimated at 7.1–7.5 GB. Keep both archive and extraction; reserve 15 GB including pilot copies and metadata. | Stage 500 utterances balanced across bona-fide/spoof labels and drawn from official train/dev/eval partitions; preserve all source partition and speaker IDs. Use 16 kHz mono, 4-second segments with 0.5-second overlap, and log-mel features; reserve up to 1 GB. |

The CIFAKE pilot input consists of copies at `datasets/interim/images/CIFAKE-v3-pilot32-input`; processed outputs are at `datasets/processed/images/CIFAKE-v3-pilot32`. Its full and pilot manifests, grouped split manifests, and leakage reports are under `datasets/manifests/`. Future pilot inputs for the other datasets will be copies and never moves of source media. Before any new write, check that each exact path is absent; do not overwrite an existing run.

## Source, rights, and access checks

### CIFAKE

- Official source: [Kaggle dataset record](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), linked by the [author repository](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images).
- Release information: Kaggle v3 lists 120,000 balanced 32x32 images and 110,717,853 bytes of file payload. The actual versioned ZIP was 109,625,224 bytes. The API metadata field says “Other (specified in description),” while that same official description and the [author README](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) explicitly state the same MIT license as CIFAR-10. The description requires citations to CIFAKE/Bird & Lotfi and CIFAR-10/Krizhevsky & Hinton. The retained [Kaggle metadata snapshot](../../datasets/raw/images/CIFAKE-v3/kaggle-record-v3-snapshot.json) and [acquisition record](../../datasets/raw/images/CIFAKE-v3/acquisition-record.json) preserve these findings.
- Access and acquisition: public Kaggle record; version 3 downloaded on 2026-11-24 from the official Kaggle API endpoint after explicit CIFAKE approval. The original `archive.zip` is retained at the exact destination; extracted files were written with exclusive creation. SHA-256: `1839206c1462371f5c885eda1660b448ea7abf52ee81fcc71e5e4651a4a11389`.
- Annotation validation: extracted paths provided the verified labels and published partition: `train/REAL`, `train/FAKE`, `test/REAL`, `test/FAKE`. A 120,000-row annotation sidecar records normalized labels, original labels, source-relative paths, and `source_split`.
- Full release validation: ZIP CRC test passed; 120,000 unique archive paths and 110,717,853 uncompressed bytes; all 120,000 images decoded as 32x32 RGB; 0 corrupt/unreadable images; labels are 60,000 REAL and 60,000 FAKE. Published partition counts are 100,000 train (50,000/class) and 20,000 test (10,000/class).
- Duplicate and split findings: exact hashing found 1,336 duplicate-content pairs, all same-label; 378 duplicate pairs cross Kaggle's published train/test boundary. Perceptual grouping at distance 6 produced 2,035 non-singleton duplicate groups covering 4,339 images (largest group 29). The derived, deterministic split retains original `source_split` as metadata but reassigns groups: 96,000 train (48,000/class), 12,000 validation (6,000/class), and 12,000 test (6,000/class). The existing leakage checker reported 0 cross-split issues at pHash distance 6. These derived splits are not Kaggle's published test split.
- Pilot preprocessing: 400 copied source images, balanced at 100 per class in each source partition; 400 processed 32x32 outputs; 0 corrupt; 0 missing or unreadable processed images; 0 pilot exact/perceptual duplicates reported by the pipeline.

### FaceForensics++ c23

- Official source: [FaceForensics++ dataset README](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md); [Terms of Use](https://kaldir.vc.in.tum.de/faceforensics/webpage/FaceForensics_TOS.pdf).
- Size and terms: the official README estimates about 10 GB for c23. This is not an exact size for the user's authorized package; inspect its official file listing before preflight. Terms limit use to non-commercial research and education. The provider estimates around 2 TB for full extracted PNG frames; this plan does not extract all frames.
- Access: the user reports provider access is granted and confirmed non-commercial research use. The authorized script/listing is not yet available locally. Do not accept terms on the user's behalf. After exact sizes, destination, and free space are verified, request separate user approval before downloading.
- Annotation validation: use only official video labels, manipulation method, source-video identity, and official split metadata. If those fields are not present or cannot be reconciled, stop processing and report the gap; do not invent labels or partitions.

### ASVspoof 2019 LA

- Official source: [Edinburgh DataShare record](https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45); record includes the LA archive, README, and [LICENSE_text](https://datashare.ed.ac.uk/server/api/core/bitstreams/60c7de6d-37d3-45c1-b52e-9f18b0338e47/content).
- Size and terms: the catalog lists LA.zip at 7.12 GB. LICENSE_text grants ODC Attribution for the database but says individual contents may have separate copyright, privacy, or personality rights not covered by that grant. Review the release README/license and content-rights terms before use.
- Access: the official record is publicly listed; current access conditions will be checked before acquisition. User approval is still required before downloading.
- Annotation validation: use the official LA protocol files for bona-fide/spoof labels, speaker IDs, attack IDs, and train/dev/eval partitions. Check path coverage and unknown labels before preprocessing; preserve protocol IDs without adding personal information.

## Bounded acquisition and validation sequence

For each dataset, independently:

1. Recheck official source URL, exact release/version, license/access terms, current listed byte size, available storage, and exact destination. For gated data, verify the user-reported access and obtain its authorized file listing.
2. Present the dataset-specific request to the user and wait. A previous approval for another source does not authorize this download.
3. Download only after approval. Record date, source, version, exact size, and cryptographic archive hash. Keep archives and original media; never delete them automatically.
4. Verify archive integrity with the provider checksum when available, otherwise record the computed SHA-256 and successful archive listing/test. Extract only to the named dataset directory, without replacing existing files.
5. Validate official annotation files, path coverage, sample counts, class/split distributions, and duplicate IDs. Compare counts with published release totals and explain any sampled/subset differences.
6. Copy a bounded, stratified pilot into the interim directory. Run its modality pipeline with the matching `configs/datasets/*-pilot.json` configuration; preserve source IDs, labels, and published partitions in manifests.
7. Verify manifest rows, processed counts, expected feature files, split generation, and cross-split leakage report. Report actual observed counts, distributions, hashes, and preprocessing outcomes. Do not call real-data validation successful until these checks pass.

The provider partition mapper in `configs/datasets/splits.json` keeps train/training, dev/valid/validation, and eval/evaluation/test in their published partitions. If an input dataset does not provide a partition for one of those categories, retain what it provides; do not fabricate a validation subset while claiming it is official.

## Current validation state

CIFAKE v3 has been downloaded, extracted, fully image-validated, manifested, split, and leakage-checked as recorded above. FaceForensics++ c23 access is reported granted but no media is acquired or validated; exact sizing and separate download approval are pending. ASVspoof 2019 LA remains unacquired and unvalidated. No video model has been trained. Do not use CIFAKE's derived split outputs as the provider's official test partition.
