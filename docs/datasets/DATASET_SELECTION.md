# Recommended Dataset Selection

## Capacity snapshot

After CIFAKE v3 acquisition and validation, C: had about 235.16 GiB free and D: about 66.93 GiB free. A newer preflight measured 220.96 GiB free on C:. The video branch now uses the dedicated outside-repository root `C:\Users\User\DeepShield-Data\VideoDetection`, with a 90 GB target and 100 GB hard cap; the path is not created until the approved package is sized. Full DFDC and DeeperForensics-1.0 exceed the video cap; extracted frames and derived features require additional headroom.

## Recommended staged combination

| Modality/task | Primary training source | Development / validation | Independent evaluation | Storage notes |
|---|---|---|---|---|
| General AI-generated still images | CIFAKE for a compact baseline only | GenImage, selected generator groups after the archive size is known | GenImage held-out generator subset; report this as generator-held-out, not fully independent dataset evaluation | CIFAKE v3 ZIP 109,625,224 bytes; extracted payload 110,717,853 bytes. Exact duplicates cross the publisher's train/test partition, so use the validated derived grouped split for internal experiments. GenImage does not publish a total byte size. |
| Face image manipulation | FaceForensics++ frames, kept as a separate task from general AI-image detection | Identity/video-disjoint validation within the chosen FF++ subset | ForgeryNet image annotations if a feasible, permission-cleared subset is available | FF++ c23 video release is about 10 GB; full extracted PNGs can be about 2 TB. ForgeryNet byte size is not published. |
| Face deepfake video | FaceForensics++ c23 subset | Grouped validation from FF++; DeeperForensics is a future robustness option only if storage is arranged | Celeb-DF v2 as a cross-dataset test, after checking the actual download size and terms | FF++ c23 about 10 GB; Celeb-DF provider byte count not stated. DeeperForensics about 284 GB and exceeds the current 100 GB video cap. |
| Synthetic speech | ASVspoof 2019 LA training partition | Its official development partition; optionally WaveFake as a domain-diverse development set | ASVspoof 2021 Speech Deepfake evaluation partition, respecting challenge keys | ASVspoof 2019 LA archive 7.12 GB; 2021 DF evaluation archives total 34.5 GB; WaveFake v1.2.0 archive 28.9 GB. ASVspoof's database license does not by itself resolve rights in individual speech content. |

The stages above are recommendations, not acquisition authorization. The user reports FaceForensics++ provider access is granted and confirmed non-commercial research use. Do not download until the exact c23 listing, size, destination, and free space are verified and the user approves that acquisition. For gated datasets, access approval does not bypass applicable terms.

## Acquisition status

The staged plan covers three separate task-specific sources. CIFAKE v3 has been acquired and validated. FaceForensics++ c23 is not acquired; the user reports provider access is granted, with exact package sizing and separate download approval pending. ASVspoof 2019 LA remains unacquired and requires separate user approval. Published source-size estimates do not include any unknown retained archive plus extraction duplication. FaceForensics++ use is restricted to approved non-commercial research/education. ASVspoof's database license does not resolve rights in individual speech contents. CIFAKE's Kaggle license field is "Other (specified in description)," but the description explicitly specifies MIT and matches the author README; preserved source snapshots are in the CIFAKE raw directory.

For the active video branch, cap combined archives, extracted media, manifests, caches, checkpoints, and temporary outputs at 100 GB (90 GB target), with 10 GB kept free on C:. This is a hard ceiling, not a measured c23 footprint; exact file sizes are pending the authorized listing. Do not extract all FaceForensics++ frames: the provider estimates about 2 TB for full PNG extraction. The pilot caps video preprocessing at 100 videos and at most 12 uniformly sampled frames per video (1,200 frames maximum). The separate audio plans are not part of this video budget. Keep source data during training; no automatic deletion is configured.

The CIFAKE destination is `datasets/raw/images/CIFAKE-v3`; its archive and source files are present. The planned FaceForensics++ destination is `C:\Users\User\DeepShield-Data\VideoDetection\raw\FaceForensics++-c23`; it does not yet exist. The still-pending audio destination is `datasets/raw/audio/ASVspoof2019-LA`. Full paths and approval gates are recorded in [PILOT_ACQUISITION_PLAN.md](PILOT_ACQUISITION_PLAN.md).

GenImage has no published archive byte size, and Celeb-DF has no authoritative size in the author README, so defer both until the authorized download page reveals exact release size. DFDC full and DeeperForensics-1.0 exceed the current 100 GB video cap.

## Scientific separation

- CIFAKE and GenImage test real-photo versus AI-generated-image classification. They do not establish a face-swap or general image-tampering detector.
- FaceForensics++, Celeb-DF, DeeperForensics, ForgeryNet, and DFDC primarily concern facial/video manipulation. Frames inherit the video label and must remain grouped with their source video.
- ASVspoof/FoR/WaveFake are audio spoof or synthetic-speech tasks. A voice-cloning benchmark does not validate arbitrary synthetic sounds.
- Do not combine unmatched image, video, or audio samples into one multimodal training example.
