# Recommended Dataset Selection

## Capacity snapshot

After CIFAKE v3 acquisition and validation, C: had about 235.16 GiB free and D: about 66.93 GiB free. The project is on C:. No approved DeepShield dataset destination exists on D:, so data remains under the project's `datasets/` directory on C:. Full DFDC and DeeperForensics-1.0 exceed C:'s available space; extracted frames and derived features require additional headroom.

## Recommended staged combination

| Modality/task | Primary training source | Development / validation | Independent evaluation | Storage notes |
|---|---|---|---|---|
| General AI-generated still images | CIFAKE for a compact baseline only | GenImage, selected generator groups after the archive size is known | GenImage held-out generator subset; report this as generator-held-out, not fully independent dataset evaluation | CIFAKE v3 ZIP 109,625,224 bytes; extracted payload 110,717,853 bytes. Exact duplicates cross the publisher's train/test partition, so use the validated derived grouped split for internal experiments. GenImage does not publish a total byte size. |
| Face image manipulation | FaceForensics++ frames, kept as a separate task from general AI-image detection | Identity/video-disjoint validation within the chosen FF++ subset | ForgeryNet image annotations if a feasible, permission-cleared subset is available | FF++ c23 video release is about 10 GB; full extracted PNGs can be about 2 TB. ForgeryNet byte size is not published. |
| Face deepfake video | FaceForensics++ c23 subset | Grouped validation from FF++; DeeperForensics is a future robustness option only if storage is arranged | Celeb-DF v2 as a cross-dataset test, after checking the actual download size and terms | FF++ c23 about 10 GB; Celeb-DF provider byte count not stated. DeeperForensics about 284 GB and exceeds current C: free space. |
| Synthetic speech | ASVspoof 2019 LA training partition | Its official development partition; optionally WaveFake as a domain-diverse development set | ASVspoof 2021 Speech Deepfake evaluation partition, respecting challenge keys | ASVspoof 2019 LA archive 7.12 GB; 2021 DF evaluation archives total 34.5 GB; WaveFake v1.2.0 archive 28.9 GB. ASVspoof's database license does not by itself resolve rights in individual speech content. |

The stages above are recommendations, not acquisition authorization. Do not download until the user approves the named dataset and release. For gated datasets, approval to download does not bypass the provider's request form, EULA, or license.

## Acquisition status

The staged pilot covers three separate task-specific sources. CIFAKE v3 has been acquired and validated. FaceForensics++ c23 and ASVspoof 2019 LA remain unacquired and require separate user approval. Their estimated source payloads plus CIFAKE's actual ZIP size total about 17.23 GB (inference from CIFAKE 0.1096 GB, FF++ c23 about 10 GB, and ASVspoof LA.zip 7.12 GB). FaceForensics++ is request-gated and restricted to approved non-commercial research/education use. ASVspoof's database license does not resolve rights in individual speech contents. CIFAKE's Kaggle license field is "Other (specified in description)," but the description explicitly specifies MIT and matches the author README; preserved source snapshots are in the CIFAKE raw directory.

The remaining provider estimates fit on C: with about 235.16 GiB free. Reserve approximately 40 GB for archives, extracted media, pilot outputs, and workspace headroom. This is a planning estimate, not a measurement of future processed output. Do not extract all FaceForensics++ frames: the provider estimates about 2 TB for full PNG extraction. The pilot caps video preprocessing at 100 videos and at most 12 uniformly sampled frames per video (1,200 frames maximum), and audio preprocessing at 500 utterances. Keep downloaded archives and source media. Adding ASVspoof 2021 DF evaluation archives would add about 34.5 GB and raise known source payload to about 51.7 GB; it is an evaluation-only stage and must remain untouched during model selection.

The CIFAKE destination is `datasets/raw/images/CIFAKE-v3`; its archive and source files are present. The still-pending destinations are `datasets/raw/video/FaceForensicsPP-c23` and `datasets/raw/audio/ASVspoof2019-LA`. Full paths and approval gates are recorded in [PILOT_ACQUISITION_PLAN.md](PILOT_ACQUISITION_PLAN.md).

GenImage has no published archive byte size, and Celeb-DF has no authoritative size in the author README, so defer both until the authorized download page reveals exact release size. DFDC full and DeeperForensics-1.0 are not suitable for the currently available C: space.

## Scientific separation

- CIFAKE and GenImage test real-photo versus AI-generated-image classification. They do not establish a face-swap or general image-tampering detector.
- FaceForensics++, Celeb-DF, DeeperForensics, ForgeryNet, and DFDC primarily concern facial/video manipulation. Frames inherit the video label and must remain grouped with their source video.
- ASVspoof/FoR/WaveFake are audio spoof or synthetic-speech tasks. A voice-cloning benchmark does not validate arbitrary synthetic sounds.
- Do not combine unmatched image, video, or audio samples into one multimodal training example.
