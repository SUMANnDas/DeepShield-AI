# Video branch storage budget

## User limit

- Target usage: **30 GB**.
- Absolute video-branch limit: **40 GB total**, across retained archives, extracted source videos, metadata/manifests, optional caches, checkpoints, experiment histories, and temporary work files.
- Keep at least **5 GB free** on the destination volume after a planned operation.
- Default dedicated root: `D:\DeepShield-Data\VideoDetection`, configurable with `DEEPSHIELD_VIDEO_DATA_ROOT`.
- Keep video data outside the Git repository. Do not count the existing CIFAKE dataset or image-model artifacts as video storage, and do not modify them.

The 40 GB limit is one combined cap; it is not 40 GB per drive. Store all new video assets under the same root so the quota check sees retained archives and their extracted copies together.

## Feasibility from the current read-only audit

The expected `D:\DeepShield-Data\VideoDetection` directory does not currently exist. The audit measured about 66.9 GiB free on D: and 221.8 GiB free on C:; these readings are snapshots, not a promise of future capacity. The video dataset, cache, and experiment directories are not present in the project. The existing CIFAKE v3 tree under the repository is about 0.21 GiB and remains untouched.

The official FaceForensics++ README estimates approximately 10 GB for its c23 compressed videos and offers a `--num_videos` option for smaller downloads. The separate Google/Jigsaw DeepFakeDetection subset adds about 25 GB at c23 (3 GB original plus 22 GB manipulated) and must be excluded from the initial acquisition. The original YouTube source-video bundle (about 38.5 GB) is not required and must also be excluded. FaceForensics++ requires an approved access request. Its terms limit use to non-commercial research and education, so the user must confirm this scope is compatible before accepting the terms. An archive, if the delivery method provides one, must be counted in addition to extracted files because the archive is retained.

Meta's official DFDC page identifies a 5k-video Preview and a 124k-video Full release, but does not publish a byte size. The full release must be excluded from this budget; do not start the Preview until the authorized download listing provides a measured total that fits the cap. DeeperForensics-1.0 reports approximately 284 GB and is outside the limit.

## Initial allocation envelope

The following are ceilings for planning, not measured dataset sizes:

| Storage item | Planning ceiling |
|---|---:|
| Retained download archives | 12 GB |
| Extracted c23 video media | 12 GB |
| Manifests, metadata, and split reports | 1 GB |
| Persistent frame/crop cache | 0 GB by default |
| Checkpoints, optimizer state, and run histories | 3 GB |
| Per-job temporary space | 2 GB |
| Unallocated quota margin | 10 GB |
| **Hard total** | **40 GB** |

If the provider's archive plus extracted files exceed their combined allowance, use an approved smaller subset or stop and ask before acquisition. Do not remove an archive to make room. The initial preprocessing approach decodes at most 12 deterministically sampled frames per selected video on demand; it does not persist every frame. Any later cache must have its own explicit cap and be included in preflight estimates.

## Enforcement

`scripts/training/video/storage_budget.py` measures current regular-file usage under the configured root, checks the planned write against the hard quota and the target-volume free-space reserve, rejects destinations outside the root, and refuses existing destination paths. It is read-only and never creates directories, deletes data, or clears caches.

Before each acquisition, extraction, cache, or experiment run, measure the actual bytes and run the budget preflight. Recheck free space; do not rely on the audit snapshot. Keep actual archive, extraction, cache, and run sizes in the eventual dataset documentation.

## Access and acquisition status

No video dataset has been downloaded. No portal terms have been accepted and no registration has been submitted. FaceForensics++ c23 without DeepFakeDetection is the best-documented initial candidate for the 40 GB cap, subject to the user's confirmation that non-commercial research/education use is acceptable, provider approval, and separate dataset-specific acquisition approval. DFDC Preview remains a possible alternative only if credentials are restored and its authorized object listing fits without exceeding 40 GB while preserving archives and extracted media.

## Official source references

- [FaceForensics++ dataset README, access, size, and download options](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)
- [FaceForensics Terms of Use PDF](https://kaldir.vc.in.tum.de/faceforensics/webpage/FaceForensics_TOS.pdf)
- [Meta DFDC dataset and access requirements](https://ai.meta.com/datasets/dfdc/)
- [DeeperForensics-1.0 storage and access requirements](https://github.com/EndlessSora/DeeperForensics-1.0/blob/master/dataset/README.md)
