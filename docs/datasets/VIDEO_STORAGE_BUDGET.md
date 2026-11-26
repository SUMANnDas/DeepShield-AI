# Video branch storage budget

## User limit

- Target usage: **90 GB**.
- Absolute video-branch limit: **100 GB total**, across retained archives, extracted source videos, metadata/manifests, optional caches, checkpoints, experiment histories, and temporary work files.
- Keep at least **10 GB free** on C: after a planned operation.
- Default dedicated root: `C:\Users\User\DeepShield-Data\VideoDetection`, configurable with `DEEPSHIELD_VIDEO_DATA_ROOT`.
- Keep video data outside the Git repository. Do not count the existing CIFAKE dataset or image-model artifacts as video storage, and do not modify them.

The 100 GB limit is one combined cap, not 100 GB per artifact category. Store all new video assets under the same root so the quota check sees retained archives and extracted copies together. Video inputs, manifests, persistent sampled frames, run configs, and checkpoints must stay beneath that root. The user plans dataset deletion after training; cleanup is not automated, and files must be retained until training is complete and the cleanup targets are confirmed.

## Feasibility from the current read-only audit

The expected `C:\Users\User\DeepShield-Data\VideoDetection` directory and c23 destination do not currently exist. The latest preflight measured about 220.96 GiB free on C:; this is a snapshot, not a promise of future capacity. The video dataset, cache, and experiment directories are not present in the project. The existing CIFAKE v3 tree remains untouched.

The official FaceForensics++ README estimates approximately 10 GB for its c23 compressed videos and offers a `--num_videos` option for smaller downloads. This is a published estimate, not the exact size of the user's authorized package. The separate Google/Jigsaw DeepFakeDetection subset adds about 25 GB at c23 (3 GB original plus 22 GB manipulated) and must be excluded from the acquisition. The original YouTube source-video bundle (about 38.5 GB) is not required and must also be excluded. The user reports that provider access is granted and confirmed non-commercial research use. The authorized script or file listing is not available in the project or Downloads folder, so exact download and extracted sizes remain unverified. Any archive must be counted alongside extracted files.

Meta's official DFDC page identifies a 5k-video Preview and a 124k-video Full release, but does not publish a byte size. The full release must be excluded from this budget; do not start the Preview until the authorized download listing provides a measured total that fits the cap. DeeperForensics-1.0 reports approximately 284 GB and is outside the limit.

## Initial allocation envelope

The following are ceilings for planning, not measured dataset sizes:

| Storage item | Planning ceiling |
|---|---:|
| Retained download archives | 40 GB |
| Extracted c23 video media | 40 GB |
| Manifests, metadata, and split reports | 1 GB |
| Persistent frame/crop cache | 0 GB by default |
| Checkpoints, optimizer state, and run histories | 3 GB |
| Per-job temporary space | 2 GB |
| Unallocated quota margin | 14 GB |
| **Hard total** | **100 GB** |

These are planning ceilings, not measured dataset sizes. If the provider's archive plus extracted files exceed the allowance or total usage would exceed the 90 GB target, stop before download and present a smaller c23-only subset plan. Never exceed the 100 GB hard cap. Do not remove an archive to make room. The initial preprocessing approach decodes at most 12 deterministically sampled frames per selected video on demand; it does not persist every frame. Any later cache must have its own explicit cap and be included in preflight estimates.

## Enforcement

`scripts/training/video/storage_budget.py` measures current regular-file usage under the configured root, checks planned writes against the hard quota and target-volume free-space reserve, rejects destinations outside the root, and refuses existing destination paths. Video preprocessing defaults to metadata-only frame sampling (`persist_frames: false`); optional JPEG persistence is quota-checked frame by frame. Manifest rows, bounded SQLite annotation indexes, experiment configs, and streamed checkpoints also use the budget guard. A quota breach during a streamed write can leave a partial artifact; it is preserved, is not a valid checkpoint, and must not be reused under the same name. The guard never deletes data or clears caches. Archive download and extraction still require their own measured-size preflight before acquisition.

Before each acquisition, extraction, cache, or experiment run, measure the actual bytes and run the budget preflight. Recheck free space; do not rely on the audit snapshot. Keep actual archive, extraction, cache, and run sizes in the eventual dataset documentation.

## Access and acquisition status

No video dataset has been downloaded. The user reports provider access is granted, but the authorized download script or file listing has not been supplied for exact size verification. The agent has not accepted any provider terms. FaceForensics++ c23 without DeepFakeDetection remains selected; the 100 GB C: allowance does not replace the measured-size preflight and separate download approval. No DFDC download is planned.

## Official source references

- [FaceForensics++ dataset README, access, size, and download options](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)
- [FaceForensics Terms of Use PDF](https://kaldir.vc.in.tum.de/faceforensics/webpage/FaceForensics_TOS.pdf)
- [Meta DFDC dataset and access requirements](https://ai.meta.com/datasets/dfdc/)
- [DeeperForensics-1.0 storage and access requirements](https://github.com/EndlessSora/DeeperForensics-1.0/blob/master/dataset/README.md)
