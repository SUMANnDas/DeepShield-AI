# Data Leakage Prevention

## Split unit

- The splitter's `auto` setting groups by source identity, identity, speaker, source/original video, audio sample, explicit group ID, exact file hash, or finally individual sample, in that priority order. Explicit `group_field` overrides this choice; absent values fall back to the unique sample ID.
- Video frame records keep their original video ID and inherit the parent video label. All frames with a shared selected group remain together.
- Audio segments group by speaker when a speaker ID is supplied; otherwise they group by source audio sample. Use speaker-disjoint claims only when the provider mapping is verified.
- Exact and perceptual duplicate detection is a separate leakage audit. The image processor records duplicate links, while the split checker reports cross-split exact hashes, pHash matches, identity, speaker, audio sample, and video groups. It does not automatically merge duplicate-image groups into the splitter's assignment.

## CIFAKE v3 validation note

The acquired CIFAKE v3 release contains 1,336 exact duplicate pairs, including 378 pairs across its published train/test boundary. The standard source-split mapping therefore cannot both preserve that partition and keep exact duplicates together. A custom 80/10/10 split was created after assigning connected groups for every pHash match at Hamming distance 6; original `source_split` values remain in the manifest. The final leakage report found zero cross-split issues. These custom splits are not the provider's official test split.

## Reproducible assignments

The splitter takes an explicit seed, train/validation/test ratios, group field, label field, and optional `source_split_field`/`source_split_map`. When a provider partition is mapped, groups retain that published partition and are not reassigned by random balancing. Keep the original value in `source_split` for audit. It aggregates group/label counts in a local SQLite index, greedily balances only groups without a mapped provider partition, and writes new JSONL split manifests. It refuses output collisions. External test datasets should be staged into a separately maintained test-only manifest; the splitter does not assign external sets specially.

Dataset-level external test sets remain entirely in test and are not used to choose preprocessing thresholds or model checkpoints. A held-out generator within GenImage is generator-held-out but not independent-dataset evaluation. Report that distinction.

## Leakage checks

Before training, run the leakage checker over the final split manifests. The report flags source group IDs, exact file hashes, and perceptual image matches found across splits. Near-duplicate image detection is advisory; the Hamming distance threshold must be reported and does not establish semantic identity.

## Limits

Identity-disjoint evaluation cannot be claimed without trustworthy identity annotations. File hashes do not detect re-encoded or cropped copies; perceptual hashes catch some simple visual duplicates but can miss transformations. Dataset licenses and source IDs must be preserved so split decisions are auditable.
