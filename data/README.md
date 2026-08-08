# Demo AFRO-TB VCF subset for NetworkParser

This directory ships a **small, sample-disjoint** demonstration split for CI,
tutorials, and smoke-testing. It is **not** the full AFRO-TB evaluation cohort
used in the manuscript.

## Layout

```
data/
├── train/                 # 150 VCF.gz files (training)
├── test/                  # 30 VCF.gz files (held-out query/eval)
├── train_metadata.csv     # labels for train IDs
├── test_metadata.csv      # labels for test IDs
├── metadata.csv           # combined train + test metadata
├── train_samples.txt
├── test_samples.txt
└── split_provenance.json
```

## Split design

- Source: AFRO-TB public VCFs from the local 800/200 split under
  `AFRO_TB_1000_VCFs_split_80_20` (itself drawn from the AFRO-TB collection).
- Sizes: **150 train** / **30 test** (sample-disjoint; no shared IDs).
- Sampling: stratified by `(Lineage_clean, AMR_binary)` with random seed **42**.
- Labels: subset of `AFRO_dataset_meta_networkparser_ready.csv`.
- Resistance labels are **genotype/catalogue-derived**, not independent phenotypic DST.

## Suggested NetworkParser usage

```bash
# Train a simple hierarchy (example)
network_parser train-hierarchy \
  --genomic data/train \
  --metadata data/train_metadata.csv \
  --hierarchy_labels Lineage_clean,AMR_binary \
  --output results/demo_train

# Query held-out VCFs
network_parser query \
  --genomic data/test \
  --registry results/demo_train/... \
  --output results/demo_query
```

Exact CLI flags depend on your installed NetworkParser version; see the
repository README and `scripts/testing_scripts/` for full recipes.

## Provenance

See `split_provenance.json` for source paths, seed, and sample counts.
