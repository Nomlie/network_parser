# Demo data packaged with NetworkParser

This directory ships a **small, sample-disjoint** demonstration package for
installation checks, tutorials, and smoke-testing after cloning the repository.
It is **not** the full AFRO-TB evaluation cohort used for the manuscript
production benchmarks.

## Layout

```
data/
├── train/                 # 150 VCF.gz files (training)
├── test/                  #  30 VCF.gz files (held-out query/eval)
├── reference/             # H37Rv FASTA + GenBank annotation
│   ├── H37Rv.fasta
│   ├── H37Rv.gbk
│   ├── reference_manifest.json
│   └── README.md
├── train_metadata.csv     # labels for train IDs
├── test_metadata.csv      # labels for test IDs
├── metadata.csv           # combined train + test metadata
├── train_samples.txt
├── test_samples.txt
└── split_provenance.json
```

## Split design

- Source: AFRO-TB public VCFs (sample-disjoint train/test subset).
- Sizes: **150 train** / **30 test** (no shared sample IDs).
- Sampling: stratified by `(Lineage_clean, AMR_binary)` with random seed **42**.
- Labels: `Lineage_clean`, `AMR_binary`, `Resistance_Profile_Collapsed`, etc.
- Resistance labels are **genotype/catalogue-derived**, not independent phenotypic DST.

## Reference genome

`data/reference/` provides the H37Rv sequence (`H37Rv.fasta`) and GenBank
annotation (`H37Rv.gbk`) required for VCF-oriented demo runs. AFRO demo VCFs
use contig name `M.tuberculosis_H37Rv`. Checksums are recorded in
`reference/reference_manifest.json`.

## Suggested NetworkParser usage (after clone)

```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
# create/activate the environment from environment.yml as documented in README

python -m network_parser.cli train-hierarchy \
  --genomic data/train \
  --meta data/train_metadata.csv \
  --hierarchy_labels Lineage_clean AMR_binary Resistance_Profile_Collapsed \
  --hierarchy_preset lineage_amr_profile \
  --ref_fasta data/reference/H37Rv.gbk \
  --output_dir demo_results/train

python -m network_parser.cli query \
  --genomic data/test \
  --bundle demo_results/train/networkparser_model_bundle.npb \
  --query_input_type vcf \
  --ref_fasta data/reference/H37Rv.gbk \
  --output_dir demo_results/query
```

The demo is intentionally small: it verifies that the install, training, and
query paths work. Manuscript production metrics used the much larger AFRO-TB
cohort described in the paper.

## Provenance

See `split_provenance.json` for source paths, seed, and sample counts.
