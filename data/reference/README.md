# Reference genome data

This directory provides the **H37Rv** reference used by the NetworkParser
demo workflows and by the AFRO-TB VCF coordinate system described in the
manuscript.

| File | Role |
|------|------|
| `H37Rv.fasta` | Reference sequence for VCF/FASTA/FASTQ query alignment context |
| `H37Rv.gbk` | GenBank annotation used for marker annotation / gene context |
| `reference_manifest.json` | File sizes and SHA-256 checksums |

## How to run with this reference

From the repository root:

```bash
python run_network_parser.py train-hierarchy \
  --genomic data/train \
  --meta data/train_metadata.csv \
  --hierarchy_labels Lineage_clean AMR_binary Resistance_Profile_Collapsed \
  --hierarchy_preset lineage_amr_profile \
  --ref_fasta data/reference/H37Rv.gbk \
  --config data/config.json \
  --output_dir demo_results/train

python run_network_parser.py query \
  --genomic data/test \
  --bundle model/networkparser_model_bundle.npb \
  --query_input_type vcf \
  --ref_fasta data/reference/H37Rv.gbk \
  --config data/config.json \
  --output_dir demo_results/query
```

AFRO VCFs use contig name `M.tuberculosis_H37Rv`. Ensure query VCFs and
the reference annotation use a compatible coordinate system.
