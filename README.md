# NetworkParser

NetworkParser trains microbial genomic classifiers from VCF files or feature matrices and applies trained models to new samples. It supports single-label analyses and ordered hierarchies such as lineage → AMR status → resistance profile.

> NetworkParser is a research tool. Its predictions are not validated clinical diagnoses.

## Workflow

```text
Training data + metadata → train model → model bundle → query new samples → evaluate predictions
```

## 1. Install

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
conda env create -f environment.yml
conda activate networkparser
```

NetworkParser currently runs from the repository root and does not require a `pip install` step.

Check that the command-line interface is available:

```bash
python -m network_parser.cli run --help
```

## 2. Prepare the inputs

NetworkParser requires genomic data and matching metadata.

### Genomic data

Use either:

- a directory containing one VCF or gVCF per sample; or
- a CSV/TSV feature matrix with sample IDs in the first column.

Example feature matrix:

```csv
Sample,chr1:761155:C:T,chr1:2155168:C:T
sample_001,0,1
sample_002,1,NaN
```

Matrix values are:

- `0` — reference or baseline
- `1` — alternate or non-baseline
- `NaN` — missing or unresolved

### Metadata

Metadata must be a CSV or TSV file. Use a `Sample` column for sample IDs and add the labels you want to predict.

```csv
Sample,Lineage,AMR_binary
sample_001,L4.1,resistant
sample_002,L2.2,susceptible
```

Sample IDs must match between the genomic data and metadata.

### Reference genome

Use `--ref_fasta` with a FASTA or GenBank reference when working with VCF, FASTA, or FASTQ inputs. It is optional for a precomputed feature matrix.

## 3. Train a model

### Single-label training

Use `run` when predicting one metadata column:

```bash
python -m network_parser.cli run \
  --genomic /path/to/training_vcfs \
  --meta /path/to/metadata.csv \
  --label Lineage \
  --ref_fasta /path/to/reference.fasta \
  --output_dir /path/to/results/single_label \
  --n_jobs -1
```

### Hierarchical training

List hierarchy labels from broadest to most specific:

```bash
python -m network_parser.cli train-hierarchy \
  --genomic /path/to/training_vcfs \
  --meta /path/to/metadata.csv \
  --hierarchy_labels Lineage AMR_binary \
  --ref_fasta /path/to/reference.fasta \
  --output_dir /path/to/results/hierarchy \
  --n_jobs -1
```

Hierarchical training writes a model registry and, by default, a portable bundle named `networkparser_model_bundle.npb`.

Available hierarchy presets:

| Preset | Metadata columns |
|---|---|
| `lineage_amr_binary` | `Lineage_clean` → `AMR_binary` |
| `lineage_amr_profile` | `Lineage_clean` → `AMR_binary` → `Resistance_Profile_Collapsed` |
| `lineage_family_amr_profile` | `Lineage_family` → `Lineage_clean` → `AMR_binary` → `Resistance_Profile_Collapsed` |

Use a preset instead of `--hierarchy_labels` when your metadata uses those column names:

```bash
python -m network_parser.cli train-hierarchy \
  --genomic /path/to/training_vcfs \
  --meta /path/to/metadata.csv \
  --hierarchy_preset lineage_amr_binary \
  --ref_fasta /path/to/reference.fasta \
  --output_dir /path/to/results/hierarchy
```

## 4. Query new samples

Use the bundle created during hierarchical training:

```bash
python -m network_parser.cli query \
  --genomic /path/to/query_input \
  --bundle /path/to/results/hierarchy/networkparser_model_bundle.npb \
  --query_input_type auto \
  --ref_fasta /path/to/reference.fasta \
  --output_dir /path/to/results/query \
  --n_jobs -1
```

Supported query types are:

| Type | Input |
|---|---|
| `matrix` | CSV/TSV feature matrix |
| `vcf` | One VCF or a directory of VCFs |
| `fasta` | FASTA sequence |
| `fastq` | Directory of paired-end FASTQ files |
| `auto` | Detect the type from the input |

The main result is `query_predictions.csv`. NetworkParser also writes compact, readable, and audit reports to the query output directory.

> Model bundles contain Python pickle objects. Only load `.npb` files from trusted sources.

## 5. Evaluate predictions

Evaluate a single label:

```bash
python -m network_parser.cli evaluate \
  --predictions /path/to/results/query/query_predictions.csv \
  --meta /path/to/test_metadata.csv \
  --label AMR_binary \
  --output_dir /path/to/results/evaluation
```

Evaluate an entire hierarchy:

```bash
python -m network_parser.cli evaluate-hierarchy \
  --predictions /path/to/results/query/query_predictions.csv \
  --meta /path/to/test_metadata.csv \
  --hierarchy_labels Lineage AMR_binary \
  --output_dir /path/to/results/hierarchy_evaluation
```

## Other commands

| Command | Purpose |
|---|---|
| `bundle` | Build a portable `.npb` bundle from an existing model registry |
| `cross-validate` | Run repeated cross-validation for one label |
| `annotate-panels` | Add gene, catalogue, and stability annotations to selected panels |

Run any command with `--help` to see all available options:

```bash
python -m network_parser.cli cross-validate --help
python -m network_parser.cli annotate-panels --help
```

`train-two-level` remains available as a legacy alias for `train-hierarchy`.

## Configuration

Most settings are available as command-line options. For repeatable runs, place configuration overrides in a JSON file and pass it with `--config`:

```json
{
  "qual_threshold": 30.0,
  "min_dp_per_sample": 10,
  "n_jobs": -1,
  "random_state": 42
}
```

```bash
python -m network_parser.cli run \
  --genomic /path/to/training_matrix.csv \
  --meta /path/to/metadata.csv \
  --label Lineage \
  --config /path/to/config.json \
  --output_dir /path/to/results
```

See [`network_parser/config.py`](network_parser/config.py) for all configuration fields.

## Main output files

File names depend on the selected command. The files most users need are:

| File | Purpose |
|---|---|
| `hierarchical_model_registry.json` | Records the trained hierarchy and model paths |
| `networkparser_model_bundle.npb` | Portable model used for queries |
| `query_predictions.csv` | Full prediction table |
| `query_predictions_compact.tsv` | Compact prediction table |
| `query_predictions_readable.html` | Human-readable prediction report |
| `query_route_audit.json` | Hierarchy route and fallback audit |
| `query_alignment_summary.json` | Query feature-recovery summary |

## Troubleshooting

**Samples do not align**

Check that sample IDs match exactly between the genomic input and metadata. Also check for duplicate IDs and extra whitespace.

**Many values are missing**

Check VCF filters, depth, genotype quality, mapping quality, ploidy, and the reference genome.

**Query feature recovery is low**

Use the same reference genome, contig names, coordinates, and allele orientation for training and querying.

**A hierarchy branch was skipped**

Check the model registry and query audit. A branch may be skipped when it does not have enough samples or supported classes.

## Testing

Run the test suite from the repository root:

```bash
pytest -q
```

## More documentation

- [Architecture overview](docs/NETWORKPARSER_FULL_PICTURE.md)
- [Known-marker configuration](docs/KNOWN_MARKER_SEED.md)
