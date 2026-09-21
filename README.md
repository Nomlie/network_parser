# NetworkParser

NetworkParser trains genomic classifiers from labelled VCF files (or a feature matrix) and uses those models to predict labels on new samples.

Typical use: train a model → query new samples → evaluate the predictions.

This is a research tool. Predictions are not clinical diagnoses.

## 1. Install

```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
conda env create -f environment.yml
conda activate networkparser
```

Stay in this folder for every command below.

## 2. How to run

The program is the file `run_network_parser.py` in this folder.

```bash
python run_network_parser.py <command> [options]
```

Start here:

```bash
python run_network_parser.py --help
python run_network_parser.py train-hierarchy --help
python run_network_parser.py query --help
python run_network_parser.py evaluate --help
```

If a setting is wrong (missing files, bad `config.json`, too few VCFs), the program prints the problems and exits.

The examples below use the demo data in `data/`. Training can take a while.

### Train

```bash
python run_network_parser.py train-hierarchy \
  --genomic data/train \
  --meta data/train_metadata.csv \
  --hierarchy_labels Lineage_clean AMR_binary \
  --ref_fasta data/reference/H37Rv.fasta \
  --output_dir results/train
```

This writes:

- `results/train/networkparser_model_bundle.npb` — use this for query
- `results/train/hierarchical_model_registry.json`

One label only:

```bash
python run_network_parser.py run \
  --genomic data/train \
  --meta data/train_metadata.csv \
  --label Lineage_clean \
  --ref_fasta data/reference/H37Rv.fasta \
  --output_dir results/single_label
```

### Query

```bash
python run_network_parser.py query \
  --genomic data/test \
  --bundle results/train/networkparser_model_bundle.npb \
  --ref_fasta data/reference/H37Rv.fasta \
  --output_dir results/query
```

Main output: `results/query/query_predictions.csv`.

`--query_input_type auto` is the default. You can set it to `vcf`, `matrix`, `fasta`, or `fastq`.

Load `.npb` files only from sources you trust.

### Evaluate

```bash
python run_network_parser.py evaluate \
  --predictions results/query/query_predictions.csv \
  --meta data/test_metadata.csv \
  --label AMR_binary \
  --output_dir results/evaluation
```

Full hierarchy:

```bash
python run_network_parser.py evaluate-hierarchy \
  --predictions results/query/query_predictions.csv \
  --meta data/test_metadata.csv \
  --hierarchy_labels Lineage_clean AMR_binary \
  --output_dir results/hierarchy_evaluation
```

## 3. Your own data

| Input | What it is |
|---|---|
| `--genomic` | Folder of per-sample VCF/VCF.gz files, or a CSV/TSV matrix |
| `--meta` | CSV/TSV with sample IDs and label columns |
| `--ref_fasta` | FASTA or GenBank file, for VCF / FASTA / FASTQ input |

VCF file names without `.vcf` / `.vcf.gz` must match the sample IDs in the metadata.

Training from a VCF folder needs at least 10 VCF files by default.

Replace the `data/...` paths in the commands above with your paths.

## 4. Commands

```bash
python run_network_parser.py <command> --help
```

| Command | What it does |
|---|---|
| `train-hierarchy` | Train a hierarchy of models |
| `query` | Predict labels for new samples |
| `evaluate` | Score predictions for one label |
| `evaluate-hierarchy` | Score a full hierarchy |
| `run` | Train a model for one label |
| `bundle` | Build a `.npb` file from an existing registry |
| `cross-validate` | Repeated cross-validation for one label |
| `annotate-panels` | Add gene/catalogue notes to selected markers |

## 5. Optional settings

Save extra settings in a JSON file and pass it with `--config`:

```json
{
  "qual_threshold": 30.0,
  "min_dp_per_sample": 10,
  "n_jobs": -1,
  "random_state": 42
}
```

```bash
python run_network_parser.py train-hierarchy \
  --genomic data/train \
  --meta data/train_metadata.csv \
  --hierarchy_labels Lineage_clean AMR_binary \
  --config path/to/config.json \
  --output_dir results/train
```

All setting names are in [`network_parser/config.py`](network_parser/config.py).

If your metadata uses these column names, you can pass a preset:

| `--hierarchy_preset` | Columns |
|---|---|
| `lineage_amr_binary` | `Lineage_clean` → `AMR_binary` |
| `lineage_amr_profile` | `Lineage_clean` → `AMR_binary` → `Resistance_Profile_Collapsed` |
| `lineage_family_amr_profile` | `Lineage_family` → `Lineage_clean` → `AMR_binary` → `Resistance_Profile_Collapsed` |

## 6. Useful output files

| File | Where |
|---|---|
| `networkparser_model_bundle.npb` | Training folder — use this for query |
| `hierarchical_model_registry.json` | Training folder |
| `query_predictions.csv` | Query folder |
| `query_predictions_readable.html` | Query folder |

## 7. If something goes wrong

- **Sample IDs do not match.** VCF names (without `.vcf` / `.vcf.gz`) must match the ID column in metadata.
- **Too few VCF files.** Training needs at least 10 VCFs in the genomic folder by default.
- **Query recovery is low.** Use the same reference genome as training.
- **A hierarchy branch was skipped.** That branch had too few samples or classes. See the model registry and query audit.

## Tests and extra docs

```bash
pytest -q
```

- [Demo data](data/README.md)
- [Architecture overview](docs/NETWORKPARSER_FULL_PICTURE.md)
- [Known-marker configuration](docs/KNOWN_MARKER_SEED.md)
