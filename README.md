# NetworkParser

NetworkParser is a research pipeline for microbial genomics. From per-sample VCFs (or precomputed feature matrices) and labelled metadata, it builds interpretable classifiers for **single labels** or **ordered biological hierarchies** (for example lineage → AMR → resistance profile). In hierarchical mode, each child model is trained only on samples that follow its parent branch; at query time the same route is walked and audited end to end.

It packages the trained feature space into portable model bundles and applies those models to new samples as matrix, VCF, FASTA, or paired FASTQ. Predictions stay tied to the genomic markers and hierarchy route that produced them. Typical outputs include model registries, portable `.npb` bundles, ranked marker tables, route audits, readable query reports, evaluation metrics, and optional post-training panel annotations.

> NetworkParser is a research workflow. Its reports are not, by themselves, validated clinical diagnoses.

## Capabilities

| Area | What it does |
|---|---|
| **Single-label training** | Discover markers, filter features on training data only, train a classifier, and optionally interpret with decision trees. |
| **Hierarchical training** | Train ordered label levels (e.g. lineage → AMR → resistance profile). Each child model is fit only on samples that belong to its parent branch. |
| **Portable bundles** | Package a trained hierarchy registry and artifacts into a single `.npb` file for inference elsewhere. |
| **Query / inference** | Apply a registry or bundle to new samples without re-running feature selection or training. |
| **Query inputs** | Matrix (CSV/TSV), VCF/gVCF directory, FASTA DNA, or paired-end FASTQ (preprocessed to VCF/gVCF). |
| **Callability-aware matrix** | Encodes callable reference (`0`), alternate (`1`), and non-callable/missing (`NaN`) without silently treating absence as reference. |
| **Central feature filtering** | Train-only association filtering (`rf_fdr`, `chi2_fdr`, `fisher_fdr`, `chi2_perm_fdr`) with FDR / Bonferroni correction. |
| **Compact panels** | Optional separability check over top-N panel sizes (default candidates: 100, 200, 500, 1000). |
| **Known markers** | Report-only overlap (`--known_markers`), or optional force-include seed of catalogue alleles into phenotype panels (`seed_known_markers`). |
| **Hierarchy safeguards** | Low-support class review, global / parent-conditioned fallbacks, AMR weak-evidence guard for susceptible calls. |
| **Evaluation** | Score saved predictions against labelled metadata (single-label or hierarchy); dedicated hierarchy evaluation pack with bootstrap CIs. |
| **Leakage-aware CV** | Repeated stratified CV that refits filtering, panel selection, imputation, and models inside each training fold only. |
| **Panel annotation** | Post-training gene / consequence / catalogue / stability summaries without changing the trained model. |

## Commands

Run from the repository root:

```bash
python -m network_parser.cli <command> ...
```

| Command | Purpose |
|---|---|
| `run` | Single-label feature discovery, model training, optional decision-tree interpretation. |
| `train-hierarchy` | Train two or more ordered label levels with branch-scoped models. |
| `train-two-level` | Deprecated alias for `train-hierarchy`. |
| `bundle` | Package a trained hierarchy registry into a portable `.npb` file. |
| `query` | Apply a registry or bundle to new matrix / VCF / FASTA / FASTQ input. |
| `evaluate` | Compare saved predictions with labelled metadata. |
| `evaluate-hierarchy` | Hierarchy evaluation pack: per-level metrics, full-path accuracy, bootstrap CIs. |
| `cross-validate` | Leakage-aware repeated cross-validation for one label (alias: `cross_validation`). |
| `annotate-panels` | Annotate trained panels with gene, catalogue, and optional CV stability context. |

```bash
python -m network_parser.cli run --help
python -m network_parser.cli train-hierarchy --help
python -m network_parser.cli query --help
```

For backward compatibility, omitting the subcommand runs the single-label workflow. New scripts should use the explicit `run` subcommand so their intent is clear.

## Installation

NetworkParser is currently run directly from its source tree; there is no packaging manifest or `pip install` step yet. [Conda](https://docs.conda.io/) or Mamba is recommended because the environment includes both Python packages and bioinformatics command-line tools.

```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
conda env create -f environment.yml
conda activate networkparser
```

Verify the source checkout and command-line entry point from the repository root:

```bash
python -c "import network_parser; print(network_parser.__version__)"
python -m network_parser.cli run --help
```

The environment includes:

| Scope | Dependencies | Used for |
|---|---|---|
| Core Python | NumPy, pandas, SciPy, scikit-learn, statsmodels, NetworkX, Joblib, PyYAML, tqdm | Training, inference, statistics, configuration, and artifact generation |
| Sequence parsing | Biopython | Reference FASTA / GenBank parsing and marker contexts |
| VCF / HTS | bcftools, HTSlib, tabix | VCF and gVCF processing |
| FASTQ | BWA, samtools, bcftools | Paired-FASTQ alignment and variant calling |
| FASTA queries | BLAST | Marker-context mapping when `blast` mode is selected or detected |
| Plotting and development | Matplotlib, pytest, Black, Flake8, mypy | Reports, tests, formatting, linting, and type checks |

## What it takes in

### Genomic input (`--genomic`)

Either of:

1. **VCF directory** — one `.vcf`, `.vcf.gz`, `.g.vcf`, or `.g.vcf.gz` file per sample.
2. **Feature matrix** — CSV or TSV; first column = sample ID, remaining columns = genomic features (IDs must be unique).

Example feature matrix:

```csv
Sample,chr1:761155:C:T,chr1:2155168:C:T
sample_001,0,1
sample_002,1,NaN
```

For VCF directories NetworkParser:

- parses samples with shared call-semantics (haploid / diploid);
- applies FILTER, QUAL, DP, GQ, MQ, and related rules;
- uses gVCF reference blocks (when present) for callable reference;
- merges retained sites into a sample-by-feature matrix;
- optionally writes matrix, annotation, and feature-manifest artifacts.

Sample IDs come from the VCF sample header when available; otherwise from the filename.

### Metadata (`--meta`)

CSV or TSV with at least two columns. A column named `Sample` is preferred as the identifier; otherwise the first column is used. All requested label columns must exist.

```csv
Sample,Lineage_Supergroup,Lineage,AMR_binary
sample_001,L4,L4.1,resistant
sample_002,L2,L2.2,susceptible
```

Genomic and metadata samples are aligned by identifier before supervised analysis.

### Reference (`--ref_fasta`)

Optional for matrix input; recommended for VCF-oriented and sequence query workflows. Accepts FASTA or GenBank. FASTQ query mode requires a suitable reference and uses BWA / samtools / bcftools before the same VCF semantics as training.

### Matrix contract (how genotypes are encoded)

| Value | Meaning |
|---:|---|
| `0.0` | Callable baseline / reference |
| `1.0` | Callable alternate / non-baseline |
| `NaN` | Non-callable, missing, filtered, or unresolved |

`NaN` is **not** converted to zero. Absence from a variant-only VCF is unknown unless you explicitly enable the legacy `assume_absent_variant_is_reference` option (not recommended solely to fill missing values). Algorithms that cannot accept `NaN` use a train-fitted imputation policy (default: `baseline`).

## How to use it

### 1. Single-label training

```bash
python -m network_parser.cli run \
  --genomic /path/to/training_vcfs \
  --meta /path/to/metadata.csv \
  --label Lineage \
  --ref_fasta /path/to/reference.fasta \
  --output_dir /path/to/results/single_label \
  --n_jobs -1
```

`--pipeline_mode` controls branches:

| Mode | Behavior |
|---|---|
| `both` (default) | Matrix + ML + supported decision-tree interpretation |
| `matrix_only` | Load / build matrix only |
| `ml_only` | ML path without decision-tree path |
| `decision_tree_only` | Decision-tree-oriented path |

Optional report-only known-marker overlap:

```bash
python -m network_parser.cli run \
  --genomic /path/to/matrix.csv \
  --meta /path/to/metadata.tsv \
  --label AMR_binary \
  --known_markers /path/to/known_markers.tsv \
  --output_dir /path/to/results/amr
```

`--known_markers` writes an overlap report; it does **not** force markers into training. To force-include catalogue alleles in phenotype panels, enable config `seed_known_markers` (default off). See [Known-marker seed](#optional-known-marker-seed) below and [`docs/KNOWN_MARKER_SEED.md`](docs/KNOWN_MARKER_SEED.md).

### 2. Hierarchical training

Ordered labels (broad → fine):

```bash
python -m network_parser.cli train-hierarchy \
  --genomic /path/to/training_vcfs \
  --meta /path/to/metadata.csv \
  --hierarchy_labels Lineage_Supergroup Lineage AMR_binary \
  --ref_fasta /path/to/reference.fasta \
  --output_dir /path/to/results/hierarchy \
  --n_jobs -1
```

Biological presets (column names must match your metadata):

```bash
python -m network_parser.cli train-hierarchy \
  --genomic /path/to/training_vcfs \
  --meta /path/to/metadata.csv \
  --hierarchy_preset lineage_amr_profile \
  --output_dir /path/to/results/hierarchy
```

| Preset | Levels |
|---|---|
| `lineage_amr_profile` | `Lineage_clean` → `AMR_binary` → `Resistance_Profile_Collapsed` |
| `lineage_family_amr_profile` | `Lineage_family` → `Lineage_clean` → `AMR_binary` → `Resistance_Profile_Collapsed` |
| `lineage_amr_binary` | `Lineage_clean` → `AMR_binary` |

Classic two-level style still works:

```bash
python -m network_parser.cli train-hierarchy \
  --genomic /path/to/training_matrix.csv \
  --meta /path/to/metadata.csv \
  --level1_label Lineage \
  --level2_label Resistance_Profile \
  --global_level2_label AMR_binary \
  --output_dir /path/to/results/two_level
```

Useful hierarchy options:

- `--hierarchy_resume` — skip nodes that already have `node_summary.json` + model under the output directory
- `--global_fallback_labels` — which levels get cohort-wide global models (`none`, `terminal`, `lineage`, `legacy`, or comma-separated label names)
- `--no_model_bundle` / `--bundle_output` — control automatic `.npb` packaging (default: write `networkparser_model_bundle.npb`)
- low-support class drop / keep flags and AMR evidence guards (see `train-hierarchy --help`)

Registries:

- recursive hierarchy → `hierarchical_model_registry.json`
- classic two-level path → `two_level_model_registry.json`

### 3. Bundle trained models

```bash
python -m network_parser.cli bundle \
  --registry /path/to/results/hierarchy/hierarchical_model_registry.json \
  --output /path/to/models/networkparser_model_bundle.npb
```

By default the bundle embeds model payloads, selected-feature manifests, and ranked feature tables so it can be moved for inference.

> `.npb` files contain pickle-based Python model objects. Only load bundles from a trusted source.

### 4. Query new samples

With a portable bundle:

```bash
python -m network_parser.cli query \
  --genomic /path/to/query_input \
  --bundle /path/to/models/networkparser_model_bundle.npb \
  --query_input_type auto \
  --ref_fasta /path/to/reference.fasta \
  --output_dir /path/to/results/query \
  --n_jobs -1
```

With a registry that can still resolve training model paths:

```bash
python -m network_parser.cli query \
  --genomic /path/to/query_matrix.csv \
  --registry /path/to/results/hierarchy/hierarchical_model_registry.json \
  --query_input_type matrix \
  --output_dir /path/to/results/query
```

| `--query_input_type` | Expected input |
|---|---|
| `auto` | Infer from path / extensions |
| `matrix` | CSV/TSV sample-by-feature matrix |
| `vcf` | One VCF or a directory of per-sample VCFs |
| `fasta` | FASTA DNA mapped against saved marker contexts |
| `fastq` | Directory of paired-end FASTQ; preprocess to VCF/gVCF first |
| `raw_sequence` | Deprecated alias for `fasta` |

Query reconstructs only the features saved at training time. It does not re-run association testing, feature selection, or model training.

Primary query artifacts:

- `query_predictions.csv` — full machine-readable predictions
- `query_predictions_compact.tsv` — compact table
- `query_predictions_readable.html` — human-readable view
- `query_route_audit.json` — hierarchy routing, terminal status, fallbacks
- `query_alignment_summary.json` — feature recovery / callability
- `query_report.json` / `query_report.txt` — run-level interpretation

FASTQ query can use whole-genome or panel-restricted calling via config `fastq_call_mode`: `full` (default), `panel_bcftools`, or `panel_majority`.

### 5. Evaluate predictions

Single label or hierarchy levels:

```bash
python -m network_parser.cli evaluate \
  --predictions /path/to/query/query_predictions.csv \
  --meta /path/to/test_metadata.csv \
  --label AMR_binary \
  --output_dir /path/to/results/evaluation
```

```bash
python -m network_parser.cli evaluate \
  --predictions /path/to/query/query_predictions.csv \
  --meta /path/to/test_metadata.csv \
  --hierarchy_labels Lineage_Supergroup Lineage AMR_binary \
  --output_dir /path/to/results/evaluation
```

Hierarchy evaluation pack (per-level metrics, full-path accuracy, bootstrap CIs):

```bash
python -m network_parser.cli evaluate-hierarchy \
  --predictions /path/to/query/query_predictions.csv \
  --meta /path/to/test_metadata.csv \
  --hierarchy_labels Lineage_clean AMR_binary Resistance_Profile_Collapsed \
  --output_dir /path/to/results/hierarchy_eval \
  --harmonize_resistance_labels
```

Evaluation never retrains models.

### 6. Leakage-aware cross-validation

```bash
python -m network_parser.cli cross-validate \
  --genomic /path/to/training_vcfs \
  --meta /path/to/metadata.csv \
  --label Lineage \
  --ref_fasta /path/to/reference.fasta \
  --n_repeats 3 \
  --n_splits 5 \
  --output_dir /path/to/results/cross_validation \
  --n_jobs -1
```

Each fold fits feature filtering, panel selection, imputation, and model training on that fold’s training partition only. Main outputs: `cv_fold_metrics.tsv`, `cv_predictions.tsv`, `cv_feature_stability.tsv`, `cv_by_class_metrics.tsv`, plus per-repeat / per-fold artifacts.

### 7. Annotate selected panels

Post-training reporting only; does not change models.

```bash
python -m network_parser.cli annotate-panels \
  --registry /path/to/results/hierarchy/hierarchical_model_registry.json \
  --catalogue /path/to/resistance_catalogue.tsv \
  --stability /path/to/cross_validation/cv_feature_stability.tsv \
  --min_stability 0.7 \
  --write_stable_report \
  --output_dir /path/to/results/panel_annotation
```

Writes `panel_features_annotated.tsv`, summary tables, and `panel_annotation_report.json`. Optional `--write_catalogue_circularity` audits known vs non-catalogue features by node.

## Optional known-marker seed

CLI `--known_markers` is **report-only**. To **force-include** catalogue alleles during training of phenotype stages, set config options (default **off**):

```json
{
  "seed_known_markers": true,
  "known_markers_path": "/path/to/resistance_catalogue.tsv",
  "seed_known_markers_mode": "force_include",
  "seed_known_markers_stage_substrings": "amr,resistance,pheno,profile,resistant,susceptible",
  "seed_known_markers_max": null
}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `seed_known_markers` | `false` | Master switch |
| `known_markers_path` | `null` | Catalogue TSV (`Position`/`Ref`/`Alt`/`Contig`) or `Feature_ID` list |
| `seed_known_markers_mode` | `force_include` | Known markers occupy the first panel slots (`rank_boost` also available) |
| `seed_known_markers_stage_substrings` | phenotype-like names | Which hierarchy stage names receive seeding |
| `seed_known_markers_max` | `null` | Optional cap on seeded markers |

Only alleles **present in the filtered training matrix** are seeded. Lineage-only stages are skipped unless their stage name matches the substrings. Details: [`docs/KNOWN_MARKER_SEED.md`](docs/KNOWN_MARKER_SEED.md).

## Feature filtering and panels

Central filtering runs on training data before model fitting:

| Method | Behavior |
|---|---|
| `rf_fdr` | Repeated RF importance + label-permutation p-values + FDR |
| `chi2_fdr` | Chi-square association + multiple-testing correction |
| `fisher_fdr` | Fisher exact (where appropriate) + correction |
| `chi2_perm_fdr` | Chi-square with label-permutation empirical p-values + correction |

Missing observations are dropped **per feature** for association tests (never treated as zero). Default failure policy is `stop` if nothing survives correction.

After filtering, an optional panel check scores candidate top-N sizes and requires a minimum score (default `0.75`). Failing nodes are recorded as unsupported rather than silently trained on an exploratory full matrix.

## Configuration

Common options are CLI flags. `--config` accepts a JSON object whose keys match `NetworkParserConfig` fields; CLI flags override the file.

Example:

```json
{
  "qual_threshold": 30.0,
  "min_dp_per_sample": 10,
  "min_gq_per_sample": 20,
  "min_sample_presence": 10,
  "max_missing_fraction_per_sample": 0.5,
  "max_missing_fraction_per_feature": 0.5,
  "genotype_impute_strategy": "baseline",
  "central_feature_filter_method": "chi2_fdr",
  "fdr_alpha": 0.05,
  "multiple_testing_method": "fdr_bh",
  "feature_filter_fallback_strategy": "stop",
  "run_feature_panel_separability_check": true,
  "feature_panel_sizes": [100, 200, 500, 1000],
  "feature_panel_min_score": 0.75,
  "feature_panel_threshold_failure_strategy": "stop",
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

Major config groups: VCF callability, missingness limits, imputation, central FDR method, panel sizes/thresholds, known-marker seed, hierarchy fallbacks and AMR evidence guards, query recovery gates, FASTQ call mode.

Full schema: [`network_parser/config.py`](network_parser/config.py). Architecture overview: [`docs/NETWORKPARSER_FULL_PICTURE.md`](docs/NETWORKPARSER_FULL_PICTURE.md).

## Outputs

Exact files depend on command and options. Common training artifacts:

- aligned metadata / sample checkpoints
- constructed and filtered matrices
- feature-filter statistics and selected-feature lists / manifests
- fitted models and selection summaries
- hierarchy node directories + JSON registry
- optional decision-tree rules and interaction tables
- optional automatic `.npb` bundle

The registry is the authoritative map of hierarchy nodes, selected features, model paths, fallbacks, and query metadata.

## Python API

The CLI is the most complete interface. Core imports:

```python
from network_parser import (
    DataLoader,
    HierarchyProtocol,
    NetworkParser,
    NetworkParserConfig,
    NetworkParserQueryEngine,
    run_repeated_cv,
)

# TwoLevelProtocol is a backward-compatible alias for HierarchyProtocol
```

## Practical notes

- Keep training and query **reference identity** consistent (contig names, coordinates, allele orientation).
- Prefer gVCFs or other callability-aware inputs when distinguishing callable reference from missing data matters.
- Do not replace `NaN` with zero before association testing or evaluation.
- Do not select features on the full dataset before CV; use `cross-validate` or an equivalent nested procedure.
- Hierarchy child nodes train only when the parent branch has enough samples and class support; skipped nodes and fallbacks are recorded.
- FASTA mapping depends on marker contexts saved at training; prefer matrix or VCF when exact callability is required.
- FASTQ mode is convenience preprocessing, not a full validated variant-calling / QC replacement.
- Model bundles are trusted-input-only pickle artifacts.

## Troubleshooting

**No features survive filtering**

Check cohort size, class balance, missingness, minor counts, and the correction method. Keep fallback at `stop` for publication-oriented runs.

**Many calls are `NaN`**

Inspect FILTER, QUAL, DP, GQ, MQ, ploidy, and reference settings. Variant-only absence ≠ callable reference.

**Genomic and metadata sample counts differ**

Check `Sample` column, VCF header names, filename-derived IDs, duplicates, and suffix / whitespace differences.

**Low feature recovery at query**

Confirm the same reference, contig naming, and allele orientation as training. Use `contig_alias_map` only when equivalence is known.

**Hierarchy branch skipped or fallback used**

Inspect the registry, route audit, class-support tables, and query report.

## Testing

```bash
pytest -q
```

Optional static checks:

```bash
black --check network_parser tests
flake8 network_parser tests
mypy network_parser
```

## Repository layout

```text
network_parser/
├── environment.yml
├── README.md
├── docs/
│   ├── NETWORKPARSER_FULL_PICTURE.md
│   └── KNOWN_MARKER_SEED.md
├── network_parser/
│   ├── cli.py                      # CLI entry and dispatch
│   ├── config.py                   # NetworkParserConfig
│   ├── data_loader.py              # VCF / matrix loading
│   ├── vcf_call_semantics.py       # shared train/query VCF rules
│   ├── matrix_contract.py          # missingness and imputation
│   ├── feature_selection.py        # RF-FDR filtering
│   ├── statistical_validation_branch.py
│   ├── feature_panel_selection.py
│   ├── known_marker_seed.py
│   ├── ml_protocol.py
│   ├── model_selector.py
│   ├── decision_tree_branch.py
│   ├── hierarchy_protocol.py
│   ├── hierarchy_artifacts.py      # presets, registry helpers
│   ├── model_bundle.py
│   ├── query_engine.py
│   ├── sequence_query_encoder.py
│   ├── fastq_processor.py
│   ├── panel_pileup_caller.py
│   ├── cross_validation.py
│   ├── model_evaluation.py
│   ├── hierarchy_evaluation_pack.py
│   └── panel_annotation.py
├── tests/
├── scripts/
└── workflow/
```
