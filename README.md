# NetworkParser

**NetworkParser** is an interpretable genomic feature-discovery and model-selection framework for microbial variant data. It converts genomic matrices or VCF-derived variant spaces into clean sample × feature matrices, applies statistically defensible feature filtering, evaluates machine-learning suitability, and conditionally triggers decision-tree interpretability for marker ranking, rule extraction, path-based interaction mining, confidence estimation, and downstream network construction.

NetworkParser is designed for microbial genomics settings where prediction alone is not enough. The framework is intended to support robust inference by linking supervised genomic classification to interpretable marker evidence, AMR phenotype interpretation, and ML-ready matrix outputs.

---

## Core Aim

NetworkParser supports two related analysis modes:

1. **Single-label supervised discovery**  
   Use one metadata label, such as a lineage, strain group, phenotype, or AMR class, to identify discriminating genomic features and optionally run interpretable decision-tree discovery.

2. **Two-level diagnostic interpretation**  
   First place a strain or sample into a supervised genomic group, then evaluate resistance-associated patterns using a second supervised phenotype or AMR-profile label.

The long-term diagnostic question is:

> Given this genomic feature profile, where does the strain belong, what phenotype is predicted, and which genomic markers support the interpretation?

---

## High-Level Architecture

```text
Input genomic data
    ↓
Data loading and preprocessing
    ↓
Sample / metadata alignment
    ↓
Central feature filtering
    ↓
ML protocol and model selector
    ↓
Conditional decision-tree interpretability branch
    ↓
Post-tree confidence scoring and interaction mining
    ↓
Ranked markers, interaction evidence, networks, and query-ready models
```

The central methodological rule is:

> **Statistical feature filtering happens before model screening and before tree construction. Bootstrap stability, confidence values, and interaction validation happen after tree construction.**

This separation keeps the workflow statistically defensible and prevents post-model confidence outputs from being misused as the primary feature-selection layer.

---

## Key Features

- Accepts genomic matrices and VCF-derived feature spaces.
- Builds sample × genomic-feature binary matrices from per-sample VCF input where enabled.
- Applies VCF-level quality control and cohort-level feature filtering.
- Supports reference-baseline or cohort-mode baseline encoding.
- Removes invariant and low-information markers before downstream analysis.
- Applies central RF-FDR feature selection using permutation-derived empirical p-values and FDR correction.
- Supports classical association-FDR filtering as an alternative central filtering route.
- Runs an ML protocol and model selector on the centrally filtered matrix.
- Conditionally triggers the decision-tree interpretability branch.
- Extracts interpretable decision paths, branch-level rules, and path-based feature interactions.
- Computes post-tree confidence and bootstrap stability evidence.
- Produces filtered matrices, ranked marker tables, model artifacts, interaction outputs, and query reports.

---

## Command-Line Entry Points

NetworkParser currently exposes three main CLI workflows:

```text
python -m network_parser.cli run             # single-label workflow
python -m network_parser.cli train-two-level # two-level training workflow
python -m network_parser.cli query           # query/inference workflow
```

For backward compatibility, calls without a subcommand are interpreted as the single-label `run` workflow.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
```

Create and activate the environment using the project environment file:

```bash
conda env create -f environment.yml
conda activate networkparser
```

Install locally in editable mode:

```bash
pip install -e .
```

NetworkParser is designed around a restricted scientific Python stack, using common packages such as `numpy`, `pandas`, `scikit-learn`, `scipy`, `statsmodels`, and `networkx`.

---

## Input Requirements

### 1. Genomic Input

NetworkParser accepts genomic data that can be represented as a sample × feature matrix.

Supported input concepts include:

- CSV or TSV genomic feature matrices.
- Binary variant matrices.
- Per-sample VCF / VCF.GZ directories routed through the DataLoader.
- VCF-derived marker matrices produced by NetworkParser artifact generation.

Expected matrix orientation:

```text
sample_id    genomic_feature_1    genomic_feature_2    genomic_feature_3    ...
sample_A     0                    1                    0                    ...
sample_B     1                    0                    0                    ...
sample_C     0                    1                    1                    ...
```

Rows represent samples or strains. Columns represent genomic features, polymorphic sites, variant encodings, or other compatible feature representations.

### 2. Metadata Input

A metadata file is required for supervised feature filtering and model selection.

Expected metadata structure:

```text
sample_id    target_label    optional_metadata_1    optional_metadata_2
sample_A     class_A         ...                    ...
sample_B     class_B         ...                    ...
sample_C     class_A         ...                    ...
```

The supervised label may represent lineage, strain group, species-complex group, AMR phenotype, resistance profile, outbreak cluster, or another biologically meaningful classification target.

### 3. Two-Level Metadata

For two-level training, the metadata must contain two supervised label columns:

```text
sample_id    level1_label    level2_label    optional_metadata
sample_A     group_A         phenotype_A     ...
sample_B     group_B         phenotype_B     ...
```

Conceptually:

- **Level 1**: strain placement, lineage, clade, cluster, or genomic group.
- **Level 2**: AMR phenotype, resistance class, or resistance-profile label.

---

## Quick Start: Single-Label Workflow

Run the full single-label workflow:

```bash
python -m network_parser.cli run \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --label target_label_column \
  --output_dir path/to/results \
  --pipeline_mode both
```

This runs:

```text
load → align → central feature filtering → ML protocol/model selector → conditional decision-tree branch
```

### Matrix-only mode

```bash
python -m network_parser.cli run \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --label target_label_column \
  --output_dir path/to/results \
  --pipeline_mode matrix_only
```

### ML-only mode

```bash
python -m network_parser.cli run \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --label target_label_column \
  --output_dir path/to/results \
  --pipeline_mode ml_only
```

### Decision-tree interpretability mode

```bash
python -m network_parser.cli run \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --label target_label_column \
  --output_dir path/to/results \
  --pipeline_mode decision_tree_only
```

### Force decision-tree consideration through ML configuration

```bash
python -m network_parser.cli run \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --label target_label_column \
  --output_dir path/to/results \
  --pipeline_mode both \
  --ml_algorithm DT
```

---

## Quick Start: Two-Level Training

Train a two-level NetworkParser model registry:

```bash
python -m network_parser.cli train-two-level \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --level1_label strain_or_group_label \
  --level2_label phenotype_or_resistance_label \
  --output_dir path/to/two_level_results \
  --ref_fasta path/to/reference.fasta \
  --n_jobs 4
```

The two-level protocol performs:

```text
Input
  ↓
DataLoader / preprocessing
  ↓
Artifact-filtered binary matrix selection where available
  ↓
Two-label metadata alignment
  ↓
Level 1 RF-FDR feature filtering
  ↓
Level 1 model training
  ↓
Global Level 2 RF-FDR feature filtering and model training
  ↓
Level 2 per-group RF-FDR feature filtering and model training where possible
  ↓
Two-level model registry
```

The two-level registry is written as:

```text
two_level_model_registry.json
```

This registry is the main trained artifact used by query mode.

---

## Quick Start: Query / Inference

Apply a trained two-level registry to new genomic input:

```bash
python -m network_parser.cli query \
  --genomic path/to/new_genomic_input \
  --registry path/to/two_level_model_registry.json \
  --output_dir path/to/query_results \
  --ref_fasta path/to/reference.fasta \
  --max_markers 10 \
  --n_jobs 4
```

Query mode is inference-only. It does **not** rerun RF-FDR, permutation testing, FDR correction, decision-tree training, or bootstrap confidence estimation. Instead, new samples are aligned to the trained feature space stored in the registry.

Query outputs include:

```text
query_predictions.csv
query_report.json
query_report.txt
query_alignment_summary.json
```

The report contains the predicted Level 1 identity, predicted Level 2 phenotype/profile, support values where available, supporting markers, and decision-path explanations when the saved model exposes tree-like structure.

---

## Pipeline Modes

The single-label `run` workflow supports:

| Mode | Behaviour |
|---|---|
| `matrix_only` | Stop after loading, preprocessing, and sample/metadata alignment. |
| `decision_tree_only` | Run central filtering and then decision-tree interpretability. |
| `ml_only` | Run central filtering and ML protocol/model selector only. |
| `both` | Run central filtering, ML protocol/model selector, and conditional decision-tree interpretation. |

The intended workflow is:

```text
Input → preprocessing → RF-FDR feature filtering → ML protocol/model selector → conditional decision-tree interpretation → post-tree confidence and interaction outputs
```

---

## Main CLI Options

### Shared options

| Argument | Description |
|---|---|
| `--genomic` | Genomic input file or VCF directory. |
| `--output_dir` | Output directory. |
| `--config` | Optional JSON file with `NetworkParserConfig` overrides. |
| `--ref_fasta` | Optional FASTA or GenBank reference context for VCF-oriented workflows. |
| `--n_jobs` | Number of parallel workers where supported. |
| `--verbose` | Enable debug-level logging. |
| `--quiet` | Show warnings and errors only. |

### Single-label workflow options

| Argument | Description |
|---|---|
| `--meta` | Metadata CSV/TSV containing the supervised label column. |
| `--label` | Metadata column used as the supervised target. |
| `--known_markers` | Optional known-marker file for comparison or annotation. |
| `--pipeline_mode` | Select `matrix_only`, `decision_tree_only`, `ml_only`, or `both`. |
| `--validate_statistics` | Compatibility flag for validation controls where supported. |
| `--validate_interactions` | Run optional post-tree interaction validation where available. |
| `--run_ml_protocol` | Force the ML protocol branch on. |
| `--disable_central_feature_filtering` | Pass the aligned matrix forward without central filtering. |
| `--disable_model_selector` | Disable automatic model-selector behaviour. |
| `--disable_conditional_dt` | Prevent selector-driven decision-tree triggering. |

### Two-level training options

| Argument | Description |
|---|---|
| `--level1_label` | Metadata column for strain/lineage/group placement. |
| `--level2_label` | Metadata column for phenotype or resistance-profile prediction. |
| `--algorithm` | Optional ML algorithm override passed to the ML protocol. |
| `--no_global_level2` | Disable the global Level 2 fallback model. |
| `--min_level2_samples_per_group` | Minimum samples needed to train group-specific Level 2 models. |

### Query options

| Argument | Description |
|---|---|
| `--registry` | Path to `two_level_model_registry.json` from training. |
| `--max_markers` | Maximum number of supporting markers shown per level per sample. |

---

## RF-FDR Feature-Selection Controls

RF-FDR is the preferred central feature-selection stage. It uses repeated observed Random Forest importance scoring, label permutations, empirical p-values, and FDR correction.

Common CLI overrides:

| Argument | Description |
|---|---|
| `--rf_selector_n_estimators` | Number of trees used during RF-FDR scoring. |
| `--rf_selector_n_observed_repeats` | Number of repeated observed RF importance runs. |
| `--rf_selector_n_permutations` | Number of label permutations for empirical p-values. |
| `--rf_selector_fdr_alpha` | FDR threshold used for feature retention. |
| `--rf_selector_random_state` | Random seed for reproducibility. |
| `--rf_selector_top_n` | Optional cap on retained RF-FDR features. |
| `--rf_selector_min_importance` | Minimum observed RF importance for retained features. |
| `--rf_selector_fallback_strategy` | Behaviour when RF-FDR retains no features: `stop`, `top_n`, or `unfiltered`. |
| `--rf_selector_fallback_top_n` | Number of RF-ranked features retained if `top_n` fallback is enabled. |
| `--feature_filter_fallback_strategy` | Fallback for association-FDR filtering when no features survive correction. |

For publication-grade runs, use `rf_selector_fallback_strategy = "stop"` unless an exploratory fallback is explicitly justified. Increasing the number of permutations improves empirical p-value resolution and supports more robust inference.

---

## Configuration

NetworkParser uses a central `NetworkParserConfig` object. A JSON config file can override selected values:

```bash
python -m network_parser.cli run \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --label target_label_column \
  --output_dir path/to/results \
  --config config.json
```

Example configuration:

```json
{
  "pipeline_mode": "both",
  "run_central_feature_filtering": true,
  "run_rf_fdr_feature_selection": true,
  "run_model_selector": true,
  "ml_algorithm": "auto",
  "trigger_decision_tree_on_selected": true,
  "trigger_decision_tree_if_candidate": true,
  "remove_invariant": true,
  "multiple_testing_method": "fdr_bh",
  "rf_selector_n_estimators": 300,
  "rf_selector_n_observed_repeats": 10,
  "rf_selector_n_permutations": 1000,
  "rf_selector_fdr_alpha": 0.05,
  "rf_selector_fallback_strategy": "stop",
  "n_jobs": -1,
  "random_state": 42
}
```

Important config areas:

| Config area | Purpose |
|---|---|
| Input/output behaviour | Controls matrix naming, artifact writing, and output paths. |
| VCF-level QC | Controls QUAL, depth, mapping-quality, and biallelic-SNP filtering. |
| Cohort-level filtering | Controls sample-presence and low-count marker handling. |
| Binary encoding | Controls reference-baseline or cohort-mode encoding. |
| Artifact filtering controls | Controls structural marker cleanup and redundancy reduction. |
| Central feature filtering | Controls RF-FDR or association-FDR feature selection. |
| ML protocol | Controls model-selector and algorithm evaluation behaviour. |
| Decision-tree branch | Controls tree depth, split behaviour, rule extraction, and interpretability. |
| Interaction mining | Controls post-tree path-based feature-interaction discovery. |
| Bootstrap / stability | Controls post-tree confidence estimation and stability evidence. |
| Query mode | Controls query-time matrix construction and trained-feature alignment. |

---

## DataLoader Behaviour

For VCF-directory input, the DataLoader scans each sample VCF, applies record-level QC, aggregates cohort-level polymorphic sites, applies presence filtering, encodes the matrix, and writes optional matrix artifacts.

Conceptual flow:

```text
Per-sample VCF files
    ↓
Record-level QC
    ↓
Cohort merge of allele-specific polymorphic features
    ↓
Sample-presence filtering
    ↓
Baseline encoding to 0/1
    ↓
Invariant and low-count marker filtering
    ↓
Artifact writing and structural marker refinement
```

Typical DataLoader artifacts include:

```text
matrices/
├── dataloader_config.snapshot.json
├── vcf_counts/
│   └── all_snp.txt
├── fasta/
│   ├── matrix_alleles.fasta
│   ├── matrix_binary.fasta
│   └── matrix_filtered.tsv
└── matrices/
    ├── matrix_alleles.tsv
    ├── matrix_binary.tsv
    ├── matrix_alleles.fasta
    ├── matrix_binary.fasta
    └── matrix_filtered.tsv
```

The artifact-filtered binary matrix is preferred for downstream modelling when it can be aligned safely. The marker annotation table is not used as the supervised feature matrix.

---

## Output Structure

### Single-label workflow

```text
results/
├── central_feature_filtering/
│   ├── filtered_matrix.csv
│   ├── feature_filtering_summary.json
│   └── RF-FDR or association-FDR result files
├── ml_protocol/
│   └── model-selector and ML protocol outputs
├── decision_tree/
│   └── decision-tree rules, feature confidence, and interaction outputs
├── matrices/
│   └── optional DataLoader matrix artifacts
└── networkparser_results_<timestamp>.json
```

The final JSON summary records the resolved configuration, selected pipeline mode, aligned matrix shape, filtered matrix shape, feature-filtering summary, ML protocol results, decision-tree results where applicable, and validation results where applicable.

### Two-level training workflow

```text
two_level_results/
├── matrices/
│   └── DataLoader and artifact-filtered matrix outputs
├── level1_strain_identity/
│   ├── rf_fdr_filter/
│   │   ├── rf_fdr_feature_results.csv
│   │   ├── rf_fdr_retained_features.csv
│   │   ├── filtered_matrix.csv
│   │   └── feature_filtering_summary.json
│   └── model/
│       └── level-1 model outputs
├── level2_resistance_profile/
│   ├── global_fallback/
│   │   ├── rf_fdr_filter/
│   │   └── model/
│   └── by_level1_group/
│       └── group-specific Level 2 outputs where trainable
├── aligned_two_level_matrix.csv
├── aligned_two_level_labels.csv
└── two_level_model_registry.json
```

### Query workflow

```text
query_results/
├── query_matrix_artifacts/
│   └── DataLoader artifacts for the query input
├── query_predictions.csv
├── query_alignment_summary.json
├── query_report.json
└── query_report.txt
```

---

## Methodological Notes

### Central Feature Filtering

Central feature filtering happens once and upstream of model screening. This avoids inconsistent feature sets between the ML protocol and the interpretability branch.

Preferred method:

```text
RF feature importance → permutation empirical p-values → FDR correction → retained genomic features
```

Alternative method:

```text
association testing → multiple-testing correction → retained genomic features
```

### RF-FDR Interpretation

RF-FDR is a feature-filtering procedure, not the final biological explanation. It is used to reduce a high-dimensional genomic matrix into a smaller, statistically defensible feature space before model screening and decision-tree interpretation.

The retained feature list should be interpreted as a supervised marker set that survived the configured empirical and multiple-testing controls. It is not a substitute for biological validation.

### ML Protocol / Model Selector

The ML protocol receives the centrally filtered matrix rather than the raw high-dimensional matrix. This makes model screening faster, more stable, and more interpretable.

The selector can recommend candidate algorithms. The decision-tree branch is triggered only under the configured conditional logic.

### Decision-Tree Interpretability Branch

The decision-tree branch assumes central filtering has already happened. It then performs:

- tree fitting on the filtered feature matrix
- extraction of root and branch features
- rule generation
- path-based interaction mining
- post-tree confidence and stability scoring

The decision tree is retained because it gives a compact, rule-based interpretation layer. It should not be replaced by a black-box model if the goal is interpretability.

### Bootstrap and Confidence Scores

Bootstrap support and confidence scores are post-tree interpretability outputs. They answer:

> How stable is this selected feature or decision-path signal under resampling?

They do not answer:

> Which features should enter the model in the first place?

That is the role of pre-tree central feature filtering.

### Interaction Mining

Interaction mining is path-based. Candidate interactions are extracted from co-occurring features along decision-tree paths, then optionally validated downstream. This avoids exhaustive all-pairs testing across the original high-dimensional feature matrix.

### Query-Time Interpretation

Query mode aligns new samples to the feature lists stored in the trained registry. Missing trained features are filled as 0, extra query features are ignored, and RF-FDR is not rerun. This keeps inference consistent with the training-time feature space.

---

## Troubleshooting

### No overlapping sample IDs

Check that sample identifiers in the genomic matrix and metadata file refer to the same biological samples. NetworkParser normalizes common VCF suffixes, but metadata and matrix IDs must still be compatible.

### No features retained after RF-FDR

This can happen when the cohort is small, labels are weakly separated, metadata are noisy, classes are imbalanced, or the empirical p-value resolution is too coarse.

Review:

- metadata label quality
- class balance
- cohort-level feature presence thresholds
- binary encoding behaviour
- RF-FDR permutation count
- FDR threshold
- fallback strategy

For robust inference, prefer increasing permutation resolution before using exploratory fallback modes.

### Decision-tree branch did not run

In `both` mode, the decision-tree branch is conditional. It runs when configured explicitly, selected by the ML protocol, recommended, or included as a candidate depending on trigger settings.

To force decision-tree consideration:

```bash
python -m network_parser.cli run \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --label target_label_column \
  --output_dir path/to/results \
  --pipeline_mode both \
  --ml_algorithm DT
```

### Query output contains many missing trained features

This indicates that the query input was not represented in the same feature space as training. Check that the same reference, VCF parsing logic, feature-ID convention, and DataLoader settings were used.

### Group-specific Level 2 model is unavailable

A group-specific Level 2 model may be skipped if there are too few samples in that Level 1 group or if the Level 2 label has only one class within that group. If enabled, the global Level 2 fallback model is used instead.

### ML protocol fails during two-level training

By default, the two-level protocol should fail loudly if the configured ML protocol fails. This is safe because it avoids silently substituting an exploratory fallback model. An explicit fallback may be enabled only for exploratory runs.

---

## Development Priorities

NetworkParser is developed as part of a doctoral research project focused on interpretable genomic feature discovery, AMR prediction, strain classification, and GNN-ready genomic outputs.

Current priorities:

- keep the pipeline fast on modest hardware
- preserve biological interpretability
- maintain statistically defensible feature filtering
- separate pre-model filtering from post-tree confidence estimation
- support clean, documented, ML-ready and GNN-ready output matrices
- improve consistent behaviour in small-cohort, high-dimensional microbial datasets
- make query-mode reports readable for downstream biological interpretation

---

## Recommended Reporting Language

A concise methods-style description:

> NetworkParser applies a modular supervised analysis workflow in which genomic variant matrices are preprocessed, aligned to metadata labels, and reduced through central statistical feature filtering before model screening. The preferred filtering strategy uses Random Forest feature importance, permutation-derived empirical p-values, and FDR correction to retain a statistically defensible marker set. The filtered matrix is passed to an ML protocol and model selector, after which a decision-tree interpretability branch can be conditionally triggered to extract rule-based markers, path-level feature interactions, and post-tree confidence evidence. In the two-level workflow, a first model performs strain or group placement, while Level 2 models evaluate phenotype or AMR-profile prediction globally and, where supported, within Level 1 groups.

---

## Citation

A formal citation will be added when the associated manuscript or publication-equivalent output is available.

For now, cite the repository:

```text
Mfuphi N. NetworkParser: Interpretable Genomic Feature Discovery Framework.
GitHub: https://github.com/Nomlie/network_parser
```

---

## License

This project is distributed under the license specified in the repository.

---

## Author

**Nomlindelo Mfuphi**  
Bioinformatics Support Scientist  
Centre for High Performance Computing
