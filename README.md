# NetworkParser

**NetworkParser** is an interpretable genomic feature-discovery, model-selection, and query-inference framework for microbial variant data. It converts genomic matrices or VCF-derived variant spaces into clean sample × feature matrices, applies statistically defensible feature filtering, evaluates machine-learning suitability, conditionally triggers decision-tree interpretability, and exports query-ready model registries for strain placement and antimicrobial-resistance interpretation.

NetworkParser is designed for microbial genomics settings where prediction alone is not enough. The framework links supervised genomic classification to traceable marker evidence, AMR phenotype interpretation, confidence-aware rule extraction, and ML-ready / GNN-ready matrix outputs.

---

## Core Aim

NetworkParser supports three related analysis modes:

1. **Single-label supervised discovery**  
   Use one metadata label, such as a lineage, strain group, phenotype, AMR class, or outbreak cluster, to identify discriminating genomic features and optionally run interpretable decision-tree discovery.

2. **Two-level diagnostic interpretation**  
   First place a strain or sample into a supervised genomic group, then evaluate resistance-associated patterns using a second supervised phenotype or AMR-profile label.

3. **Query-time inference from a new sample**  
   Project a new sample onto the trained selected-feature space and use the saved model registry to predict strain/group placement and AMR phenotype or resistance profile. Query mode can use either a prebuilt genomic feature row/matrix or a raw FASTA-like DNA sequence when a query-ready feature manifest was saved during training.

The long-term diagnostic question is:

> Given this genomic evidence, where does the strain belong, what phenotype is predicted, and which trained genomic markers support the interpretation?

---

## High-Level Architecture

```text
Training / discovery mode
-------------------------
Input genomic data + metadata
    ↓
Data loading and preprocessing
    ↓
Feature manifest construction
    ↓
Sample / metadata alignment
    ↓
Central feature filtering
    ↓
Ranked feature-panel separability check
    ↓
ML protocol and model selector
    ↓
Conditional decision-tree interpretability branch
    ↓
Post-tree confidence scoring and interaction mining
    ↓
Ranked markers, selected feature manifests, model registry, networks, and GNN-ready outputs

Query / inference mode
----------------------
New sample
    ↓
Matrix alignment OR raw-sequence selected-marker extraction
    ↓
One-sample selected-feature matrix
    ↓
Saved Level 1 model
    ↓
Saved Level 2 model
    ↓
Prediction report + marker evidence report
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
- Carries feature annotation forward through a query-ready **feature manifest**.
- Stores feature identity, reference/alternate allele, baseline allele, encoding rule, genomic context, and annotation where available.
- Applies configurable central feature selection using RF-FDR, association-FDR, or chi-square permutation-FDR.
- Supports classical chi-square/Fisher screening as faster association-based filtering routes.
- Runs an optional ranked feature-panel separability check after FDR-based filtering and before ML training.
- Evaluates compact top-N feature panels using supervised balanced accuracy and unsupervised clustering diagnostics.
- Runs an ML protocol and model selector on the centrally filtered matrix.
- Conditionally triggers the decision-tree interpretability branch.
- Extracts interpretable decision paths, branch-level rules, and path-based feature interactions.
- Computes post-tree confidence and bootstrap stability evidence.
- Produces filtered matrices, ranked marker tables, model artifacts, interaction outputs, selected marker manifests, and query reports.
- Supports raw FASTA-like query mode by mapping saved marker-context sequences back to a user-supplied DNA sequence and reconstructing the trained selected-feature matrix.

---

## Command-Line Entry Points

NetworkParser exposes three main CLI workflows:

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

For raw-sequence query mode, NetworkParser can use external BLAST command-line tools when they are already available on `PATH`. If they are not available, `auto` mode falls back to exact context matching.

---

## Input Requirements

### 1. Training Genomic Input

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

### 2. Reference / Annotation Input

A reference FASTA or GenBank-like reference context is optional for matrix-only training, but strongly recommended when query mode must accept raw DNA sequence.

When provided, NetworkParser can carry forward a feature manifest containing:

```text
Feature_ID
chrom / contig
position
REF allele
ALT allele
baseline allele
encoding rule
context sequence
marker-centre index
gene / region annotation where available
```

This manifest becomes the bridge between the trained selected features and the raw sequence supplied later by a user in query mode.

### 3. Metadata Input

A metadata file is required for supervised feature filtering and model selection.

Expected metadata structure:

```text
sample_id    target_label    optional_metadata_1    optional_metadata_2
sample_A     class_A         ...                    ...
sample_B     class_B         ...                    ...
sample_C     class_A         ...                    ...
```

The supervised label may represent lineage, strain group, species-complex group, AMR phenotype, resistance profile, outbreak cluster, or another biologically meaningful classification target.

### 4. Two-Level Metadata

For two-level training, the metadata must contain two supervised label columns:

```text
sample_id    level1_label    level2_label    optional_metadata
sample_A     group_A         phenotype_A     ...
sample_B     group_B         phenotype_B     ...
```

Conceptually:

- **Level 1**: strain placement, lineage, clade, cluster, or genomic group.
- **Level 2**: AMR phenotype, resistance class, or resistance-profile label.

### 5. Query Input

Query mode supports two concepts:

1. **Prebuilt genomic feature row or matrix**  
   The query sample is already represented using genomic feature names compatible with training.

2. **Raw FASTA-like DNA sequence**  
   NetworkParser uses the selected marker manifest from training, maps saved context sequences against the user sequence, extracts the marker-centre nucleotide, encodes the result, and creates the one-sample selected-feature matrix required by the trained models.

Raw-sequence query mode is intended for consensus FASTA, pseudogenome FASTA, or assembled contig FASTA. Raw FASTQ reads should first be processed through an appropriate external read-processing, alignment, variant-calling, or consensus-generation workflow before NetworkParser query mode is used.

---

## Quick Start: Single-Label Workflow

Run the full single-label workflow:

```bash
python -m network_parser.cli run \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --label target_label_column \
  --output_dir path/to/results \
  --pipeline_mode both \
  --feature_panel_check on \
  --feature_panel_sizes 100,200,500,1000
```

This runs:

```text
load → align → central feature filtering → ranked feature-panel separability check → ML protocol/model selector → conditional decision-tree branch
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
  --feature_panel_check on \
  --feature_panel_sizes 100,200,500,1000 \
  --n_jobs 4
```

The two-level protocol performs:

```text
Input
  ↓
DataLoader / preprocessing
  ↓
Feature manifest construction where reference context is available
  ↓
Artifact-filtered binary matrix selection where available
  ↓
Two-label metadata alignment
  ↓
Level 1 configured central feature filtering
  ↓
Level 1 ranked feature-panel separability check
  ↓
Level 1 selected feature manifest
  ↓
Level 1 model training
  ↓
Global Level 2 configured feature filtering and feature-panel checking before model training
  ↓
Global Level 2 selected feature manifest
  ↓
Level 2 per-group configured feature filtering and feature-panel checking before model training where possible
  ↓
Group-specific selected feature manifests where possible
  ↓
Two-level model registry
```

The two-level registry is written as:

```text
two_level_model_registry.json
```

This registry is the main trained artifact used by query mode. It should store model paths, selected feature lists, selected feature manifest paths, and relevant encoding/configuration metadata.

---

## Quick Start: Query / Inference

Apply a trained two-level registry to new genomic input:

```bash
python -m network_parser.cli query \
  --genomic path/to/new_genomic_input \
  --registry path/to/two_level_model_registry.json \
  --output_dir path/to/query_results \
  --max_markers 10 \
  --n_jobs 4
```

Query mode is inference-only. It does **not** rerun central feature filtering, permutation testing, FDR correction, decision-tree training, or bootstrap confidence estimation. Instead, new samples are aligned to the trained feature space stored in the registry.

### Raw FASTA query mode

Raw FASTA sequence can be queried directly when the training run was created with a selected feature manifest containing reference context. In this mode, NetworkParser uses the saved context sequence for each selected genomic feature, maps the context back to the raw query DNA, extracts the centre nucleotide, and rebuilds the selected-feature matrix before prediction:

```bash
python -m network_parser.cli query \
  --genomic path/to/new_sample.fasta \
  --registry path/to/two_level_model_registry.json \
  --output_dir path/to/query_results \
  --query_input_type raw_sequence \
  --raw_sequence_mapping_mode auto
```

`--raw_sequence_mapping_mode auto` uses BLAST context mapping when `makeblastdb` and `blastn` are available on `PATH`, otherwise it falls back to exact flanking-context matching. Use `blast` to require BLAST, or `exact` to skip BLAST.

Raw-sequence query mode performs:

```text
raw DNA sequence
    ↓
load selected marker manifest
    ↓
map marker context sequence to query DNA
    ↓
extract nucleotide at marker centre
    ↓
compare observed nucleotide to REF / ALT / baseline allele
    ↓
encode using the same training rule
    ↓
build one-sample selected-feature matrix
    ↓
apply saved Level 1 and Level 2 models
```

Query outputs include:

```text
query_predictions.csv
query_report.json
query_report.txt
query_alignment_summary.json
raw_sequence_query_encoding/raw_sequence_selected_feature_matrix.csv
raw_sequence_query_encoding/raw_sequence_feature_calls.tsv
raw_sequence_query_encoding/raw_sequence_mapping_summary.json
```

The report contains the predicted Level 1 identity, predicted Level 2 phenotype/profile, support values where available, supporting markers, observed nucleotide evidence where available, marker recovery metrics, and decision-path explanations when the saved model exposes tree-like structure.

### Paired FASTQ query mode

FASTQ query mode accepts a directory of paired-end reads, converts each sample into a VCF.GZ file using BWA, samtools, and bcftools, then passes the resulting VCF directory through the existing NetworkParser query pathway. This is a preprocessing bridge only; query mode remains inference-only and does not rerun central statistical filtering, model training, decision-tree construction, or bootstrap confidence scoring.

Required external command-line tools on `PATH`:

```text
bwa
samtools
bcftools
```

Run FASTQ query mode:

```bash
python -m network_parser.cli query \
  --genomic path/to/paired_fastq_dir \
  --registry path/to/two_level_model_registry.json \
  --output_dir path/to/query_results \
  --query_input_type fastq \
  --ref_fasta path/to/reference.fasta \
  --fastq_threads 8 \
  --fastq_max_parallel_samples 2
```

FASTQ mode performs:

```text
paired FASTQ reads
    ↓
BWA-MEM alignment to the supplied reference
    ↓
sorted/indexed BAM generation
    ↓
bcftools VCF calling
    ↓
DataLoader VCF-directory matrix construction
    ↓
trained-feature alignment
    ↓
apply saved Level 1 and Level 2 models
```

FASTQ-specific outputs include:

```text
fastq_query_preprocessing/final/vcf/*.vcf.gz
fastq_query_preprocessing/bams/*.sorted.bam
fastq_query_preprocessing/stats/*.flagstat.txt
fastq_query_preprocessing/stats/*.alignment.stats.txt
fastq_query_preprocessing/stats/*.vcf.stats.txt
fastq_query_preprocessing/logs/*.log
fastq_query_preprocessing/fastq_processing_summary.json
```

---

## Pipeline Modes

The single-label `run` workflow supports:

| Mode | Behaviour |
|---|---|
| `matrix_only` | Stop after loading, preprocessing, and sample/metadata alignment. |
| `decision_tree_only` | Run central filtering and then decision-tree interpretability. |
| `ml_only` | Run central filtering and ML protocol/model selector only. |
| `both` | Run central filtering, ML protocol/model selector, and conditional decision-tree interpretation. |
| `two_level` | Route into two-level strain/group placement and phenotype/resistance-profile modelling where exposed by the CLI/config. |

The intended publication workflow is:

```text
Input → preprocessing → feature manifest → configurable central feature filtering → ranked feature-panel separability check → ML protocol/model selector → conditional decision-tree interpretation → post-tree confidence and interaction outputs → query-ready registry
```

---

## Main CLI Options

### Shared options

| Argument | Description |
|---|---|
| `--genomic` | Genomic input file, VCF directory, query matrix, raw FASTA, or paired FASTQ directory depending on workflow. |
| `--output_dir` | Output directory. |
| `--config` | Optional JSON file with `NetworkParserConfig` overrides. |
| `--ref_fasta` | Optional FASTA or GenBank reference context for VCF-oriented workflows and raw-sequence query support. |
| `--n_jobs` | Number of parallel workers where supported. |
| `--verbose` | Enable debug-level logging. |
| `--quiet` | Show warnings and errors only. |

### Single-label workflow options

| Argument | Description |
|---|---|
| `--meta` | Metadata CSV/TSV containing the supervised label column. |
| `--label` | Metadata column used as the supervised target. |
| `--known_markers` | Optional known-marker file for comparison or annotation. |
| `--pipeline_mode` | Select `matrix_only`, `decision_tree_only`, `ml_only`, `both`, or `two_level` where supported. |
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
| `--query_input_type` | Use `raw_sequence` when `--genomic` is raw FASTA DNA; `auto` detects common FASTA suffixes where supported. |
| `--raw_sequence_mapping_mode` | Use `auto`, `blast`, or `exact` for context-based raw-sequence feature reconstruction. |

---

## Central Feature-Selection Controls

NetworkParser supports four central feature-filtering modes: `rf_fdr`, `chi2_fdr`, `fisher_fdr`, and `chi2_perm_fdr`. RF-FDR remains the default because it captures multivariate importance patterns, while chi-square/Fisher modes provide faster association-based screening. The `chi2_perm_fdr` option keeps the classical chi-square statistic but estimates empirical p-values from label permutations before FDR correction.

Common CLI overrides:

| Argument | Description |
|---|---|
| `--central_feature_filter_method` | Choose `rf_fdr`, `chi2_fdr`, `fisher_fdr`, or `chi2_perm_fdr`. |
| `--n_permutation_tests` | Number of label permutations for `chi2_perm_fdr` and downstream permutation utilities. |
| `--fdr_alpha` | FDR alpha for association-FDR and `chi2_perm_fdr`. |
| `--multiple_testing_method` | Multiple-testing method, for example `fdr_bh` or `bonferroni`. |
| `--rf_selector_n_estimators` | Number of trees used during RF-FDR scoring. |
| `--rf_selector_n_observed_repeats` | Number of repeated observed RF importance runs. |
| `--rf_selector_n_permutations` | Number of label permutations for empirical p-values. |
| `--rf_selector_fdr_alpha` | FDR threshold used for feature retention. |
| `--rf_selector_random_state` | Random seed for reproducibility. |
| `--rf_selector_top_n` | Optional cap on retained RF-FDR features. |
| `--rf_selector_min_importance` | Minimum observed RF importance for retained features. |
| `--rf_selector_fallback_strategy` | Behaviour when RF-FDR retains no features: `stop`, `top_n`, or `unfiltered`. |
| `--rf_selector_fallback_top_n` | Number of RF-ranked features retained if `top_n` fallback is enabled. |
| `--feature_filter_fallback_strategy` | Fallback for association-FDR or chi-square permutation-FDR filtering when no features survive correction. |

For publication-grade runs, use `rf_selector_fallback_strategy = "stop"` and `feature_filter_fallback_strategy = "stop"` unless an exploratory fallback is explicitly justified. Increasing the number of permutations improves empirical p-value resolution and supports more robust inference.

---

## Ranked Feature-Panel Separability Check

The ranked feature-panel separability check is controlled by `--feature_panel_check`. It runs after central statistical filtering and before ML training or decision-tree construction. Its purpose is to avoid two common failure modes:

```text
Too many retained genomic features  → slower training and harder interpretation
Too few retained genomic features   → weak label separability and lost signal
```

The algorithm uses only the already retained filtered matrix. It does not rerun RF-FDR, chi-square/Fisher-FDR, or chi-square permutation-FDR. It also does not compute post-tree bootstrap confidence.

Conceptual flow:

```text
central filtered matrix
    ↓
rank retained features by statistical evidence
    ↓
evaluate top-N panels
    ↓
select the smallest acceptable panel, or the best available fallback
    ↓
send selected panel matrix into ML / downstream training
```

Ranking priority is:

```text
corrected p-value ascending
empirical p-value ascending
raw p-value ascending
RF mean importance descending
mutual information descending
Cramer's V descending
chi-square/statistic descending
original column order as tie-breaker
```

Each candidate top-N panel is scored with:

- supervised cross-validated balanced accuracy using a lightweight logistic-regression probe
- adjusted Rand index comparing KMeans clusters to known labels
- normalized mutual information comparing KMeans clusters to known labels
- silhouette score as an unsupervised clustering diagnostic

The default selection metric is `balanced_accuracy`, because ordinary accuracy can be misleading when AMR phenotypes or group labels are imbalanced. The clustering metrics are useful diagnostics, but they are not treated as direct prediction accuracy because cluster IDs are arbitrary.

Common CLI overrides:

| Argument | Description |
|---|---|
| `--feature_panel_check` | Use `on` or `off` to enable or disable the post-filter, pre-model panel check. |
| `--feature_panel_sizes` | Comma-separated top-N panels to evaluate, for example `100,200,500,1000`. |
| `--feature_panel_metric` | Selection metric: `balanced_accuracy`, `adjusted_rand`, `normalized_mutual_info`, or `silhouette`. |
| `--feature_panel_min_score` | Minimum score required for a panel to pass. |
| `--feature_panel_selection_rule` | Panel-selection rule: `smallest_passing`, `best_passing`, or `best_available`. |
| `--feature_panel_cv_splits` | Maximum number of stratified CV folds used by the supervised balanced-accuracy probe. |

Default behaviour:

```text
smallest_passing:
    choose the smallest top-N panel that reaches the configured minimum score

best_passing:
    choose the highest-scoring panel among panels that reach the minimum score

best_available:
    choose the highest-scoring available panel even if the minimum threshold is not reached
```

If the configured metric cannot be computed, the stage falls back to the full central filtered matrix and records the fallback reason in `feature_panel_separability_summary.json`. This keeps the pipeline operational while making the limitation visible.

Outputs:

```text
feature_panel_separability/
├── ranked_features.csv
├── panel_scores.csv
├── selected_panel_matrix.csv
└── feature_panel_separability_summary.json
```

Important interpretation:

> The feature-panel score is an internal pre-model separability diagnostic. It should not be reported as final unbiased model performance, because the same labels contributed to statistical ranking and panel evaluation. Final performance should still come from the downstream ML evaluation strategy and, where possible, independent validation.

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
  "central_feature_filter_method": "rf_fdr",
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
  "n_permutation_tests": 1000,
  "feature_filter_fallback_strategy": "stop",
  "run_feature_panel_separability_check": true,
  "feature_panel_sizes": [100, 200, 500, 1000],
  "feature_panel_metric": "balanced_accuracy",
  "feature_panel_min_score": 0.75,
  "feature_panel_selection_rule": "smallest_passing",
  "feature_panel_cv_splits": 5,
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
| Feature manifest | Carries feature identity, context, alleles, baseline, encoding, and annotation into training outputs. |
| Central feature filtering | Controls RF-FDR, association-FDR, or chi-square permutation-FDR feature selection. |
| Ranked feature-panel separability | Controls top-N panel evaluation and selected model-ready matrix construction. |
| ML protocol | Controls model-selector and algorithm evaluation behaviour. |
| Decision-tree branch | Controls tree depth, split behaviour, rule extraction, and interpretability. |
| Interaction mining | Controls post-tree path-based feature-interaction discovery. |
| Bootstrap / stability | Controls post-tree confidence estimation and stability evidence. |
| Query mode | Controls query-time matrix construction and trained-feature alignment. |
| Raw-sequence query mode | Controls selected-marker context mapping and allele extraction from user-supplied DNA sequence. |

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
Feature manifest construction
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
    ├── matrix_filtered.tsv
    └── matrix_feature_manifest.tsv
```

The artifact-filtered binary matrix is preferred for downstream modelling when it can be aligned safely. The marker annotation table is not used as the supervised feature matrix; instead, marker information is carried forward as a synchronized feature manifest.

---

## Feature Manifest

The feature manifest is a first-class training artifact. It prevents annotation from being saved and then lost before model training or query mode.

Conceptually, each retained genomic feature should be traceable through:

```text
feature ID → genomic location → allele state → encoding rule → selected model feature → query-time evidence
```

A manifest may include:

```text
Feature_ID
chrom
pos
ref
alt
baseline_allele
encoding
context_sequence
context_marker_index
gene
region_type
nucleotide_change
amino_acid_change
gene_annotation
```

During central feature filtering and ranked feature-panel selection, the matrix is reduced to the exact features used by the trained model. The manifest is reduced in parallel:

```text
all-feature manifest
    ↓ subset by central-filtered Level 1 features
    ↓ subset by selected Level 1 feature panel
Level 1 selected feature manifest

all-feature manifest
    ↓ subset by central-filtered Level 2 features
    ↓ subset by selected Level 2 feature panel
Level 2 selected feature manifest
```

The selected manifests are then saved into the two-level registry so that query mode can reconstruct the same selected-feature matrix from a new sequence.

---

## Output Structure

### Single-label workflow

```text
results/
├── central_feature_filtering/
│   ├── filtered_matrix.csv
│   ├── feature_filtering_summary.json
│   └── RF-FDR, association-FDR, or chi-square permutation-FDR result files
├── feature_panel_separability/
│   ├── ranked_features.csv
│   ├── panel_scores.csv
│   ├── selected_panel_matrix.csv
│   └── feature_panel_separability_summary.json
├── ml_protocol/
│   └── model-selector and ML protocol outputs
├── decision_tree/
│   └── decision-tree rules, feature confidence, and interaction outputs
├── matrices/
│   └── optional DataLoader matrix and feature-manifest artifacts
└── networkparser_results_<timestamp>.json
```

The final JSON summary records the resolved configuration, selected pipeline mode, aligned matrix shape, central filtered matrix shape, selected panel matrix shape where available, feature-filtering summary, feature-panel summary, ML protocol results, decision-tree results where applicable, and validation results where applicable.

### Two-level training workflow

```text
two_level_results/
├── matrices/
│   └── DataLoader matrix and feature-manifest outputs
├── level1_strain_identity/
│   ├── central-filter directory, for example rf_fdr_filter/ or chi2_perm_fdr_filter/
│   │   ├── feature-level statistical results
│   │   ├── filtered_matrix.csv
│   │   └── feature_filtering_summary.json
│   ├── feature_panel_separability/
│   │   ├── ranked_features.csv
│   │   ├── panel_scores.csv
│   │   ├── selected_panel_matrix.csv
│   │   └── feature_panel_separability_summary.json
│   ├── selected_feature_manifest.tsv
│   └── model/
│       └── level-1 model outputs
├── level2_resistance_profile/
│   ├── global_fallback/
│   │   ├── level2_class_support_filter_summary.json
│   │   ├── central-filter directory
│   │   ├── feature_panel_separability/
│   │   ├── selected_feature_manifest.tsv
│   │   └── model/
│   └── by_level1_group/
│       └── group-specific Level 2 outputs where trainable
│           └── level2_class_support_filter_summary.json
├── aligned_two_level_matrix.csv
├── aligned_two_level_labels.csv
└── two_level_model_registry.json
```

### Query workflow

```text
query_results/
├── query_matrix_artifacts/
│   └── DataLoader artifacts for matrix or VCF query input
├── raw_sequence_query_encoding/
│   ├── raw_sequence_selected_feature_matrix.csv
│   ├── raw_sequence_feature_calls.tsv
│   └── raw_sequence_mapping_summary.json
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

Alternative methods:

```text
association testing → multiple-testing correction → retained genomic features
chi-square statistic → label-permutation empirical p-values → FDR correction → retained genomic features
```

### RF-FDR Interpretation

RF-FDR is a feature-filtering procedure, not the final biological explanation. It is used to reduce a high-dimensional genomic matrix into a smaller, statistically defensible feature space before model screening and decision-tree interpretation.

The retained feature list should be interpreted as a supervised marker set that survived the configured empirical and multiple-testing controls. It is not a substitute for biological validation.

### Ranked Feature-Panel Separability Check

The feature-panel check is a compactness and separability checkpoint between statistical filtering and downstream model training. It answers:

> Among the statistically retained genomic features, what is the smallest top-ranked panel that still separates the known labels strongly enough for training?

It evaluates configured top-N panels and writes both the ranked feature table and the selected model-ready matrix. By default, the selected panel is the smallest panel that reaches the configured balanced-accuracy threshold. If no panel reaches the threshold, the stage records that the threshold was not met and uses the best available panel or passes through the full central filtered matrix when scoring fails.

This stage should be described as supervised separability, not final clustering accuracy. KMeans-based adjusted Rand index, normalized mutual information, and silhouette score are diagnostics. The default decision metric remains balanced accuracy from a lightweight supervised probe.

### ML Protocol / Model Selector

The ML protocol receives the selected panel matrix when the feature-panel check is enabled; otherwise it receives the central filtered matrix. It does not train directly on the raw high-dimensional matrix unless central filtering has been explicitly disabled. This makes model screening faster, more stable, and more interpretable.

The selector can recommend candidate algorithms. The decision-tree branch is triggered only under the configured conditional logic.

### Decision-Tree Interpretability Branch

The decision-tree branch assumes central filtering has already happened. It then performs:

- tree fitting on the filtered feature matrix
- extraction of root and branch features
- rule generation
- path-based interaction mining
- post-tree confidence and stability scoring

The decision tree is retained because it gives a compact, rule-based interpretation layer. It should not be replaced by a black-box model if the goal is publication-ready interpretability.

### Bootstrap and Confidence Scores

Bootstrap support and confidence scores are post-tree interpretability outputs. They answer:

> How stable is this selected feature or decision-path signal under resampling?

They do not answer:

> Which features should enter the model in the first place?

That is the role of pre-tree central feature filtering.

### Interaction Mining

Interaction mining is path-based. Candidate interactions are extracted from co-occurring features along decision-tree paths, then optionally validated downstream. This avoids exhaustive all-pairs testing across the original high-dimensional feature matrix.

### Query-Time Interpretation

Query mode aligns new samples to the feature lists stored in the trained registry. Missing trained features are filled conservatively, extra query features are ignored, and central feature filtering is not rerun. This keeps inference consistent with the training-time feature space.

For raw FASTA queries, the selected-feature manifest is the bridge between training and inference. It carries feature identity, reference/alternate allele, baseline allele, annotation, and context sequence forward so that query-time nucleotide extraction remains traceable.

### Raw-Sequence Query Logic

Raw-sequence query mode is not new feature discovery. It asks whether the new sequence contains the previously selected markers.

```text
trained selected feature
    ↓
saved context sequence
    ↓
context mapped to query DNA
    ↓
observed nucleotide extracted at marker centre
    ↓
encoded using training rule
    ↓
model-compatible query matrix
```

Each marker call should be reported with evidence status, for example:

```text
unique_hit
multi_hit
no_hit
low_confidence_hit
ambiguous_marker_base
unexpected_allele
```

A prediction should therefore be interpreted alongside marker recovery metrics. This supports robust inference because the user can distinguish confidently observed marker evidence from unresolved sequence evidence.

---

## Troubleshooting

### No overlapping sample IDs

Check that sample identifiers in the genomic matrix and metadata file refer to the same biological samples. NetworkParser normalizes common VCF suffixes, but metadata and matrix IDs must still be compatible.

### No features retained after RF-FDR or chi-square permutation-FDR

This can happen when the cohort is small, labels are weakly separated, metadata are noisy, classes are imbalanced, or the empirical p-value resolution is too coarse.

Review:

- metadata label quality
- class balance
- cohort-level feature presence thresholds
- binary encoding behaviour
- RF-FDR or chi-square permutation count
- FDR threshold
- fallback strategy

For robust inference, prefer increasing permutation resolution before using exploratory fallback modes.

### Feature manifest is missing after training

Raw-sequence query mode requires selected feature manifests. If the registry does not contain selected manifest paths, rerun training with reference context available and ensure DataLoader writes the feature manifest artifact.

### Raw-sequence query has low marker recovery

Low marker recovery means many selected training markers could not be confidently resolved in the query sequence.

Check:

- whether the query sequence is a consensus, pseudogenome, or assembly rather than raw reads
- whether the query sequence uses compatible contig/reference context
- whether the selected marker contexts are present in the query sequence
- whether many markers produced multi-hit, no-hit, low-confidence, or ambiguous calls
- whether exact matching is too strict for the expected sequence divergence

### Query output contains many missing trained features

This indicates that the query input was not represented in the same feature space as training. Check that the same reference, VCF parsing logic, feature-ID convention, and DataLoader settings were used. For raw-sequence query mode, check the selected feature manifest and marker recovery summary.

### Raw FASTQ reads were supplied directly

NetworkParser raw-sequence query mode is not intended to infer marker absence directly from raw reads. Raw reads should first be processed through appropriate external QC, alignment, variant-calling, or consensus-generation steps.

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

### Group-specific Level 2 model is unavailable

A group-specific Level 2 model may be skipped when the Level 1 subgroup does not support robust Level 2 training. Common reasons include too few samples, only one Level 2 class inside the group, too few samples in the smallest Level 2 class for stratified cross-validation, or no finite model-selector probe scores after filtering. In these cases, the group summary JSON records the skip reason, label-balance diagnostics, feasible CV splits, and whether prediction will use the global Level 2 fallback.

### Level 2 has rare classes that make cross-validation impossible

When a Level 2 phenotype or resistance-profile class is represented by too few samples, stratified cross-validation cannot produce valid folds. For publication-safe training, either keep the run strict and report that the class is underpowered, or explicitly enable the Level 2 class-support gate:

```bash
python -m network_parser.cli train-two-level \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --level1_label lineage_column \
  --level2_label phenotype_column \
  --output_dir path/to/two_level_results \
  --level2_drop_low_support_classes \
  --level2_min_class_count 2
```

This removes only Level 2 classes below the configured support threshold before Level 2 statistical filtering and model screening. Level 1 training still uses the aligned cohort. The audit files `level2_class_support_filter_summary.json` and, when needed, `level2_low_support_classes.tsv` record what was excluded.

### Use a global binary resistant/susceptible fallback for underrepresented lineages

For datasets where detailed Level 2 resistance-profile classes are sparse within some lineages, NetworkParser can train an additional global binary Level 2 fallback model. This model asks a broader question across all lineages: resistant versus susceptible for the selected antibiotic endpoint. It is used at query time only when the predicted Level 1 group does not have a usable group-specific Level 2 model and the detailed global Level 2 fallback is unavailable.

Use a dedicated binary metadata column when available:

```bash
python -m network_parser.cli train-two-level \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --level1_label lineage_column \
  --level2_label detailed_resistance_profile_column \
  --output_dir path/to/two_level_results \
  --level2_train_binary_global_fallback \
  --level2_binary_label_column antibiotic_binary_column
```

Or derive the binary endpoint from a controlled mapping file:

```bash
python -m network_parser.cli train-two-level \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --level1_label lineage_column \
  --level2_label detailed_resistance_profile_column \
  --output_dir path/to/two_level_results \
  --level2_train_binary_global_fallback \
  --level2_binary_label_mapping_file level2_to_binary_mapping.tsv
```

The mapping file should contain:

```text
original_level2_label    binary_level2_label
detailed_profile_A       resistant
detailed_profile_B       susceptible
```

The binary model is saved under `level2_resistance_profile/global_binary_fallback/` and recorded in the registry as `level2.global_binary_fallback`. Query mode reports `level2_model_source=global_binary_fallback` when this broader endpoint is used.

### ML protocol fails during two-level training

By default, the two-level protocol should fail loudly if the configured ML protocol fails. This is publication-safe because it avoids silently substituting an exploratory fallback model. An explicit fallback may be enabled only for exploratory runs.

---

## Development Priorities

NetworkParser is developed as part of a doctoral research project focused on interpretable genomic feature discovery, AMR prediction, strain classification, and GNN-ready genomic outputs.

Current priorities:

- keep the pipeline fast on modest hardware
- preserve biological interpretability
- maintain statistically defensible feature filtering
- carry feature annotation and context through training and query mode
- separate pre-model filtering from post-tree confidence estimation
- support clean, documented, ML-ready and GNN-ready output matrices
- improve consistent behaviour in small-cohort, high-dimensional microbial datasets
- make query-mode reports readable for downstream biological interpretation

---

## Recommended Reporting Language

A concise methods-style description:

> NetworkParser applies a modular supervised analysis workflow in which genomic variant matrices are preprocessed, aligned to metadata labels, and reduced through central statistical feature filtering before model screening. The default filtering strategy uses Random Forest feature importance, permutation-derived empirical p-values, and FDR correction to retain a statistically defensible marker set, while configurable chi-square/Fisher alternatives support faster association-based screening. A ranked feature-panel separability check can then evaluate compact top-N marker panels and forward a selected model-ready matrix to the ML protocol and model selector, after which a decision-tree interpretability branch can be conditionally triggered to extract rule-based markers, path-level feature interactions, and post-tree confidence evidence. In the two-level workflow, a first model performs strain or group placement, while Level 2 models evaluate phenotype or AMR-profile prediction globally and, where supported, within Level 1 groups. Query mode applies the trained registry to new samples by projecting them onto the saved selected-feature space; for raw FASTA input, selected marker context sequences are mapped back to the query DNA, marker-centre nucleotides are extracted, and a one-sample selected-feature matrix is reconstructed before prediction.

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


### Separate global Level 2 fallback label

For datasets where the detailed Level 2 phenotype/profile is too sparse for a
robust global fallback, the standard global Level 2 fallback can be trained on a
different metadata column while group-specific models continue to use the detailed
`--level2_label`. This is useful when `--level2_label` is a detailed resistance
profile, but the fallback should answer a broader endpoint such as resistant vs
susceptible.

```bash
python -m network_parser.cli train-two-level \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --level1_label Lineage_clean \
  --level2_label Resistance_Profile_Collapsed \
  --global_level2_label AMR_binary \
  --output_dir path/to/two_level_results
```

In this mode:

```text
group-specific Level 2 models → trained on Resistance_Profile_Collapsed
standard global Level 2 fallback → trained on AMR_binary
```

The registry records both `level2.label_column` and `level2.global_label_column`
so query reports can state whether the returned Level 2 prediction came from a
detailed group-specific model or from the broader global fallback endpoint.


### Query feature-recovery diagnostics

FASTA query mode now reports feature recovery separately from model prediction. This is important because a selected feature can be recovered from the query genome but still encode as `0` when the query carries the baseline allele.

Additional query outputs include:

```text
query_predictions.csv
query_alignment_summary.json
query_report.json
query_report.txt
fasta_query_encoding/fasta_mapping_summary.json
fasta_query_encoding/fasta_sample_mapping_summary.tsv
fasta_query_encoding/fasta_feature_calls.tsv
```

The prediction table includes confidence-oriented fields such as:

```text
level1_marker_recovery_status
level2_marker_recovery_status
level1_interpretation_confidence
level2_interpretation_confidence
level1_confidence_note
level2_confidence_note
```

The FASTA mapping summary separates:

```text
selected features requested
features with unique context mapping
features mapped to baseline allele
features mapped to ALT / known non-baseline allele
features unresolved or missing context
features with multi-hit context
features with ambiguous bases
features with non-training alleles
```

A prediction can therefore be generated but still marked as low confidence when selected-feature recovery, model support, or active supporting-marker evidence is weak. This keeps query-mode inference statistically defensible and prevents baseline-filled query rows from being over-interpreted.

Preferred FASTA query command:

```bash
python -m network_parser.cli query \
  --genomic path/to/query_assembly.fna \
  --registry path/to/two_level_model_registry.json \
  --output_dir path/to/query_results \
  --query_input_type fasta \
  --fasta_mapping_mode auto
```

`raw_sequence` is still accepted as a backward-compatible alias for `fasta`, but new documentation and examples should use `fasta`.
