# NetworkParser

**NetworkParser** is an interpretable framework for statistically defensible genomic feature discovery, hierarchical model training, and query-time inference from microbial variant data. It converts genomic matrices, VCF-derived variant spaces, FASTA assemblies, or FASTQ-derived query calls into aligned sample × genomic-feature matrices, applies pre-model feature filtering with multiple-testing control, evaluates compact feature panels, trains query-ready model registries, and preserves marker-level evidence for strain placement and antimicrobial-resistance interpretation.

NetworkParser is built for microbial genomics workflows where prediction must remain biologically traceable. The pipeline keeps discovery, validation, and inference separated: statistical filtering happens before model construction; decision-tree interpretation, bootstrap stability, confidence scoring, and path-based interaction mining happen after tree construction; query mode reuses the saved selected-feature space without rerunning training-time statistics. Outputs include ranked marker panels, selected-feature manifests, model bundles, readable query reports, interaction graphs, sample-feature networks, and ML-ready / GNN-ready matrices.

---

## Core Aim

NetworkParser supports four related analysis concepts:

1. **Single-label supervised discovery**  
   Use one metadata label, such as strain group, lineage, phenotype, AMR class, or outbreak cluster, to identify discriminating genomic features and optionally run decision-tree interpretability.

2. **Two-level diagnostic interpretation**  
   First place a strain or sample into a supervised genomic group, then evaluate phenotype or AMR-profile patterns using a second supervised label.

3. **Recursive multi-level hierarchy training**  
   Train a true ordered hierarchy of supervised labels using `--hierarchy_labels`, for example demographic grouping → species/group placement → lineage/clade placement → phenotype interpretation. Each node can learn its own feature-filtered model where the data support it.

4. **Query-time inference from a new sample**  
   Project a new query sample onto the saved selected-feature space and use the trained registry or binary bundle to predict the most likely placement and phenotype/profile, together with supporting trained genomic markers.

The long-term diagnostic question is:

> Given this genomic evidence, where does the strain belong, what phenotype is predicted, and which trained genomic markers support the interpretation?

---

## Methodological Boundary

The central methodological rule is:

> **Statistical feature filtering happens before model screening and before tree construction. Bootstrap stability, confidence values, and interaction validation happen after tree construction.**

This separation keeps the workflow statistically defensible:

```text
pre-model evidence        post-tree interpretation
------------------        ------------------------
χ² / Fisher / RF-FDR  →    decision paths
FDR-BH correction     →    bootstrap stability
feature retention     →    confidence values
panel selection       →    path-based interactions
```

Query mode is inference-only. It does not rerun RF-FDR, chi-square/Fisher testing, permutation testing, FDR correction, model selection, decision-tree fitting, bootstrapping, or confidence-score computation.

---

## High-Level Architecture

```text
Training / discovery mode
-------------------------
Input genomic data + metadata
    ↓
Data loading and preprocessing
    ↓
Remove invariant features; treat missing values as baseline where configured
    ↓
Feature manifest construction when reference context is available
    ↓
Sample / metadata alignment
    ↓
Central statistical feature filtering
    ↓
Ranked feature-panel separability check
    ↓
ML protocol and model selector
    ↓
Conditional decision-tree interpretability branch
    ↓
Post-tree bootstrap confidence and path-based interaction mining
    ↓
Ranked marker lists, selected-feature manifests, model registry, binary bundle,
interaction graph, sample-feature network, and GNN-ready matrices

Query / inference mode
----------------------
New sample
    ↓
Matrix / VCF / FASTA / FASTQ route
    ↓
Trained selected-feature alignment
    ↓
Saved model registry or binary bundle
    ↓
Prediction report + marker evidence report + route audit
```

---

## Key Features

- Accepts genomic matrices, binary variant matrices, per-sample VCF / VCF.GZ directories, FASTA assemblies, and paired FASTQ query directories.
- Builds sample × genomic-feature matrices from VCF-derived variant spaces.
- Applies VCF-level quality control and cohort-level feature filtering.
- Supports reference-baseline or cohort-mode baseline encoding.
- Removes invariant and low-information genomic features before downstream analysis.
- Carries feature identity, allele states, baseline definition, genomic context, and annotation through a query-ready feature manifest.
- Applies central feature selection using RF-FDR, chi-square/FDR, Fisher/FDR, or chi-square permutation-FDR.
- Runs ranked feature-panel separability after central filtering and before model training.
- Evaluates compact top-N feature panels with supervised balanced accuracy and unsupervised clustering diagnostics.
- Runs an ML protocol and model selector on the filtered or selected-panel matrix.
- Retains decision-tree interpretability as the rule-based explanation layer.
- Extracts decision paths, branch-level rules, and path-based epistatic interaction candidates.
- Computes post-tree confidence and bootstrap stability evidence.
- Trains two-level and recursive multi-level hierarchy registries.
- Supports portable binary `.npb` model bundles for query deployment when a bundle has been produced.
- Supports query inference from a registry or binary bundle.
- Produces terminal-friendly, machine-readable, and browser-readable query reports.
- Exports ranked feature lists, selected feature manifests, interaction outputs, sample-feature networks, and GNN-ready adjacency matrices where enabled.

---

## Command-Line Entry Points

NetworkParser exposes three CLI workflows:

```text
python -m network_parser.cli run             # single-label workflow
python -m network_parser.cli train-two-level # two-level or recursive hierarchy training
python -m network_parser.cli query           # query / inference workflow
```

For backward compatibility, calls without a subcommand are interpreted as the single-label `run` workflow.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate networkparser
```

Install locally in editable mode:

```bash
pip install -e .
```

NetworkParser is designed around a restricted scientific Python stack, using common packages such as `numpy`, `pandas`, `scikit-learn`, `scipy`, `statsmodels`, and `networkx`.

For FASTA query mode, external BLAST tools can be used when `makeblastdb` and `blastn` are available on `PATH`. If they are not available and mapping mode is `auto`, NetworkParser falls back to exact flanking-context matching.

For FASTQ query mode, the following external command-line tools must be available on `PATH`:

```text
bwa
samtools
bcftools
```

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

A reference FASTA or GenBank reference is optional for matrix-only training, but strongly recommended when query mode must accept FASTA input.

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

This manifest becomes the bridge between the trained selected features and the new sequence supplied during query mode.

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

### 5. Multi-Level Hierarchy Metadata

For recursive hierarchy training, provide an ordered list of metadata columns:

```text
sample_id    hierarchy_level_1    hierarchy_level_2    hierarchy_level_3    final_label
sample_A     group_A              subgroup_A           clade_A              phenotype_A
sample_B     group_B              subgroup_B           clade_B              phenotype_B
```

The order supplied to `--hierarchy_labels` defines the training route from broad grouping to increasingly specific supervised labels.

### 6. Query Input

Query mode supports:

1. **Prebuilt genomic feature row or matrix**  
   The query sample is already represented using feature names compatible with training.

2. **VCF / VCF.GZ input**  
   The query sample is converted into the same feature convention where possible, then aligned to the saved selected features.

3. **FASTA input**  
   NetworkParser uses the selected-feature manifest from training, maps saved context sequences against the query DNA, extracts the marker-centre nucleotide, encodes the result, and creates the one-sample selected-feature matrix required by the trained models.

4. **FASTQ input**  
   NetworkParser converts paired-end FASTQ reads into per-sample VCF.GZ files using external alignment and variant-calling tools, then uses the VCF query pathway.

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
  --central_feature_filter_method chi2_fdr \
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
  --central_feature_filter_method chi2_fdr \
  --ref_fasta path/to/reference.fasta \
  --output_dir path/to/two_level_results \
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
Global Level 2 configured feature filtering and feature-panel checking
  ↓
Global Level 2 selected feature manifest
  ↓
Group-specific Level 2 configured feature filtering and feature-panel checking where possible
  ↓
Group-specific selected feature manifests where possible
  ↓
Two-level model registry and optional portable model bundle
```

Core training artifacts:

```text
two_level_model_registry.json
networkparser_model_bundle.npb       # optional bundle artifact
```

The registry records model paths, selected feature lists, selected feature manifest paths, label columns, fallback routes, and relevant encoding/configuration metadata. When a binary bundle is produced, it packages the registry, trained model payloads, selected-feature metadata, and query-time evidence resources into one portable object.

---

## Quick Start: Recursive Multi-Level Hierarchy Training

Train an ordered supervised hierarchy with more than two labels:

```bash
python -m network_parser.cli train-two-level \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --hierarchy_labels level_1_label level_2_label level_3_label final_label \
  --central_feature_filter_method chi2_fdr \
  --ref_fasta path/to/reference.fasta \
  --output_dir path/to/hierarchy_results \
  --feature_panel_check on \
  --feature_panel_sizes 100,200,500,1000 \
  --n_jobs 4
```

When `--hierarchy_labels` is supplied, it supersedes `--level1_label` and `--level2_label`. NetworkParser trains one model per eligible hierarchy node and stores the recursive routing structure in the registry.

Conceptual flow:

```text
root model
  ↓
child model for predicted Level 1 group
  ↓
child model for predicted Level 2 group
  ↓
continue until terminal hierarchy label
```

This is useful when the biological interpretation naturally moves from broad population structure to more specific lineage, clade, phenotype, or resistance endpoints.

---

## Quick Start: Query / Inference

Apply a trained registry to new genomic input:

```bash
python -m network_parser.cli query \
  --genomic path/to/new_genomic_input \
  --registry path/to/two_level_model_registry.json \
  --output_dir path/to/query_results \
  --max_markers 10 \
  --n_jobs 4
```

Apply a portable binary bundle:

```bash
python -m network_parser.cli query \
  --genomic path/to/new_genomic_input \
  --bundle path/to/networkparser_model_bundle.npb \
  --output_dir path/to/query_results \
  --max_markers 10 \
  --n_jobs 4
```

A `.npb` path passed through `--registry` is treated as a bundle for backward compatibility, but `--bundle` is the clearer route.

Query mode aligns new samples to the trained feature space stored in the registry or bundle. Missing trained features are filled conservatively, extra query features are ignored, and the alignment summary records the burden of missing trained markers.

### FASTA query mode

FASTA sequence can be queried directly when training saved selected-feature manifests with reference context:

```bash
python -m network_parser.cli query \
  --genomic path/to/new_sample.fasta \
  --registry path/to/two_level_model_registry.json \
  --output_dir path/to/query_results \
  --query_input_type fasta \
  --fasta_mapping_mode auto
```

`--raw_sequence_mapping_mode` is retained as an alias for `--fasta_mapping_mode`. The `raw_sequence` query input type is retained as an alias for `fasta`.

FASTA query mode performs:

```text
FASTA DNA sequence
    ↓
load selected-feature manifest
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
apply saved models
```

### FASTQ query mode

FASTQ query mode accepts a directory of paired-end reads, converts each sample into a VCF.GZ file using BWA, samtools, and bcftools, then passes the resulting VCF directory through the existing query pathway:

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
apply saved models
```

---

## Query Outputs

Query mode writes several files intended for different users:

```text
query_results/
├── query_predictions.csv
├── query_predictions_compact.tsv
├── query_predictions_readable.html
├── query_route_audit.json
├── query_report.json
├── query_report.txt
├── query_alignment_summary.json
├── query_matrix_artifacts/
├── fasta_query_encoding/ or raw_sequence_query_encoding/
│   ├── selected_feature_matrix.csv
│   ├── feature_calls.tsv
│   └── mapping_summary.json
└── fastq_query_preprocessing/                 # FASTQ route only
    ├── final/vcf/*.vcf.gz
    ├── bams/*.sorted.bam
    ├── stats/*
    ├── logs/*
    └── fastq_processing_summary.json
```

The report contains predicted hierarchy or Level 1 identity, predicted Level 2 phenotype/profile where available, support values where exposed by the saved model, supporting markers, marker recovery metrics, active feature evidence, route audit information, and per-feature evidence calls for FASTA-derived queries.

Supporting markers are not restricted to non-baseline states only. A resolved training marker can support interpretation whether the query carries the baseline state or a non-baseline state, as long as the feature is resolved and belongs to the trained selected-feature space.

---

## Pipeline Modes

The single-label `run` workflow supports:

| Mode | Behaviour |
|---|---|
| `matrix_only` | Stop after loading, preprocessing, and sample/metadata alignment. |
| `decision_tree_only` | Run central filtering and then decision-tree interpretability. |
| `ml_only` | Run central filtering and ML protocol/model selector only. |
| `both` | Run central filtering, ML protocol/model selector, and conditional decision-tree interpretation. |

The intended publication workflow is:

```text
Input → preprocessing → feature manifest → central statistical feature filtering → ranked feature-panel separability check → ML protocol/model selector → conditional decision-tree interpretation → post-tree confidence and interaction outputs → query-ready registry / bundle
```

---

## Main CLI Options

### Shared options

| Argument | Description |
|---|---|
| `--genomic` | Genomic input file or directory. Meaning depends on workflow: matrix, VCF, FASTA, or FASTQ directory. |
| `--output_dir` | Output directory. |
| `--config` | Optional JSON file with `NetworkParserConfig` overrides. |
| `--ref_fasta` | Optional FASTA or GenBank reference context for VCF-oriented workflows, FASTA query support, and FASTQ alignment. |
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
| `--ml_algorithm` | Optional algorithm override, for example `auto`, `RF`, `MLP`, `LR`, `DT`, `SVC`, `MBCS`, or `DNL`. |

### Two-level and hierarchy training options

| Argument | Description |
|---|---|
| `--level1_label` | Metadata column for strain/lineage/group placement. Required unless `--hierarchy_labels` is used. |
| `--level2_label` | Metadata column for phenotype or resistance-profile prediction. Required unless `--hierarchy_labels` is used. |
| `--hierarchy_labels` | Ordered metadata columns for recursive hierarchy training. Supersedes `--level1_label` and `--level2_label`. |
| `--global_level2_label` | Optional broader metadata column for the standard global Level 2 fallback. |
| `--algorithm` | Optional ML algorithm override passed to the ML protocol. |
| `--no_global_level2` | Disable the global Level 2 fallback model. |
| `--min_level2_samples_per_group` | Optional absolute minimum for group-specific Level 2 models. When unset, eligibility is adaptive. |
| `--level2_drop_low_support_classes` | Exclude Level 2 classes below the configured support threshold before Level 2 filtering and model screening. |
| `--level2_min_class_count` | Minimum samples per Level 2 class when low-support class filtering is enabled. |
| `--level2_train_binary_global_fallback` | Train an additional resistant/susceptible global Level 2 fallback. |
| `--level2_binary_label_column` | Metadata column containing the binary fallback endpoint. |
| `--level2_binary_label_mapping_file` | Mapping file for collapsing detailed Level 2 labels into a binary endpoint. |

### Query options

| Argument | Description |
|---|---|
| `--registry` | Path to `two_level_model_registry.json` from training. A `.npb` path is treated as `--bundle`. |
| `--bundle` | Path to `networkparser_model_bundle.npb`; preferred for portable query inference. |
| `--max_markers` | Maximum number of supporting markers shown per prediction level per sample. |
| `--query_input_type` | Use `auto`, `matrix`, `vcf`, `fasta`, `raw_sequence`, or `fastq`. |
| `--fasta_mapping_mode` | Use `auto`, `blast`, or `exact` for FASTA context mapping. |
| `--raw_sequence_mapping_mode` | Alias for `--fasta_mapping_mode`. |
| `--fastq_threads` | Total threads available to FASTQ preprocessing. |
| `--fastq_max_parallel_samples` | Number of paired FASTQ samples processed concurrently. |
| `--fastq_memory_per_sample_mb` | Optional memory guard per FASTQ sample. |
| `--fastq_clean_intermediates` | Remove intermediate working files after successful FASTQ preprocessing. |
| `--fastq_no_auto_index_reference` | Do not automatically create missing reference indexes. |
| `--fastq_min_mapping_quality` | Minimum mapping quality used by FASTQ-derived variant calling. |

---

## Central Feature-Selection Controls

NetworkParser supports four central feature-filtering modes:

```text
rf_fdr
chi2_fdr
fisher_fdr
chi2_perm_fdr
```

`rf_fdr` uses repeated random-forest feature importance against label permutations, then applies FDR correction. `chi2_fdr` and `fisher_fdr` provide faster association-based routes. `chi2_perm_fdr` keeps a chi-square association statistic but estimates empirical p-values by label permutation before FDR correction.

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
| `--feature_filter_fallback_strategy` | Fallback for association-FDR or chi-square permutation-FDR when no features survive correction. |

For robust inference, prefer strict fallback settings and increase permutation resolution before using exploratory fallback modes.

---

## Ranked Feature-Panel Separability Check

The ranked feature-panel separability check runs after central statistical filtering and before ML training or decision-tree construction.

Conceptual flow:

```text
central filtered matrix
    ↓
rank retained genomic features by statistical evidence
    ↓
evaluate top-N panels
    ↓
select the smallest acceptable panel or best available fallback
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

The default selection metric is `balanced_accuracy`, because ordinary accuracy can be misleading when AMR phenotypes or group labels are imbalanced. Clustering metrics are written as diagnostics but should not be interpreted as direct prediction accuracy.

Outputs:

```text
feature_panel_separability/
├── ranked_features.csv
├── panel_scores.csv
├── selected_panel_matrix.csv
└── feature_panel_separability_summary.json
```

Important interpretation:

> The feature-panel score is an internal pre-model separability diagnostic. It should not be reported as final unbiased model performance, because the same labels contributed to statistical ranking and panel evaluation.

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
  "central_feature_filter_method": "chi2_fdr",
  "run_model_selector": true,
  "ml_algorithm": "auto",
  "trigger_decision_tree_on_selected": true,
  "trigger_decision_tree_if_candidate": true,
  "remove_invariant": true,
  "multiple_testing_method": "fdr_bh",
  "feature_filter_fallback_strategy": "stop",
  "run_feature_panel_separability_check": true,
  "feature_panel_sizes": [100, 200, 500, 1000],
  "feature_panel_metric": "balanced_accuracy",
  "feature_panel_selection_rule": "smallest_passing",
  "build_model_bundle": true,
  "model_bundle_filename": "networkparser_model_bundle.npb",
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
| Two-level / hierarchy training | Controls Level 1, Level 2, recursive hierarchy, and fallback behaviour. |
| Model bundle | Controls portable query bundle naming and packaging behaviour where bundle creation is invoked. |
| Query mode | Controls query-time matrix construction and trained-feature alignment. |
| FASTA query mode | Controls selected-marker context mapping and allele extraction. |
| FASTQ query mode | Controls read-alignment and VCF-generation preprocessing. |

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

The artifact-filtered binary matrix is preferred for downstream modelling when it can be aligned safely. The marker annotation table is not used as the supervised feature matrix; marker information is carried forward as a synchronized feature manifest.

---

## Feature Manifest

The selected-feature manifest is the bridge between training-time discovery and query-time inference. It keeps the biological meaning of each retained genomic feature attached to the model-ready matrix, so a prediction can be traced back to marker identity, allele state, baseline definition, context mapping, and annotation.

Conceptually, each retained genomic feature should remain traceable through:

```text
feature ID → genomic location → allele state → encoding rule → selected model feature → query-time evidence
```

### Training / Discovery Side

```text
Genomic input                    Metadata input
(VCF / matrix)                   (labels: group, phenotype, AMR profile)
      │                                      │
      │                                      │
      ▼                                      ▼
Clean genomic matrix                  Supervised labels
sample × genomic features             sample → target label
      │                                      │
      └───────────────┬──────────────────────┘
                      ▼
        Central statistical feature filtering
        χ² / Fisher / RF-FDR / permutation-FDR
                      │
                      ▼
              Filtered matrix
      only retained genomic features enter modelling
                      │
                      ▼
        Hierarchy (/ˈhʌɪ(ə)rɑːki/)-based model training
        ordered targets from broad placement to phenotype/profile
                      │
                      ▼
       Selected feature lists saved in registry
                      │
                      ▼
        Selected-feature manifest saved
        ┌────────────────────────────────────────────┐
        │ Feature_ID                                 │
        │ genomic position / contig                  │
        │ REF allele                                 │
        │ ALT allele                                 │
        │ baseline allele                            │
        │ encoding rule                              │
        │ context sequence around marker             │
        │ marker-centre offset                       │
        │ gene / region annotation                   │
        └────────────────────────────────────────────┘
                      │
                      ▼
              Query-ready model registry / bundle
```

During central feature filtering and ranked feature-panel selection, the matrix is reduced to the exact genomic features used by each trained model slot. The manifest is reduced in parallel and saved as the selected-feature manifest for the corresponding hierarchy node, registry entry, or bundled model payload.

### Query / Inference Side

```text
New query sample
(matrix, VCF-derived matrix, FASTQ-derived calls, or FASTA sequence)
                      │
                      ▼
          Load saved model registry / bundle
          "Which features do the trained models need?"
                      │
                      ▼
          Load selected-feature manifest
          "What does each feature mean biologically?"
                      │
                      ▼
        For FASTA query:
        map saved context sequence to query DNA
                      │
                      ▼
        Extract nucleotide at marker centre
                      │
                      ▼
        Compare observed nucleotide with:
        REF / ALT / baseline allele from manifest
                      │
                      ▼
        Encode using the same training rule
        baseline = 0, known non-baseline = 1
                      │
                      ▼
        Build one-sample selected-feature matrix
        same columns/order expected by trained model
                      │
                      ▼
        Apply saved hierarchy model(s)
        broad placement → phenotype / resistance-profile interpretation
                      │
                      ▼
        Prediction report + marker evidence report
```

Query mode does not rerun RF-FDR, chi-square/Fisher FDR, permutation testing, model selection, tree construction, or bootstrap confidence scoring. It projects the new sample into the saved selected-feature space and reports which trained genomic markers were resolved, unresolved, baseline-matching, or non-baseline-matching.

Selected manifests are stored in the registry and bundle so query mode can reconstruct the same selected-feature matrix from a new query sample with confidence and robust inference traceability.

---

## Model Registry and Binary Bundle

The JSON registry is transparent and easy to inspect. It records the model hierarchy, selected features, manifest paths, fallback logic, and training summaries.

The binary bundle is portable and query-ready. It can include:

```text
registry payload
trained model payloads
selected-feature manifests
ranked feature tables
feature hashes
runtime metadata
```

Bundle creation is controlled by:

```text
build_model_bundle
model_bundle_filename
model_bundle_include_model_payloads
model_bundle_include_feature_manifests
model_bundle_include_ranked_feature_tables
model_bundle_fail_on_error
```

For most query use, a bundle is preferable when available because it avoids broken relative paths when results are moved between machines.

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

### Two-level training workflow

```text
two_level_results/
├── matrices/
│   └── DataLoader matrix and feature-manifest outputs
├── level1_*/
│   ├── central-filter directory
│   ├── feature_panel_separability/
│   ├── selected_feature_manifest.tsv
│   └── model/
├── level2_*/
│   ├── global_fallback/
│   ├── global_binary_fallback/          # when enabled
│   └── by_level1_group/
├── aligned_two_level_matrix.csv
├── aligned_two_level_labels.csv
├── two_level_model_registry.json
└── networkparser_model_bundle.npb       # optional bundle artifact
```

### Recursive hierarchy workflow

```text
hierarchy_results/
├── matrices/
├── hierarchy_models/
│   ├── root model outputs
│   └── child-node model outputs where trainable
├── hierarchical_model_registry.json
└── networkparser_model_bundle.npb       # optional bundle artifact when applicable
```

### Query workflow

```text
query_results/
├── query_predictions.csv
├── query_predictions_compact.tsv
├── query_predictions_readable.html
├── query_route_audit.json
├── query_report.json
├── query_report.txt
├── query_alignment_summary.json
├── query_matrix_artifacts/
├── fasta_query_encoding/ or raw_sequence_query_encoding/
└── fastq_query_preprocessing/           # FASTQ route only
```

---

## Evaluation and Cross-Validation Utilities

NetworkParser includes evaluation utilities for labelled holdout or repeated cross-validation workflows.

### Held-out prediction evaluation

`model_evaluation.evaluate_predictions()` compares predicted labels to true labels and writes:

```text
model_performance_summary.json
model_performance_by_class.tsv
confusion_matrix.tsv
roc_auc_summary.tsv
roc_curve_points.tsv
pr_curve_points.tsv
evaluated_sample_predictions.tsv
```

Metrics include accuracy, balanced accuracy, macro/weighted F1, sensitivity/recall, specificity, PPV, NPV, MCC, and one-vs-rest ROC/PR summaries when class-support scores are available.

### Leakage-aware repeated CV

`cross_validation.run_repeated_cv()` follows the correct statistical order:

```text
split samples first
    ↓
fit feature filtering on training fold only
    ↓
run feature-panel selection on training fold only
    ↓
train model on training fold only
    ↓
evaluate held-out fold
```

It writes:

```text
cross_validation_summary.json
cv_fold_metrics.tsv
cv_predictions.tsv
cv_feature_stability.tsv
cv_by_class_metrics.tsv
aggregate_performance/
```

This wrapper is intended for robust inference because held-out samples do not drive supervised marker selection.

---

## Methodological Notes

### Central Feature Filtering

Central feature filtering happens upstream of model screening and tree construction. This avoids inconsistent feature sets between the ML protocol and interpretability branch.

Preferred strict workflow:

```text
feature-level evidence → multiple-testing correction → retained genomic features → model-ready panel → model training
```

### RF-FDR Interpretation

RF-FDR is a feature-filtering procedure, not the final biological explanation. It reduces a high-dimensional genomic matrix into a statistically defensible feature space before model screening and decision-tree interpretation.

The retained feature list should be interpreted as a supervised marker set that survived configured empirical and multiple-testing controls. It is not a substitute for biological validation.

### ML Protocol / Model Selector

The ML protocol receives the selected panel matrix when the feature-panel check is enabled; otherwise it receives the central filtered matrix. It does not train directly on the raw high-dimensional matrix unless central filtering has been explicitly disabled.

The selector can recommend candidate algorithms. The decision-tree branch is triggered only under the configured conditional logic.

### Decision-Tree Interpretability Branch

The decision-tree branch assumes central filtering has already happened. It then performs:

- tree fitting on the filtered feature matrix
- extraction of root and branch features
- rule generation
- path-based interaction mining
- post-tree confidence and stability scoring

The decision tree is retained because it gives a compact, rule-based interpretation layer.

### Bootstrap and Confidence Scores

Bootstrap support and confidence scores are post-tree interpretability outputs. They answer:

> How stable is this selected feature or decision-path signal under resampling?

They do not answer:

> Which features should enter the model in the first place?

That is the role of pre-tree central feature filtering.

### Interaction Mining

Interaction mining is path-based. Candidate interactions are extracted from co-occurring genomic features along decision-tree paths, then optionally validated downstream. This avoids exhaustive all-pairs testing across the original high-dimensional feature matrix.

### Query-Time Interpretation

Query mode aligns new samples to the feature lists stored in the trained registry or bundle. Missing trained features are filled conservatively, extra query features are ignored, and central feature filtering is not rerun.

For FASTA queries, the selected-feature manifest is the bridge between training and inference. It carries feature identity, reference/alternate allele, baseline allele, annotation, and context sequence forward so query-time nucleotide extraction remains traceable.

Resolved marker evidence may include:

```text
baseline_match
alt_match
known_nonbaseline_match
```

Non-evidence or conservative zero-fill states may include:

```text
not_called
ambiguous_base
multi_hit_context
unresolved_context
non_training_allele
```

A query prediction should therefore be interpreted alongside marker recovery and active feature evidence metrics.

---

## Troubleshooting

### No overlapping sample IDs

Check that sample identifiers in the genomic matrix and metadata file refer to the same biological samples. NetworkParser normalizes common VCF suffixes and conservative library suffixes, but metadata and matrix IDs must still be compatible.

### No features retained after central filtering

This can happen when the cohort is small, labels are weakly separated, metadata are noisy, classes are imbalanced, or empirical p-value resolution is too coarse.

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

FASTA query mode requires selected feature manifests. If the registry does not contain selected manifest paths, rerun training with reference context available and ensure DataLoader writes the feature manifest artifact.

### FASTA query has low marker recovery

Low marker recovery means many selected training markers could not be confidently resolved in the query sequence.

Check:

- whether the query is a consensus FASTA, pseudogenome FASTA, or assembled contig FASTA
- whether the query sequence uses compatible contig/reference context
- whether selected marker contexts are present in the query sequence
- whether many markers produced multi-hit, no-hit, ambiguous, or non-training-allele calls
- whether exact matching is too strict for the expected sequence divergence

### Query output contains many missing trained features

This indicates that the query input was not represented in the same feature space as training. Check that the same reference, VCF parsing logic, feature-ID convention, and DataLoader settings were used. For FASTA query mode, check the selected feature manifest and marker recovery summary.

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

A group-specific Level 2 model may be skipped when the Level 1 subgroup does not support robust Level 2 training. The default eligibility rule is adaptive rather than based on one fixed cohort-size cutoff. Common reasons include only one Level 2 class inside the group, too few samples in the smallest Level 2 class for stratified cross-validation, or no finite model-selector probe scores after filtering.

When this happens, query mode can use a configured global Level 2 fallback if one exists.

### Level 2 has rare classes that make cross-validation impossible

When a Level 2 phenotype or resistance-profile class has too little support, stratified cross-validation cannot produce valid folds. For publication-safe training, keep the run strict and report the limitation, or explicitly enable the Level 2 class-support gate:

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

The audit files record what was excluded and why.

### Use a global binary resistant/susceptible fallback

For sparse detailed Level 2 resistance-profile labels, NetworkParser can train an additional global binary fallback model:

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

Query mode reports the fallback source when this broader endpoint is used.

### Use a broader standard global Level 2 fallback label

When group-specific models should learn a detailed Level 2 label but the standard global fallback should learn a broader endpoint, use `--global_level2_label`:

```bash
python -m network_parser.cli train-two-level \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --level1_label lineage_column \
  --level2_label detailed_resistance_profile_column \
  --global_level2_label broader_binary_or_profile_column \
  --output_dir path/to/two_level_results
```

In this mode:

```text
group-specific Level 2 models → trained on the detailed Level 2 label
standard global Level 2 fallback → trained on the broader fallback label
```

The registry records both label targets so query reports can distinguish detailed group-specific predictions from broader fallback predictions.

### Bundle query fails after moving result directories

Use `--bundle path/to/networkparser_model_bundle.npb` instead of `--registry` when moving trained artifacts between machines. The bundle is designed to carry the trained knowledge object, selected-feature metadata, and model payloads together.

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

> NetworkParser applies a modular supervised analysis workflow in which genomic variant matrices are preprocessed, aligned to metadata labels, and reduced through central statistical feature filtering before model screening. The filtering stage can use Random Forest feature importance with permutation-derived empirical p-values and FDR correction, or configurable chi-square/Fisher alternatives for faster association-based screening. A ranked feature-panel separability check can then evaluate compact top-N marker panels and forward a selected model-ready matrix to the ML protocol and model selector, after which a decision-tree interpretability branch can be conditionally triggered to extract rule-based markers, path-level feature interactions, and post-tree confidence evidence. In two-level or recursive hierarchy workflows, supervised models are trained along ordered biological labels so that query samples can be routed from broad strain/group placement toward phenotype or AMR-profile interpretation. Query mode applies the trained registry or binary bundle to new samples by projecting them onto the saved selected-feature space; for FASTA input, selected marker context sequences are mapped back to the query DNA, marker-centre nucleotides are extracted, and a one-sample selected-feature matrix is reconstructed before prediction.

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
