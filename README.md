# NetworkParser

**NetworkParser** is an interpretable genomic feature-discovery, model-selection, and query-inference framework for microbial variant data. It converts genomic matrices or VCF-derived variant spaces into clean sample × feature matrices, applies statistically defensible feature filtering, evaluates machine-learning suitability, conditionally triggers decision-tree interpretability, and exports query-ready model registries for strain placement and antimicrobial-resistance interpretation.

NetworkParser is designed for microbial genomics settings where prediction alone is not enough. The framework links supervised genomic classification to traceable marker evidence, AMR phenotype interpretation, confidence-aware rule extraction, and ML-ready / GNN-ready matrix outputs.

---

## Core Aim

NetworkParser supports four related analysis modes:

1. **Single-label supervised discovery**  
   Use one metadata label, such as a lineage, strain group, phenotype, AMR class, or outbreak cluster, to identify discriminating genomic features and optionally run interpretable decision-tree discovery.

2. **Two-level diagnostic interpretation**  
   First place a strain or sample into a supervised genomic group, then evaluate resistance-associated patterns using a second supervised phenotype or AMR-profile label.

3. **True multi-level hierarchy training with branch-aware fallback**  
   Train an ordered recursive hierarchy from multiple metadata labels, for example a broad genomic grouping, then a finer lineage/clade label, then an AMR phenotype or resistance endpoint. Each child model is trained only inside the parent branch that defines it. For AMR workflows, NetworkParser can attempt exact resistance-profile prediction inside a genomic branch and fall back to a broader binary AMR endpoint when profile-level inference is unsupported, unavailable, or not statistically defensible.

4. **Query-time inference from a new sample**  
   Project a new sample onto the trained selected-feature space and use the saved model registry to predict strain/group placement and AMR phenotype or resistance profile. Query mode can use either a prebuilt genomic feature row/matrix, VCF-derived query input, FASTQ-derived VCF input, or a FASTA-like DNA sequence when a query-ready feature manifest was saved during training.

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
Matrix alignment OR selected-marker reconstruction from VCF / FASTA / FASTQ-derived VCF
    ↓
One-sample selected-feature matrix
    ↓
Saved registry model(s)
    ↓
Two-level prediction OR recursive hierarchy traversal
    ↓
Branch-aware endpoint decision
exact profile prediction where validated; broader endpoint fallback where needed
    ↓
Prediction report + marker evidence report + terminal/fallback status
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
- Supports standard two-level training and true recursive multi-level hierarchy training.
- Supports branch-aware AMR endpoint fallback for recursive hierarchies, allowing exact profile prediction only where branch-level validation supports it and broader binary AMR prediction where profile inference collapses or is unavailable.
- Stores selected-feature manifests per trained hierarchy node where context metadata is available.
- Conditionally triggers the decision-tree interpretability branch.
- Extracts interpretable decision paths, branch-level rules, and path-based feature interactions.
- Computes post-tree confidence and bootstrap stability evidence.
- Produces filtered matrices, ranked marker tables, model artifacts, interaction outputs, selected marker manifests, query reports, and validation summaries.
- Supports FASTA-like query mode by mapping saved marker-context sequences back to a user-supplied DNA sequence and reconstructing the trained selected-feature matrix.
- Evaluates saved prediction tables against metadata using per-class true-positive rate / sensitivity, specificity, false-positive rate, precision, F1, confusion matrices, hierarchy full-path correctness, and branch-level fallback recommendations.
- Applies query-time safeguards: low-support review flags, global and terminal fallback escalation, interpretation-confidence categories, and optional terminal AMR evidence guarding when branch marker resolution is weak.

---

## Demonstration Benchmark (held-out test cohort)

On a three-level *M. tuberculosis* hierarchy (`Lineage_Supergroup` → `Lineage_clean` → `AMR_binary`) with chi-squared FDR feature filtering, held-out query evaluation on **3,589** labelled isolates achieved:

| Level | Accuracy | Balanced accuracy |
|---|---|---|
| Fine-grained lineage (`Lineage_clean`) | 99.5% | 85.7% |
| Terminal AMR binary endpoint | 95.9% | 94.9% |

Resistant recall was **90.3%**; susceptible recall **99.5%**. Most terminal errors occurred among lineage-correct samples with insufficient branch-specific marker resolution rather than upstream routing failure. Fallback-completed query paths (**5.2%** of samples) retained **98.9%** terminal accuracy. See `scripts/testing_scripts/02_hierarchy_with_AMR_binary.sh` for the reference workflow.

---

## Command-Line Entry Points

NetworkParser exposes seven main CLI workflows:

```text
python -m network_parser.cli run              # single-label workflow
python -m network_parser.cli train-two-level  # two-level or true multi-level hierarchy training
python -m network_parser.cli bundle           # package trained registry into portable .npb bundle
python -m network_parser.cli query            # query/inference workflow
python -m network_parser.cli evaluate         # evaluate saved predictions against metadata
python -m network_parser.cli cross-validate    # leakage-aware repeated cross-validation
python -m network_parser.cli validate-cv      # alias for cross-validate (backward compatible)
```

The `train-two-level` command keeps the original two-level interface through `--level1_label` and `--level2_label`, and also exposes true recursive hierarchy training through `--hierarchy_labels`. The `evaluate` command can use the same ordered hierarchy labels to evaluate per-level metrics, full-path correctness, and branch-level fallback behaviour from saved query predictions. Training runs with `build_model_bundle=True` (default) also write `networkparser_model_bundle.npb` automatically; use `bundle` to rebuild or export a bundle from an existing registry.

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

For FASTA query mode, NetworkParser can use external BLAST command-line tools when they are already available on `PATH`. If they are not available, `auto` mode falls back to exact context matching.

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

A reference FASTA or GenBank-like reference context is optional for matrix-only training, but strongly recommended when query mode must accept FASTA DNA sequence.

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

### 5. Multi-Level Hierarchy Metadata

For true recursive hierarchy training, the metadata must contain every label column listed in `--hierarchy_labels`. The order matters because it defines the parent-to-child route.

Example concept:

```text
sample_id    broad_group    fine_group    terminal_endpoint    optional_metadata
sample_A     parent_A       child_A       phenotype_A          ...
sample_B     parent_A       child_B       phenotype_B          ...
sample_C     parent_B       child_C       phenotype_A          ...
```

Conceptually:

```text
broad genomic grouping
  └── finer strain / lineage / clade placement
        └── AMR phenotype, resistance class, or resistance-profile endpoint
```

For AMR interpretation, a useful hierarchy can separate the genomic placement step from the endpoint decision:

```text
Lineage / genomic branch
  └── Resistance_Profile when branch-level validation is acceptable
  └── AMR_binary fallback when exact profile inference collapses, is unavailable, or has insufficient support
```

Each deeper model is trained only on samples that reached the parent branch. This is useful for interpretable hierarchical reporting, but it can fragment small or imbalanced cohorts. A metadata label with weak genomic signal, such as a sampling location or administrative source, should usually be treated as auxiliary context rather than the primary parent of AMR phenotype prediction.

### 6. Query Input

Query mode supports four concepts:

1. **Prebuilt genomic feature row or matrix**  
   The query sample is already represented using genomic feature names compatible with training.

2. **VCF / VCF.GZ input**  
   NetworkParser constructs a query-time sample × genomic-feature matrix, relaxes cohort-level query filters, and aligns observed query markers to the selected feature space stored in the registry.

3. **FASTA-like DNA sequence**  
   NetworkParser uses the selected marker manifest from training, maps saved context sequences against the user sequence, extracts the marker-centre nucleotide, encodes the result, and creates the one-sample selected-feature matrix required by the trained models.

4. **Paired FASTQ directory**  
   FASTQ mode is a preprocessing bridge: reads are aligned to the supplied reference, variants are called, and the resulting VCFs are passed through the normal query-time feature-alignment route.

FASTA query mode is intended for consensus FASTA, pseudogenome FASTA, or assembled contig FASTA. FASTQ mode requires external alignment and variant-calling tools on `PATH` and remains inference-only after VCF construction.

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

---

## Quick Start: True Multi-Level Hierarchy Training

Train a recursive hierarchy from an ordered list of metadata labels:

```bash
python -m network_parser.cli train-two-level \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --hierarchy_labels broad_group_label fine_group_label terminal_endpoint_label \
  --central_feature_filter_method chi2_fdr \
  --ref_fasta path/to/reference.fasta \
  --output_dir path/to/hierarchy_results \
  --feature_panel_check on \
  --feature_panel_sizes 100,200,500,1000 \
  --n_jobs 4
```

For AMR branch-aware inference, the ordered labels can define a route such as:

```bash
python -m network_parser.cli train-two-level \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --hierarchy_labels Lineage Resistance_Profile AMR_binary \
  --central_feature_filter_method chi2_fdr \
  --ref_fasta path/to/reference.fasta \
  --output_dir path/to/hierarchy_results \
  --feature_panel_check on \
  --feature_panel_sizes 100,200,500,1000 \
  --n_jobs 4
```

The ordered labels define the route:

```text
broad_group_label
  └── fine_group_label
        └── terminal_endpoint_label

AMR example:
Lineage
  └── Resistance_Profile
        └── AMR_binary
```

For each trainable node, NetworkParser runs the same statistically defensible sequence used elsewhere in the pipeline:

```text
parent-specific sample subset
    ↓
configured central feature filtering
    ↓
ranked feature-panel separability check
    ↓
node-specific model training
    ↓
selected-feature manifest for that node
```

Nodes with only one child label are stored as deterministic branches. Nodes with insufficient per-label support are skipped and written with explicit diagnostics rather than being forced into unstable stratified model probes. This is especially important for small cohorts and high-dimensional binary variant matrices.

For three-level routes such as `Lineage → Resistance_Profile → AMR_binary`, NetworkParser can also save broader terminal fallback models for the final endpoint. These fallback models are used when the exact profile branch is unavailable, skipped, deterministic, or not suitable for robust inference. This preserves recursive hierarchy reporting while avoiding overconfident profile-level predictions inside weak or unstable branches. Query reports record whether the endpoint prediction came from the exact hierarchy node, a parent-level terminal fallback, or a global terminal fallback.

### Branch-aware AMR endpoint rule

The recommended interpretation rule is:

```text
1. Place the sample into the genomic branch.
2. Attempt exact Resistance_Profile prediction only if that branch is trainable and validated.
3. If exact profile inference collapses, is unavailable, or is insufficiently supported, report the broader AMR_binary fallback.
4. If the parent branch itself has insufficient support, report the branch as exploratory rather than forcing a confident endpoint.
```

This makes the hierarchy conservative by design. NetworkParser should not overclaim exact resistance-profile resolution in branches where the evidence supports only a broader endpoint.

The hierarchy registry is written as:

```text
hierarchical_model_registry.json
```

For AMR phenotype prediction, prefer hierarchy orders where the parent levels have plausible genomic signal. For example, a broad lineage/clade grouping followed by a finer lineage label and then an AMR endpoint is usually more statistically defensible than using a sampling-location label as the first parent.

---

## Test-Only Hierarchy Label Helper

When testing true recursive hierarchy mechanics, it can be useful to create a synthetic parent label from an existing genomic label. This should be used for software validation only, not as biological evidence.

Example using the reusable helper script:

```bash
python add_test_hierarchy_column.py \
  --input path/to/metadata.csv \
  --output path/to/metadata_with_test_hierarchy.csv \
  --source-col fine_group_label \
  --output-col Test_Supergroup_Label \
  --strategy frequency_bins
```

Then train:

```bash
python -m network_parser.cli train-two-level \
  --genomic path/to/genomic_input \
  --meta path/to/metadata_with_test_hierarchy.csv \
  --hierarchy_labels Test_Supergroup_Label fine_group_label terminal_endpoint_label \
  --central_feature_filter_method chi2_fdr \
  --ref_fasta path/to/reference.fasta \
  --output_dir path/to/test_hierarchy_results \
  --n_jobs 4
```

Supported helper strategies:

| Strategy | Purpose |
|---|---|
| `frequency_bins` | Groups source-label states into coarse frequency-based parents for testing hierarchy mechanics. |
| `prefix` | Builds parent labels from the first part(s) of a structured source label. |
| `manual_map` | Uses a user-supplied mapping file from source label to parent label. |

The helper also writes a small summary JSON so the generated parent-label distribution can be audited before training.

---

## Quick Start: Query / Inference

Apply a trained two-level or hierarchy registry to new genomic input:

```bash
python -m network_parser.cli query \
  --genomic path/to/new_genomic_input \
  --registry path/to/two_level_model_registry.json \
  --output_dir path/to/query_results \
  --max_markers 10 \
  --n_jobs 4
```

For a true multi-level hierarchy run, point `--registry` to:

```text
hierarchical_model_registry.json
```

Query mode is inference-only. It does **not** rerun central feature filtering, permutation testing, FDR correction, decision-tree training, or bootstrap confidence estimation. Instead, new samples are aligned to the trained feature space stored in the registry.

### FASTA query mode

FASTA sequence can be queried directly when the training run was created with a selected feature manifest containing reference context. In this mode, NetworkParser uses the saved context sequence for each selected genomic feature, maps the context back to the query DNA, extracts the centre nucleotide, and rebuilds the selected-feature matrix before prediction:

```bash
python -m network_parser.cli query \
  --genomic path/to/new_sample.fasta \
  --registry path/to/trained_registry.json \
  --output_dir path/to/query_results \
  --query_input_type fasta \
  --fasta_mapping_mode auto
```

`--fasta_mapping_mode auto` uses BLAST context mapping when `makeblastdb` and `blastn` are available on `PATH`, otherwise it falls back to exact flanking-context matching. Use `blast` to require BLAST, or `exact` to skip BLAST. The older `raw_sequence` input type and `--raw_sequence_mapping_mode` flag are retained as compatibility aliases.

FASTA query mode performs:

```text
FASTA DNA sequence
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
apply saved registry model(s)
```

Query outputs include:

```text
query_predictions.csv
query_report.json
query_report.txt
query_alignment_summary.json
fasta_query_encoding/raw_sequence_selected_feature_matrix.csv
fasta_query_encoding/raw_sequence_feature_calls.tsv
fasta_query_encoding/raw_sequence_mapping_summary.json
vcf_query_encoding/vcf_selected_feature_matrix.csv
vcf_query_encoding/vcf_feature_calls.tsv
vcf_query_encoding/vcf_mapping_summary.json
```

The report contains the predicted Level 1 identity and Level 2 phenotype/profile for two-level registries, or the ordered hierarchy route for recursive hierarchy registries. For branch-aware hierarchies, it also records terminal status and terminal reason fields so users can distinguish an exact hierarchy-node prediction from a deterministic branch, parent-level fallback, global fallback, or unavailable endpoint. The report also includes support values where available, supporting markers, observed nucleotide evidence where available, marker recovery metrics, and decision-path explanations when the saved model exposes tree-like structure.

Compact query outputs (`query_predictions_compact.tsv`) include interpretation-confidence categories, low-support review fields, and terminal evidence-guard status where enabled.

### Query-time safeguards

NetworkParser is **inference-only** at query time: it does not rerun FDR filtering, model training, or bootstrap confidence estimation. Instead, it aligns new samples to the trained feature space and applies the safeguards below when configured.

| Safeguard | Purpose | Typical CLI / config |
|---|---|---|
| **Low-support review** | Flags rare predicted classes for manual review instead of forcing a hard label when training support was insufficient | `--low_support_review_enabled`, `--low_support_review_min_training_count` |
| **Global lineage fallback** | Recovers intermediate placement when branch models are unavailable, low-confidence, or disagree with a global lineage model | `hierarchy_global_lineage_fallback_on_low_confidence`, `hierarchy_global_lineage_fallback_on_disagreement` |
| **Terminal fallback models** | Parent-conditioned and global terminal models complete the deepest endpoint when exact child nodes are missing | Trained automatically under `hierarchy_models/terminal_fallbacks/` |
| **AMR evidence guard** | Escalates weak majority-class terminal calls (e.g. susceptible) when branch marker panels resolve too few features | `--amr_weak_evidence_review_enabled`, `--amr_weak_evidence_min_resolved_fraction`, `--hierarchy_global_amr_fallback_on_weak_evidence` |
| **Interpretation confidence** | Combines model support with resolved-marker evidence → `high_confidence`, `moderate_confidence`, `low_confidence`, or review-required endpoints | Reported per hierarchy level in query outputs |

Example hierarchy query with AMR evidence guard:

```bash
python -m network_parser.cli query \
  --genomic path/to/test_vcfs \
  --bundle path/to/networkparser_model_bundle.npb \
  --output_dir path/to/query_results \
  --n_jobs 8 \
  --amr_weak_evidence_review_enabled \
  --amr_weak_evidence_min_resolved_fraction 0.15 \
  --hierarchy_global_amr_fallback_on_weak_evidence
```

When the evidence guard triggers, query reports preserve the branch model's candidate prediction but may route to global terminal fallback or emit an explicit review-required endpoint rather than a silent false negative on minority classes.

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
  --registry path/to/trained_registry.json \
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
apply saved registry model(s)
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

## Quick Start: Prediction Evaluation and Branch Diagnostics

After training and query inference, NetworkParser predictions should be evaluated against labelled metadata. This stage is evaluation-only: it does not rerun statistical filtering, feature selection, model training, tree construction, or bootstrap confidence scoring.

Evaluate a saved query prediction table for a recursive hierarchy:

```bash
python -m network_parser.cli evaluate \
  --predictions path/to/query_predictions.csv \
  --meta path/to/metadata.csv \
  --hierarchy_labels Lineage Resistance_Profile AMR_binary \
  --output_dir path/to/validation_results
```

The hierarchy evaluator writes per-level metrics, full-path correctness, and branch-conditioned diagnostics. These diagnostics are designed to answer whether exact profile-level inference is supported inside each genomic branch or whether a broader endpoint fallback is more statistically defensible.

Main validation outputs include:

```text
validation_results/
├── hierarchy_level_01__<label>/
│   ├── model_performance_by_class.tsv
│   ├── confusion_matrix.tsv
│   └── model_performance_summary.json
├── hierarchy_level_02__<label>/
│   └── per-level metric outputs
├── hierarchy_level_03__<label>/
│   └── per-level metric outputs
├── hierarchy_full_path/
│   ├── hierarchy_full_path_predictions.tsv
│   ├── hierarchy_prefix_depth_counts.tsv
│   └── hierarchy_full_path_summary.json
├── hierarchy_branch_diagnostics/
│   ├── per_parent_child_metrics.tsv
│   ├── per_parent_fallback_recommendations.tsv
│   ├── per_sample_branch_predictions.tsv
│   ├── recommendation_counts.tsv
│   └── branch_diagnostics_summary.json
└── networkparser_validation_summary.json
```

Per-class metrics include:

```text
TP / FP / TN / FN
true_positive_rate / sensitivity / recall
false_positive_rate
true_negative_rate / specificity
false_negative_rate
PPV / precision
NPV
F1
```

For AMR hierarchy validation, the branch-diagnostics table is especially important. It separates ordinary misclassification from unsupported or unavailable downstream inference. A branch can therefore be reported as:

```text
use_exact_child_prediction
fallback_to_binary_endpoint
insufficient_support
routing_or_prediction_coverage_issue
```

### Leakage-aware repeated cross-validation

For robust inference during model validation, use repeated cross-validation where the fold split happens before supervised statistical filtering:

```bash
python -m network_parser.cli validate-cv \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --label target_label_column \
  --central_feature_filter_method chi2_fdr \
  --output_dir path/to/cv_validation \
  --n_repeats 3 \
  --n_splits 5 \
  --n_jobs 4
```

For per-level hierarchy validation, the same wrapper can be run against each hierarchy label, or with `--hierarchy_labels` when enabled:

```bash
python -m network_parser.cli validate-cv \
  --genomic path/to/genomic_input \
  --meta path/to/metadata.csv \
  --hierarchy_labels Lineage Resistance_Profile AMR_binary \
  --central_feature_filter_method chi2_fdr \
  --output_dir path/to/cv_validation \
  --n_repeats 3 \
  --n_splits 5 \
  --n_jobs 4
```

The leakage-aware validation rule is:

```text
split first
  ↓
fit feature filtering / panel selection on the training fold only
  ↓
train the model on the training fold only
  ↓
predict the held-out fold
  ↓
calculate TP / FP / TN / FN and aggregate metrics
```

This avoids reporting optimistic performance from a feature set selected using the full labelled cohort.

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
Input → preprocessing → feature manifest → configurable central feature filtering → ranked feature-panel separability check → ML protocol/model selector → conditional decision-tree interpretation → post-tree confidence and interaction outputs → query-ready registry
```

---

## Main CLI Options

### Shared options

| Argument | Description |
|---|---|
| `--genomic` | Genomic input file, VCF directory, query matrix, FASTA, or paired FASTQ directory depending on workflow. |
| `--output_dir` | Output directory. |
| `--config` | Optional JSON file with `NetworkParserConfig` overrides. |
| `--ref_fasta` | Optional FASTA or GenBank reference context for VCF-oriented workflows and FASTA query support. |
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
| `--min_level2_samples_per_group` | Optional absolute minimum for group-specific Level 2 models. When unset, eligibility is adaptive and scales with the number of Level 2 labels represented in the group. |
| `--hierarchy_labels` | Ordered metadata columns for true recursive hierarchy training. When supplied, this replaces the standard two-level `--level1_label` / `--level2_label` route. |

### Query options

| Argument | Description |
|---|---|
| `--registry` | Path to `two_level_model_registry.json` or `hierarchical_model_registry.json` from training. |
| `--max_markers` | Maximum number of supporting markers shown per level per sample. |
| `--query_input_type` | Use `fasta` when `--genomic` is FASTA DNA; use `vcf`, `fastq`, `matrix`, or `auto` for other query routes. |
| `--fasta_mapping_mode` | Use `auto`, `blast`, or `exact` for context-based FASTA feature reconstruction. The older `--raw_sequence_mapping_mode` remains an alias. |

### Evaluation options

| Argument | Description |
|---|---|
| `--predictions` | Saved NetworkParser prediction table, usually `query_predictions.csv` or `query_predictions_compact.tsv`. |
| `--meta` | Metadata CSV/TSV containing the ground-truth labels. |
| `--level1_label` / `--level2_label` | Two-level truth columns for evaluating standard two-level predictions. |
| `--hierarchy_labels` | Ordered truth columns for evaluating recursive hierarchy predictions, full-path correctness, and branch diagnostics. |
| `--sample_id_column` | Optional metadata sample-id column override. |
| `--output_dir` | Directory for evaluation summaries, confusion matrices, and branch diagnostics. |
| `--n_repeats` / `--n_splits` | Repeated cross-validation controls for `cross-validate` / `validate-cv`. |

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
| FASTA query mode | Controls selected-marker context mapping and allele extraction from user-supplied DNA sequence. |

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

### True hierarchy training workflow

```text
hierarchy_results/
├── matrices/
│   └── DataLoader matrix and feature-manifest outputs
├── hierarchy_models/
│   ├── root/
│   │   ├── node_summary.json
│   │   ├── selected_feature_manifest.tsv
│   │   ├── feature_panel_separability/
│   │   ├── model/
│   │   └── children/
│   │       └── recursive child-node outputs
│   └── terminal_fallbacks/
│       └── broader endpoint fallback models where trainable
├── aligned_hierarchy_matrix.csv
├── aligned_hierarchy_labels.csv
└── hierarchical_model_registry.json
```

### Query workflow

```text
query_results/
├── query_matrix_artifacts/
│   └── DataLoader artifacts for matrix or VCF query input
├── fasta_query_encoding/
│   ├── raw_sequence_selected_feature_matrix.csv
│   ├── raw_sequence_feature_calls.tsv
│   └── raw_sequence_mapping_summary.json
├── vcf_query_encoding/
│   ├── vcf_selected_feature_matrix.csv
│   ├── vcf_feature_calls.tsv
│   └── vcf_mapping_summary.json
├── query_predictions.csv
├── query_predictions_compact.tsv
├── query_predictions_readable.html
├── query_alignment_summary.json
├── query_route_audit.json
├── query_report.json
└── query_report.txt

Important hierarchy columns include:

```text
predicted_level1, predicted_level2, predicted_level3, ...
hierarchy_terminal_status
hierarchy_terminal_reason
```

These fields allow downstream validation to distinguish exact node prediction from terminal fallback behaviour.
```

### Validation workflow

```text
validation_results/
├── hierarchy_level_01__<label>/
├── hierarchy_level_02__<label>/
├── hierarchy_level_03__<label>/
├── hierarchy_full_path/
├── hierarchy_branch_diagnostics/
└── networkparser_validation_summary.json

cv_validation/
├── cross_validation_summary.json
├── cv_fold_metrics.tsv
├── cv_predictions.tsv
├── cv_feature_stability.tsv
├── cv_by_class_metrics.tsv
└── aggregate_performance/
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

### Branch-aware AMR fallback

Branch-aware fallback is an inference and reporting rule, not a replacement for statistical filtering. The profile-level model is still trained only when its branch passes the configured training and support checks. The fallback endpoint is used to avoid overclaiming exact resistance-profile resolution when the branch-level evidence supports only a broader AMR call.

For AMR interpretation, this gives a conservative route:

```text
Genomic placement with high confidence
    ↓
Exact profile prediction when validated
    ↓
Binary AMR fallback when profile prediction is unsupported
```

The fallback decision should be justified using branch diagnostics, not by manually inspecting a single global metric.

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

For FASTA queries, the selected-feature manifest is the bridge between training and inference. It carries feature identity, reference/alternate allele, baseline allele, annotation, and context sequence forward so that query-time nucleotide extraction remains traceable.

### FASTA Query Logic

FASTA query mode is not new feature discovery. It asks whether the new sequence contains the previously selected markers.

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

FASTA query mode requires selected feature manifests. If the registry does not contain selected manifest paths, rerun training with reference context available and ensure DataLoader writes the feature manifest artifact.

### FASTA query has low marker recovery

Low marker recovery means many selected training markers could not be confidently resolved in the query sequence.

Check:

- whether the query sequence is a consensus, pseudogenome, or assembly rather than raw reads
- whether the query sequence uses compatible contig/reference context
- whether the selected marker contexts are present in the query sequence
- whether many markers produced multi-hit, no-hit, low-confidence, or ambiguous calls
- whether exact matching is too strict for the expected sequence divergence

### Query output contains many missing trained features

This indicates that the query input was not represented in the same feature space as training. Check that the same reference, VCF parsing logic, feature-ID convention, and DataLoader settings were used. For FASTA query mode, check the selected feature manifest and marker recovery summary.

### FASTQ query mode fails

FASTQ query mode requires `bwa`, `samtools`, and `bcftools` on `PATH`, plus a compatible reference FASTA supplied through `--ref_fasta`. If these tools are unavailable or the read-processing route fails, convert reads to VCF externally and query with `--query_input_type vcf` instead.

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

A group-specific Level 2 model may be skipped when the Level 1 subgroup does not support robust Level 2 training. The default eligibility rule is adaptive rather than based on one fixed cohort-size cutoff: the required training support scales with the number of Level 2 labels represented inside the group, with at least two samples per label needed for stratified model probes. Common reasons include only one Level 2 class inside the group, too few samples in the smallest Level 2 class for stratified cross-validation, or no finite model-selector probe scores after filtering. In these cases, the group summary JSON records the skip reason, label-balance diagnostics, feasible CV splits, adaptive minimum sample requirement, and whether prediction will use the global Level 2 fallback.

### True hierarchy reaches a skipped terminal node

In a recursive hierarchy, each deeper model is trained inside the parent branch. A terminal phenotype or resistance endpoint may therefore be skipped if the parent branch contains only one terminal class, too few samples per terminal class, or no feasible stratified model probes after filtering. This does not mean query-time marker extraction failed. It means the training subset for that exact path was not statistically defensible.

Check the node summary files under:

```text
hierarchy_models/**/node_summary.json
hierarchy_models/**/hierarchy_label_support_diagnostics.tsv
```

A common fix is to use a biologically stronger parent label order, for example:

```text
broad genomic group → fine lineage/clade → AMR endpoint
```

instead of placing a weak sampling or location label before the AMR endpoint. For robust inference, the main diagnostic route should keep AMR phenotype prediction under a parent with clear genomic signal.

### Sampling-location or source-label prediction is weak

Sampling labels can be epidemiologically useful but may not be clean genomic targets. If a label reflects where samples were collected rather than a stable biological grouping, it can behave poorly as the first hierarchy level and can also fragment downstream phenotype training. Treat such labels as auxiliary metadata or train them as separate exploratory single-label models rather than forcing them to be the parent of AMR phenotype prediction.

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

## Recommended Hierarchy Design Notes

Use true hierarchy mode when the ordered labels represent progressively finer genomic or clinically meaningful decisions. Good parent levels should have enough label support and should be expected to have a detectable genomic signal.

Recommended diagnostic pattern:

```text
coarse genomic group → finer lineage/clade → AMR phenotype or resistance endpoint
```

Use caution with:

```text
sampling location → lineage/clade → AMR phenotype
```

because the first level may encode sampling structure rather than biology, and the downstream phenotype model may become underpowered inside each location-specific branch.

For software testing only, use the test hierarchy helper to create a controlled synthetic parent label from an existing genomic label. Do not present that synthetic parent as a discovered biological grouping.

---

## Citation

If you use NetworkParser in academic work, please cite:

```text
Mfuphi N. Development of NetworkParser: An Integrated Automated System for
Analyzing Evolutionary Processes and Generating AI-based Diagnostic Tools for Microbes.
PhD thesis, University of Pretoria, 2026 (in preparation).
```

Repository:

```text
Mfuphi N. NetworkParser: An interpretable genomic feature-discovery and
hierarchy-aware query-inference framework for microbial variant data.
GitHub: https://github.com/Nomlie/network_parser
```

A peer-reviewed manuscript citation will be added when available.

---

## License

This project is distributed under the license specified in the repository.

---

## Author

**Nomlindelo Mfuphi**  
Bioinformatics Support Scientist  
Centre for High Performance Computing
 