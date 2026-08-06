# NetworkParser validation scripts

Copy-pasteable shell scripts for NetworkParser hierarchy experiments.

## What each experiment script does

Scripts `01`–`08` are **self-contained** after sourcing `00_config.sh`. Train, query, evaluate, CV, and panel annotation remain **inline** so the full flow is easy to follow. Before training, each script invokes the shared input preparation and preflight checks.

Typical order:

1. model training (`train-hierarchy`)
2. held-out test querying
3. held-out evaluation (`evaluate-hierarchy` pack for primary recipes)
4. leakage-aware repeated cross-validation (if `RUN_LEAKAGE_AWARE_CV=1`)
5. panel annotation / `annotate-panels` with stable report + catalogue circularity (if `RUN_PANEL_ANNOTATION=1`)

Leakage-aware CV uses `TRAIN_GENOMIC` only. Held-out `TEST_GENOMIC` is reserved for query/evaluation.

Panel annotation does **not** re-run feature selection. It summarises selected-feature manifests already written during training (genes, consequences, optional catalogue), and can filter to CV-stable markers.

Important: the current NetworkParser `cross-validate` CLI validates **one supervised metadata label at a time**. For hierarchy experiments, the scripts therefore run CV separately for each relevant target column instead of passing `--hierarchy_labels` into CV.

## Files

| File | Role |
|------|------|
| `00_config.sh` | Settings only (paths, flags, CV, catalogue). Edit like a config file. |
| `afro_vcf_config.json` | Explicit callability policy shared by train, query, and CV for the AFRO bcftools VCFs. |
| `00_prepare_inputs.sh` | Builds clean, disjoint train/test symlink views; derives **`Lineage_family`**; writes training + held-out evaluation metadata. |
| `00_preflight.sh` | Validates paths, metadata labels/IDs, split isolation, VCF contig, GenBank build, catalogue schema, and catalogue identity. |
| `01`–`02` | **Primary biological hierarchies** (no artificial supergroup) |
| `03`–`08` | Legacy / comparison experiments (includes old supergroup runs) |
| `09_annotate_existing_run.sh` | Runs WHO catalogue annotation on an already-trained hierarchy run. |
| **`10_phenotype_AMR_profile_known_marker_seed.sh`** | **Light A/B:** `AMR_binary → Resistance_Profile` only (+ optional known-marker seed). No lineage. CV/annotate off by default. |
| `run_all_networkparser_validation.sh` | Runs selected experiments sequentially (default: `01` `02`) |

**Config keys:** `META_SOURCE`, `META`, `EVALUATION_META`, `TRAIN_GENOMIC`, `TEST_GENOMIC`, `REF`, `BASE_OUT`, `N_JOBS`, `FILTER`, `RUN_LEAKAGE_AWARE_CV`, `CV_*`, `RUN_PANEL_ANNOTATION`, `CATALOGUE`, `STABILITY_TSV`, `MIN_STABILITY`, `HIERARCHY_RESUME`, `GLOBAL_FALLBACK_LABELS`.

### Global fallbacks (opt-in)

By default **`GLOBAL_FALLBACK_LABELS=none`**: only path-local nodes train (plus optional parent-conditioned terminal fallbacks). Cohort-wide globals are not trained unless you name levels:

```bash
# No globals (default)
GLOBAL_FALLBACK_LABELS=none bash 01_Lineage_AMR_Resistance_Profile.sh

# Only a global lineage model
GLOBAL_FALLBACK_LABELS=Lineage_clean bash 01_Lineage_AMR_Resistance_Profile.sh

# Terminal phenotype global only
GLOBAL_FALLBACK_LABELS=terminal bash 01_Lineage_AMR_Resistance_Profile.sh

# Old behaviour (terminal + lineage globals)
GLOBAL_FALLBACK_LABELS=legacy bash 01_Lineage_AMR_Resistance_Profile.sh
```

CLI equivalent: `--global_fallback_labels none|terminal|lineage|legacy|ColA,ColB`  
Disable parent-conditioned terminal fallbacks: `--no_parent_conditioned_fallbacks`

### Biological hierarchy recipes (recommended)

| Script | Hierarchy | Notes |
|--------|-----------|--------|
| **`01_Lineage_AMR_Resistance_Profile.sh`** | `Lineage_clean → AMR_binary → Resistance_Profile_Collapsed` | Primary 3-level biological recipe |
| **`02_Lineage_family_Lineage_AMR_profile.sh`** | `Lineage_family → Lineage_clean → AMR_binary → Resistance_Profile_Collapsed` | 4-level with coarse lineage families |

`Lineage_family` groups:

| Family | Lineages |
|--------|----------|
| Indo_Oceanic | L1 |
| East_Asian | L2 |
| East_African_Indian | L3 |
| Euro_American | L4 |
| M_africanum_animal | L5, L6, L7, BOV_AFRI |

### Legacy / comparison scripts

- `03` Classic two-level + global AMR fallback  
- `04` Supergroup → Lineage → AMR (test scaffolding; not for biological claims)  
- `05` Supergroup → Lineage → resistance profile  
- `06` Two-level Lineage → AMR  
- `07`–`08` Older multi-level naming variants  

### AFRO VCF callability policy

```json
{
  "min_gq_per_sample": 0,
  "assume_absent_variant_is_reference": true
}
```

## Run one experiment

```bash
cd scripts/testing_scripts
# edit 00_config.sh if needed
bash 01_Lineage_AMR_Resistance_Profile.sh
```

Skip CV for a faster smoke run:

```bash
RUN_LEAKAGE_AWARE_CV=0 bash 01_Lineage_AMR_Resistance_Profile.sh
```

Resume interrupted hierarchy training:

```bash
HIERARCHY_RESUME=1 bash 01_Lineage_AMR_Resistance_Profile.sh
```

## Run selected experiments

```bash
bash run_all_networkparser_validation.sh          # default 01 02
bash run_all_networkparser_validation.sh 01 02 06
```

## Training artifacts (new)

Each successful `train-hierarchy` run now also writes:

- `resource_profile_train.json` — CPU/RAM/parallel budget  
- `hierarchy_node_dashboard.tsv` — one row per node (status, algo, n, features)  
- `figures/hierarchy_dendrogram.png` — path-local model tree  
- optional parquet/gzip matrix when `memory_efficient=True`  

Annotation can write:

- `stable_panel_features_annotated.tsv`  
- `catalogue_circularity_by_node.tsv` / `catalogue_circularity_summary.json`  

Evaluation pack (`evaluate-hierarchy`):

- per-level metrics + confusions + bootstrap CIs  
- `full_path_predictions.tsv`  
- `evaluation_summary.json`  
