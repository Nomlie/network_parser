# NetworkParser — full picture

Research pipeline: **genotypes + labels → branch-local models → portable query → audited predictions**.  
Not a clinical diagnostic product.

**Figure:** `docs/NetworkParser_full_picture.jpg`  
**Known-marker detail:** `docs/KNOWN_MARKER_SEED.md`  
**Panel FASTQ calling:** `Data/PRJCA040523/scripts/README_PANEL_CALLING.md`

---

## 1. Lifecycle (what you run)

```text
┌─────────────┐   ┌──────────────────┐   ┌─────────┐   ┌──────────────────┐
│  train /    │──▶│  registry + .npb │──▶│  query │──▶│  evaluate /      │
│  train-     │   │  selected panels │   │  new    │   │  CV / annotate   │
│  hierarchy  │   │  route tree      │   │  samples│   │  (optional)      │
└─────────────┘   └──────────────────┘   └─────────┘   └──────────────────┘
       │                                         │
       │         train-only stats                │  frozen feature space
       │         (no query leakage)              │  (no re-filter / retrain)
```

| CLI | Role |
|-----|------|
| `run` | Single-label train |
| `train-hierarchy` | Multi-level, parent-scoped models |
| `bundle` | Package registry → `.npb` |
| `query` | Infer on matrix / VCF / FASTA / FASTQ |
| `evaluate` / `evaluate-hierarchy` | Score vs metadata |
| `cross-validate` | Leakage-aware CV (one label) |
| `annotate-panels` | Gene / catalogue context on panels |

---

## 2. Inputs → matrix contract

```text
  Genomic                         Metadata
  ────────                        ────────
  VCF/gVCF dir  ─┐                Sample + labels
  or matrix CSV ─┼── align IDs ──▶  (Lineage, AMR, Profile, …)
                 │
                 ▼
        Callability / GT semantics
        (GQ, DP, absence-as-REF option)
                 │
                 ▼
        Sample × Feature matrix
        Feature_ID = Contig:Pos:Ref:Alt
```

**Train-only** unsupervised filters and **central supervised filter** (e.g. `chi2_fdr`) run on the training partition. Query never redoes association testing.

### Callability knobs (train + query)

| Config | Typical AFRO / PRJCA use | Meaning |
|--------|--------------------------|---------|
| `min_gq_per_sample` | `0` (variant-only VCFs often lack GQ) | Drop low-GQ genotypes |
| `assume_absent_variant_is_reference` | `true` for variant-only callsets | Site not in VCF → treat as REF |
| `expand_gvcf_ref_blocks` | `true` for gVCF | Use REF blocks for callability |
| `min_feature_recovery_fraction` | e.g. `0.5` | Query gate: fraction of panel recovered |
| `min_callable_fraction` | e.g. `0.5` | Query gate: callable fraction |

---

## 3. Hierarchy idea (the “network”)

Biological recipe used in AFRO work:

```text
                    ┌──────────────────┐
                    │  Lineage_clean   │  Level 1  (placement)
                    │  global model    │
                    └────────┬─────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
      lineage 1         lineage 2           … L_k
           │                 │
           ▼                 ▼
     ┌───────────┐     ┌───────────┐
     │ AMR_binary│     │ AMR_binary│  Level 2  (phenotype)
     │ parent= L │     │ parent= L │
     └─────┬─────┘     └─────┬─────┘
        R  │  S           R  │  S
           ▼                 ▼
     ┌───────────┐     ┌───────────┐
     │ Resistance│     │ Resistance│  Level 3  (profile)
     │ Profile   │     │ Profile   │
     └───────────┘     └───────────┘
```

- Each **node** = model trained only on samples that took that parent route.  
- Full 3-level tree ≈ **~20+ nodes**.  
- **Light experiment**: drop lineage → only `AMR → Profile` ≈ **~3 nodes**.

Query walks the **same path**: L1 → L2 → L3, recording route, confidence, fallbacks, weak evidence.

---

## 4. What happens *inside* each node (train)

```text
  Branch samples + filtered matrix for this node
                    │
                    ▼
         Rank features (p / effect)
                    │
        ┌───────────┴───────────┐
        │  OPTIONAL (config)    │
        │  seed_known_markers   │──▶ WHO catalogue hits first
        │  phenotype stages only│    (AMR / profile; not lineage)
        └───────────┬───────────┘
                    ▼
         Probe panels 100 / 200 / 500 / 1000
         (metric ≥ threshold, default 0.75)
                    │
           pass ────┼──── fail → node unsupported (no silent cheat)
                    ▼
         Model select (LR / RF / …) + impute fit
                    │
                    ▼
         Save: model.pkl · panel · node_summary · registry entry
```

---

## 5. Known-marker seed option (phenotype endpoints)

**Purpose:** test whether **forcing WHO / catalogue resistance alleles** into panels improves AMR and resistance-profile prediction.

**Default: OFF.** Statistical ranking alone until you opt in.

### Where it sits in the graph

```text
  Central filter → retained matrix X
         │
         ▼
  Rank by stats (all stages)
         │
         ├── Lineage stages ──────────────────────────▶ unchanged rank
         │
         └── Phenotype stages (name matches substrings)
                    │
                    │  if seed_known_markers=true
                    ▼
              Load catalogue TSV
              Match Contig:Pos:Ref:Alt present in X
                    │
                    ▼
              Known markers FIRST in rank list
              Top-N panels always include them, then fill with stats
                    │
                    ▼
              Probe / train as usual
```

### Config keys

| Key | Default | Meaning |
|-----|---------|---------|
| `seed_known_markers` | `false` | Master switch |
| `known_markers_path` | `null` | WHO-style catalogue TSV or `Feature_ID` list |
| `seed_known_markers_mode` | `force_include` | Known markers occupy first panel slots |
| `seed_known_markers_stage_substrings` | `amr,resistance,pheno,profile,resistant,susceptible` | Stages that seed |
| `seed_known_markers_max` | `null` | Optional cap on number of seeded markers |

### Catalogue format

Same as `Data/AFRO_TB/metadata/resistance_catalogue.tsv`:

- columns: `Position`, `Ref`, `Alt`, `Contig` (optional), or  
- `Feature_ID` = `M.tuberculosis_H37Rv:761155:C:T`

**Only markers present in the filtered training matrix are seeded** — rare alleles absent from the cohort are not invented.

### Example train JSON

```json
{
  "seed_known_markers": true,
  "known_markers_path": "/path/to/resistance_catalogue.tsv",
  "seed_known_markers_mode": "force_include",
  "seed_known_markers_stage_substrings": "amr,resistance,profile,resistant"
}
```

Ready-made configs:

| File | Use |
|------|-----|
| `scripts/testing_scripts/afro_seed_known_markers_config.json` | Light AFRO phenotype A/B |
| `Data/PRJCA040523/scripts/example_seed_known_markers_config.json` | Example seed keys |
| `scripts/testing_scripts/afro_vcf_config.json` | AFRO VCF policy **without** seed |

### CLI distinction (do not confuse)

| Mechanism | What it does |
|-----------|----------------|
| **`seed_known_markers` (config)** | Force-includes catalogue alleles into **phenotype panels during training** |
| **CLI `--known_markers`** | Writes **overlap report only** — does **not** change feature selection or training |

### Audit artefacts (when applied)

- `feature_panel_separability/ranked_features.csv` — column `known_marker_seed`  
- `feature_panel_separability_summary.json` → `known_marker_seed` block  
- `n_known_markers_in_selected_panel`

### Fair A/B design

1. **Control:** same split/filter, `seed_known_markers=false`  
2. **Seeded:** identical setup, `seed_known_markers=true` + catalogue  
3. Compare AFRO hold-out; external **pDST** cleaner than gDST (catalogue circularity risk on in-distribution labels)  
4. Report **profile recall conditional on correct AMR** (false-S at L2 still kills L3)

### Light experiment (recommended first)

```bash
cd scripts/testing_scripts

# Seeded: AMR → Profile only (~3 nodes)
bash 10_phenotype_AMR_profile_known_marker_seed.sh

# Control arm
SEED_KNOWN_MARKERS=0 bash 10_phenotype_AMR_profile_known_marker_seed.sh
```

Outputs:

- `Results/.../Phenotype_AMR_Profile_seeded_10/`  
- `Results/.../Phenotype_AMR_Profile_control_10/`

CV / annotate off by default (`RUN_LEAKAGE_AWARE_CV=1`, `RUN_PANEL_ANNOTATION=1` to enable).

### Caveats

- Catalogue ∩ training labels can inflate **in-distribution** scores.  
- Seeding is not a fix for L2 false-susceptible routing.  
- Code: `network_parser/known_marker_seed.py` → hook in `feature_panel_selection.py`.

---

## 6. Query path (inference)

```text
  New sample(s)
       │
       ├── matrix ──────────────────────┐
       ├── VCF/gVCF ────────────────────┤
       ├── FASTA (marker BLAST) ────────┤
       └── FASTQ ─▶ (see §7) ───────────┤
                                        ▼
                         Align to *training* feature list
                         (missing → callability / absence policy)
                                        ▼
                         Walk hierarchy route (or single model)
                                        ▼
                         query_predictions · route_audit · reports
```

Query **does not** re-run chi2 / panel discovery / training.

`--query_input_type`: `auto` | `matrix` | `vcf` | `fasta` | `fastq` (`raw_sequence` = deprecated FASTA alias).

---

## 7. FASTQ → VCF options (query preprocessing)

FASTQ is **not** trained on directly. Query-time bridge:

```text
  Paired FASTQ
       │
       ▼
  BWA mem → samtools sort → BAM
       │
       ▼
  Allele calling (mode choice)  ──▶  per-sample VCF.GZ
       │
       ▼
  Same VCF loader + callability contract as training
       │
       ▼
  Align to frozen training features → hierarchy walk
```

Requires on `PATH`: **bwa**, **samtools**, **bcftools**.  
Module: `network_parser/fastq_processor.py` (+ `panel_pileup_caller.py` for majority mode).

### Three call modes (`fastq_call_mode`)

```text
                    ┌─────────────────────────────────────┐
   BAM ────────────▶│  fastq_call_mode                    │
                    └─────────────────────────────────────┘
                         │            │            │
              full       │  panel_    │  panel_    │
                         │  bcftools  │  majority  │
                         ▼            ▼            ▼
              whole-genome     bcftools on     samtools pileup
              bcftools         panel BED only  on panel sites
              mpileup|call     mpileup|call    + majority base
                         │            │            │
                         └────────────┴────────────┘
                                      │
                                      ▼
                               compact / full VCF.GZ
```

| Mode | What happens | When to use | Cost |
|------|----------------|-------------|------|
| **`full`** (legacy default) | Whole-genome `bcftools mpileup \| call` | Need gVCF-like breadth, debug, or non-panel work | Slowest; largest VCF |
| **`panel_bcftools`** | `bcftools mpileup -R panel.bed \| call` on **trained marker sites only** | Standard restricted calling; still bcftools GT | Call step much faster; align still dominates |
| **`panel_majority`** | `samtools mpileup -l panel.bed` → count A/C/G/T → majority/median base | Fast external cohorts (e.g. PRJCA); “count the bases” | Call step fastest; tiny VCF |

**PRJCA external query config** uses **`panel_majority`** by default  
(`Data/PRJCA040523/scripts/prjca_fastq_config.json`).

### How `panel_majority` decides an allele

At each panel SNP (after MQ / base-quality filters):

1. Count A / C / G / T  
2. Require depth ≥ `fastq_panel_min_depth` (default **10**)  
3. Require majority fraction ≥ `fastq_panel_min_majority_fraction` (default **0.7**)  
4. Called base = majority allele  
5. If called == REF → often **omitted** when `fastq_panel_emit_reference_sites=false` (pair with `assume_absent_variant_is_reference=true`)  
6. If called == panel ALT → emit ALT genotype in compact VCF  

Query engine injects **Feature_IDs from the trained selected-feature manifest** automatically in panel modes (no manual BED required unless you override).

### FASTQ / panel config keys

| Key | Default | Role |
|-----|---------|------|
| `fastq_call_mode` | `full` | `full` \| `panel_bcftools` \| `panel_majority` |
| `fastq_panel_min_depth` | `10` | Min pileup depth (majority) |
| `fastq_panel_min_majority_fraction` | `0.7` | Majority threshold |
| `fastq_panel_min_base_quality` | `20` | Base-Q filter for pileup |
| `fastq_panel_emit_reference_sites` | `false` | Emit REF calls in VCF |
| `fastq_panel_sites_bed` | `null` | Optional BED override |
| `fastq_panel_manifest` | `null` | Optional Feature_ID TSV override |
| `fastq_emit_gvcf` | `true` | gVCF emission (mainly `full` path) |
| `fastq_gvcf_min_dp` | `10` | Min DP for gVCF REF blocks |
| `fastq_min_mapping_quality` | `20` | Mapping quality into mpileup |
| `fastq_max_parallel_samples` | `1` | Parallel sample jobs |
| `fastq_threads` | `null` | Threads budget |
| `fastq_clean_intermediates` | `false` | Delete BAM/temp after VCF |
| `fastq_auto_index_reference` | `true` | Index ref if missing |
| `fastq_multi_lane_policy` | `fail` | `fail` \| `merge` multi-lane pairs |
| `fastq_ploidy` | `1` | Haploid microbial default |

### Example: PRJCA-style FASTQ query config

```json
{
  "min_gq_per_sample": 0,
  "assume_absent_variant_is_reference": true,
  "fastq_emit_gvcf": false,
  "fastq_call_mode": "panel_majority",
  "fastq_panel_min_depth": 10,
  "fastq_panel_min_majority_fraction": 0.7,
  "fastq_panel_min_base_quality": 20,
  "fastq_panel_emit_reference_sites": false,
  "min_feature_recovery_fraction": 0.5,
  "min_callable_fraction": 0.5,
  "enforce_query_callability_gates": true
}
```

### Expected speedup (panel vs full)

| Stage | Effect of panel modes |
|-------|------------------------|
| **Alignment (BWA)** | Unchanged — still dominates wall time |
| **Calling** | Much faster (panel sites only) |
| **Overall** | Often ~**20–30%** wall-time save |
| **VCF size** | Tiny vs whole-genome |

### When to pick which mode

```text
  Need whole-genome / REF blocks for diagnostics?  ──▶  full
  Want standard GT on trained sites only?          ──▶  panel_bcftools
  Want fastest external FASTQ pilot / majority?    ──▶  panel_majority
  Already have VCF/gVCF from another pipeline?     ──▶  query --query_input_type vcf
                                                      (skip FASTQ bridge entirely)
```

### CLI hints

```bash
python -m network_parser.cli query \
  --config prjca_fastq_config.json \
  --genomic /path/to/fastq_dir \
  --bundle model.npb \
  --query_input_type fastq \
  --ref_fasta H37Rv.fa \
  --output_dir query_out \
  --n_jobs 4
```

Related flags: `--fastq_max_parallel_samples`, `--fastq_threads`, `--fastq_clean_intermediates`, `--fastq_min_mapping_quality`.

---

## 8. End-to-end AFRO + external map

```text
  AFRO train VCFs + meta              AFRO hold-out VCFs
           │                                  │
           ▼                                  │
  train-hierarchy                             │
    · full 01_  Lineage→AMR→Profile           │
    · light 10_ AMR→Profile ± known seed      │
           │                                  │
           ▼                                  ▼
  registry + bundle  ──────────────────▶  query (VCF) ──▶ evaluate-hierarchy
           │
           ├─ annotate-panels (optional; catalogue context, not seed)
           ├─ cross-validate (optional)
           │
           └─ external PRJCA FASTQ
                    │
                    ▼
              FASTQ→BAM→VCF (panel_majority)
                    │
                    ▼
              query → pDST / gDST evaluation
```

| Recipe | Hierarchy | Seed | Cost |
|--------|-----------|------|------|
| Full `01_` | Lineage → AMR → Profile | off (default) | Heavy (~23 nodes) |
| Light `10_` seeded | AMR → Profile | **on** | Light (~3 nodes) |
| Light `10_` control | AMR → Profile | off | Light (~3 nodes) |

---

## 9. Artifacts that matter

| Artifact | Meaning |
|----------|---------|
| `hierarchical_model_registry.json` | Tree of nodes + model paths |
| `networkparser_model_bundle.npb` | Portable package |
| `hierarchy_node_dashboard.tsv` | Per-node status / algo / n |
| `feature_panel_separability/*` | Ranked markers; `known_marker_seed` if used |
| `query_predictions.csv` | Predictions |
| `query_route_audit.json` | Path + fallbacks |
| `query_alignment_summary.json` | Feature recovery / callability |
| `evaluate/*` | Metrics, confusions, path scores |
| FASTQ work dir VCFs | Intermediate genotypes from §7 |

---

## 10. Design principles (one glance)

1. **Leakage discipline** — filter / panel / impute fit on train only.  
2. **Traceability** — every call has markers + hierarchy route.  
3. **Conservative panels** — fail node if panel score < threshold (no silent full-matrix cheat).  
4. **Callability** — missing genotypes are first-class (unless you opt into absence-as-REF).  
5. **Optional biology seed** — known resistance markers for **phenotype** nodes only; off by default.  
6. **Flexible FASTQ bridge** — full genome call, panel-bcftools, or panel-majority pileup; or skip to ready VCFs.  
7. **Research tool** — external validation (e.g. PRJCA pDST) is the hard test.

---

## Related docs & code

| Topic | Location |
|-------|----------|
| CLI / install | `README.md` |
| Known-marker seed | `docs/KNOWN_MARKER_SEED.md`, `network_parser/known_marker_seed.py` |
| Panel FASTQ calling | `Data/PRJCA040523/scripts/README_PANEL_CALLING.md`, `panel_pileup_caller.py` |
| FASTQ processor | `network_parser/fastq_processor.py` |
| Full hierarchy script | `scripts/testing_scripts/01_Lineage_AMR_Resistance_Profile.sh` |
| Light seed A/B | `scripts/testing_scripts/10_phenotype_AMR_profile_known_marker_seed.sh` |
| AFRO VCF config | `scripts/testing_scripts/afro_vcf_config.json` |
| AFRO + seed config | `scripts/testing_scripts/afro_seed_known_markers_config.json` |
| PRJCA FASTQ config | `Data/PRJCA040523/scripts/prjca_fastq_config.json` |
