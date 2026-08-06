# Optional known-marker seed (phenotype endpoints)

## Goal

Add **known resistance mutations** (e.g. WHO catalogue) into feature panels for **AMR / resistance-profile** nodes, as an **opt-in** experiment to test whether forcing biologically known alleles improves profile recall.

**Default: off.** Statistical ranking is unchanged until you enable it.

## How it works

```text
Central filter (chi2/FDR/…)  →  retained matrix X
         │
         ▼
Rank features by p-value / effect
         │
         ▼  [if seed_known_markers=True AND stage is phenotype-like]
Load catalogue → match Contig:Pos:Ref:Alt present in X
         │
         ▼
Known markers FIRST in rank list
         │
         ▼
Top-N panels always include known hits, then fill with stats rank
         │
         ▼
Probe / train as usual
```

Lineage-only stages are **skipped** unless their stage name matches your substrings.

## Config (all optional)

| Key | Default | Meaning |
|-----|---------|---------|
| `seed_known_markers` | `false` | Master switch |
| `known_markers_path` | `null` | Catalogue TSV or Feature_ID list |
| `seed_known_markers_mode` | `force_include` | Known markers occupy first panel slots |
| `seed_known_markers_stage_substrings` | `amr,resistance,pheno,profile,…` | Which stages seed |
| `seed_known_markers_max` | `null` | Cap number of known markers |

### Catalogue format

Same as `Data/AFRO_TB/metadata/resistance_catalogue.tsv`:

- columns: `Position`, `Ref`, `Alt`, `Contig` (optional)
- or lines / column `Feature_ID` = `M.tuberculosis_H37Rv:761155:C:T`

Only markers **present in the filtered training matrix** are seeded (no hallucinated columns).

## Example train config snippet

```json
{
  "seed_known_markers": true,
  "known_markers_path": "/path/to/resistance_catalogue.tsv",
  "seed_known_markers_mode": "force_include",
  "seed_known_markers_stage_substrings": "amr,resistance,profile,resistant"
}
```

Example file for PRJCA experiments:

`Data/PRJCA040523/scripts/example_seed_known_markers_config.json`

## Audit artefacts

Per node, after panel selection:

- `feature_panel_separability/ranked_features.csv` — column `known_marker_seed` when applied  
- `feature_panel_separability_summary.json` → `known_marker_seed` block  
- `n_known_markers_in_selected_panel`

## Fair experiment design

1. **Baseline:** `seed_known_markers=false` (current AFRO models)  
2. **Seeded:** same split/filter, `seed_known_markers=true` + catalogue  
3. Compare: AFRO hold-out + PRJCA external (pDST and gDST)  
4. Report **profile recall conditional on correct AMR** (L2 false-S still kills L3)

### Light A/B (recommended first)

Skip lineage; train only phenotype stages (~3 nodes vs ~23 on the full tree):

```bash
cd scripts/testing_scripts

# Arm A — known markers forced into AMR/profile panels
bash 10_phenotype_AMR_profile_known_marker_seed.sh

# Arm B — same 2-level recipe, no seed (control)
SEED_KNOWN_MARKERS=0 bash 10_phenotype_AMR_profile_known_marker_seed.sh
```

Outputs:

- `Results/.../Phenotype_AMR_Profile_seeded_10/`
- `Results/.../Phenotype_AMR_Profile_control_10/`

CV and panel annotation are **off** by default; set `RUN_LEAKAGE_AWARE_CV=1` or `RUN_PANEL_ANNOTATION=1` if needed.

## Caveats

- Catalogue overlap with AFRO gDST labels can inflate **in-distribution** scores (circularity). External **pDST** is the cleaner test.  
- Known markers must exist as columns after filtering; rare alleles absent from the cohort are not invented.  
- Seeding is not a substitute for fixing L2 false-susceptible routing.

## Code

- `network_parser/known_marker_seed.py`  
- Hook: `feature_panel_selection.run_feature_panel_separability_check`  
- Config: `NetworkParserConfig.seed_known_markers*`  
