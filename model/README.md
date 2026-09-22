# Trained NetworkParser model

This folder ships the portable AFRO hierarchy model:

| File | Use |
|---|---|
| `networkparser_model_bundle.npb` | `query --bundle` |
| `hierarchical_model_registry.json` | `query --registry`, `annotate-panels`, or `bundle` |

Hierarchy: `Lineage_clean` → `AMR_binary` → `Resistance_Profile_Collapsed`  
Trained on the AFRO-TB VCF cohort (chi2-FDR experiment `Hierarchy_Lineage_AMR_Resistance_Profile_01`; 10,974 training samples). Bundle schema 1.3.

Query against new samples with `--bundle model/networkparser_model_bundle.npb`. Retraining with `--output_dir model` replaces these files.

`.npb` files contain Python pickle objects. Load them only from this repository or another trusted training run.
