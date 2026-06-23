# NetworkParser validation scripts

This folder contains copy-pasteable shell scripts for three NetworkParser experiments:

1. `Two_levels_with_global_AMR_binary_fallback`
2. `Hierarchy_with_AMR_binary`
3. `Hierarchy_with_resistance_profiles`

## Files

- `00_config.sh`  
  Shared path variables and run settings. Edit this file if your paths change.

- `01_two_levels_with_global_AMR_binary_fallback.sh`  
  Standard two-level training: `Lineage_clean -> Resistance_Profile_Collapsed`, with `AMR_binary` as the global Level-2 fallback evaluation target.

- `02_hierarchy_with_AMR_binary.sh`  
  Recursive hierarchy: `Lineage_Supergroup -> Lineage_clean -> AMR_binary`.

- `03_hierarchy_with_resistance_profiles.sh`  
  Recursive hierarchy: `Lineage_Supergroup -> Lineage_clean -> Resistance_Profile_Collapsed`.

- `run_all_networkparser_validation.sh`  
  Runs all three experiments sequentially.

## Run one experiment

```bash
bash 01_two_levels_with_global_AMR_binary_fallback.sh
```

or:

```bash
bash 02_hierarchy_with_AMR_binary.sh
```

or:

```bash
bash 03_hierarchy_with_resistance_profiles.sh
```

## Run all experiments

```bash
bash run_all_networkparser_validation.sh
```

## Output layout

Each experiment writes into:

```text
/Users/nmfuphicsir.co.za/Documents/pHDProject/Results/ALL_VCFs/chi2_fdr/<RUN_NAME>/
```

Each run has:

```text
<RUN_NAME>/
├── networkparser_model_bundle.npb
├── query/
└── evaluate/
```
