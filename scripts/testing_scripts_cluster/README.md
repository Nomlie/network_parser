# NetworkParser cluster (CHPC / PBS) validation scripts

Twin of `scripts/testing_scripts`, adapted for CHPC PBS jobs and lustre paths.

## Paths (defaults in `00_config.sh`)

| Role | Path |
|------|------|
| Data root | `/mnt/lustre/users/nmfuphi/phDproject/Data` |
| Unsplit VCF pool (existing) | `/mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_TB_ANNOTATION_VCF` |
| Training / test VCFs | Same pool; split via manifests + symlinks only |
| Metadata + manifests + catalogue | `Data/meta/` |
| H37Rv GenBank | `Data/ref/H37Rv.gbk` |
| Clean symlink split (generated) | `Data/networkparser_manifest_split/{train,test}/` |
| Results | `/mnt/lustre/users/nmfuphi/phDproject/Results` |
| Experiment outputs | `Results/All_VCFs/chi2_fdr/<RUN_NAME>/` |
| PBS stdout/err | `Results/pbs_logs/` |
| Code / package | `/home/nmfuphi/network_parser` |
| PRJCA040523 (after rsync) | `Data/PRJCA040523/{CRA025985,meta,ref,scripts}` |

### Expected layout under `Data/meta/`

```
Data/meta/
  train_samples.csv
  test_samples.csv
  AFRO_dataset_meta_with_test_hierarchy.csv
  resistance_catalogue.tsv
```

Manifests need a sample-ID column (`sample_id`, `sample`, `ID`, or `id`).  
Optional `split` column must be `train` / `test` respectively.

## Files

| File | Role |
|------|------|
| `00_config.sh` | Cluster paths, `N_JOBS=24`, flags (`SEED_KNOWN_MARKERS`, CV off by default) |
| `00_env_cluster.sh` | `module load`, `conda activate networkparser`, job banner |
| `00_prepare_inputs.sh` | Symlink train/test views + `Lineage_family` metadata |
| `00_preflight.sh` | Path / VCF / catalogue checks |
| `01_Lineage_AMR_Resistance_Profile.sh` | **Seeded** hierarchy train → query → evaluate → annotate (**no CV**) |
| `11_leakage_aware_cv.sh` | **Standalone** leakage-aware CV (train partition) |
| `12_prjca_external_fastq_query.sh` | PRJCA FASTQ query with **`panel_bcftools`** + evaluate |
| `13_prjca_phenotype_seeded10_vcf_query.sh` | Query **Phenotype_AMR_Profile_seeded_10** on **existing Chinese VCFs** + evaluate (AMR→Profile) |
| `pbs_01_*.pbs` | Main seeded hierarchy job |
| `pbs_11_leakage_aware_cv.pbs` | CV job (same PBS resources as 01) |
| `pbs_12_prjca_external_fastq.pbs` | External PRJCA FASTQ job |
| `pbs_13_prjca_phenotype_seeded10_vcf.pbs` | Phenotype light model × Chinese VCFs |
| `pbs_run_experiment.sh` | Generic: `qsub -v EXPERIMENT=01\|11\|12 ...` |
| `afro_vcf_config.json` | AFRO callability without seed |
| `afro_seed_known_markers_config.json` | Template seed config (01 writes runtime copy with catalogue path) |
| `prjca_panel_bcftools_config.json` | FASTQ query: **panel-only bcftools** (not whole-genome) |
| `sync_prjca040523_to_chpc.sh` | rsync PRJCA data Mac → lustre |

## Recommended job order

```bash
cd /home/nmfuphi/network_parser/scripts/testing_scripts_cluster
mkdir -p /mnt/lustre/users/nmfuphi/phDproject/Results/pbs_logs

# 1) Train hierarchy with WHO catalogue seeded on phenotype stages (AMR / profile)
qsub pbs_01_Lineage_AMR_Resistance_Profile.pbs
# Output: Results/All_VCFs/chi2_fdr/Hierarchy_Lineage_AMR_Resistance_Profile_seeded_01/

# 2) Leakage-aware CV (separate job, same 24 CPU / 120 GB / 96 h)
qsub pbs_11_leakage_aware_cv.pbs
# or after 01 finishes, pin the run dir:
# qsub -v RUN_DIR=/mnt/lustre/users/nmfuphi/phDproject/Results/All_VCFs/chi2_fdr/Hierarchy_Lineage_AMR_Resistance_Profile_seeded_01 \
#      pbs_11_leakage_aware_cv.pbs

# 3) External Chinese cohort FASTQ query (panel_bcftools only)
qsub pbs_12_prjca_external_fastq.pbs
# smoke: qsub -v LIMIT=5 pbs_12_prjca_external_fastq.pbs

# 4) Light phenotype model (seeded_10) on already-called Chinese VCFs
qsub pbs_13_prjca_phenotype_seeded10_vcf.pbs
# control arm: qsub -v ARM=control pbs_13_prjca_phenotype_seeded10_vcf.pbs
```

### Local Mac: phenotype seeded_10 × Chinese VCFs (no re-calling)

```bash
cd /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/network_parser
export PYTHONPATH=.

# All available Chinese final/vcf (~198)
bash ../Data/PRJCA040523/scripts/query_phenotype_seeded10_chinese_vcfs.sh
# path if run from repo:
bash /Users/nmfuphicsir.co.za/Documents/pHDProject/Data/PRJCA040523/scripts/query_phenotype_seeded10_chinese_vcfs.sh

# Smoke
LIMIT=10 bash /Users/nmfuphicsir.co.za/Documents/pHDProject/Data/PRJCA040523/scripts/query_phenotype_seeded10_chinese_vcfs.sh

# Control (unseeded light model)
ARM=control bash /Users/nmfuphicsir.co.za/Documents/pHDProject/Data/PRJCA040523/scripts/query_phenotype_seeded10_chinese_vcfs.sh
```

Outputs under:
`Results/PRJCA040523_external/Phenotype_AMR_Profile_seeded_10_chinese_vcfs/`


### Unseeded control (optional)

```bash
qsub -v SEED_KNOWN_MARKERS=0 pbs_01_Lineage_AMR_Resistance_Profile.pbs
# → Hierarchy_Lineage_AMR_Resistance_Profile_01 (no seed)
```

### What changed vs older cluster scripts

1. **WHO seed on phenotype endpoints** — default `SEED_KNOWN_MARKERS=1` for experiment 01; catalogue alleles present in the filtered matrix are force-included first for AMR/profile stages only.  
2. **No leakage-aware CV in the main PBS job** — use `pbs_11_leakage_aware_cv.pbs` (same `#PBS` resources).  
3. **PRJCA FASTQ calling** — `fastq_call_mode=panel_bcftools` (trained sites only), not whole-genome `full`.

## Copy PRJCA040523 to CHPC (from Mac)

```bash
# Small first (meta/ref/scripts only)
bash scripts/testing_scripts_cluster/sync_prjca040523_to_chpc.sh --meta-only

# Full FASTQs (~100GB+ CRA025985) — use screen/tmux
bash scripts/testing_scripts_cluster/sync_prjca040523_to_chpc.sh

# Dry-run
bash scripts/testing_scripts_cluster/sync_prjca040523_to_chpc.sh --dry-run
```

Defaults: `nmfuphi@scp.chpc.ac.za` (CHPC data-transfer node) →  
`/mnt/lustre/users/nmfuphi/phDproject/Data/PRJCA040523`  
Override with `CHPC_HOST=lengau.chpc.ac.za` only if needed.

Also sync the **code** checkout (same transfer host by default):

```bash
bash scripts/testing_scripts_cluster/sync_code_to_chpc.sh
# or:
rsync -avh --exclude '.git' --exclude '__pycache__' --exclude '.mypy_cache' \
  /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/network_parser/ \
  nmfuphi@scp.chpc.ac.za:/home/nmfuphi/network_parser/
```

Ensure `Data/meta/resistance_catalogue.tsv` on lustre is the WHO catalogue used for seeding (same file as local AFRO catalogue if that is your 2nd-ed TSV).

## One-time setup on the cluster

1. Sync the repo to `/home/nmfuphi/network_parser` (or set `PROJECT_ROOT` / `CLUSTER_SCRIPTS`).
2. Install / activate conda env `networkparser` with the package importable:
   ```bash
   conda activate networkparser
   python -c "import network_parser; print(network_parser.__file__)"
   ```
3. Stage AFRO data on lustre (VCFs already under `AFRO_TB_ANNOTATION_VCF`; meta + ref as above).
4. Stage PRJCA via `sync_prjca040523_to_chpc.sh`.
5. Tools for FASTQ query: `bwa`, `samtools`, `bcftools` on PATH in the env.

## Resource defaults

- `select=1:ncpus=24:mem=120GB`
- `walltime=96:00:00`
- project `RCHPC`
- email `nmfuphi@csir.co.za` on begin/end/abort (`-m abe`)
- `N_JOBS=24` for train / query / CV

Thread libraries are pinned to 1 OS thread each so sklearn/joblib owns the 24 cores.

## Local vs cluster

| | `testing_scripts` | `testing_scripts_cluster` |
|--|-------------------|---------------------------|
| Paths | macOS `Documents/pHDProject/...` | lustre `.../phDproject/{Data,Results}` |
| Launch | `bash 01_....sh` | `qsub pbs_01_....pbs` |
| Conda | local env | `conda activate networkparser` via PBS |
| 01 CV | optional env flag | **separate PBS 11** |
| 01 seed | optional (local light script 10) | **default on** |
| PRJCA FASTQ | local scripts | **PBS 12 + panel_bcftools** |
