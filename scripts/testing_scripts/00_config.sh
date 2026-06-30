#!/bin/bash
# Shared path/config variables for NetworkParser training -> query -> evaluate runs.
# Edit these paths if your project folders change.

set -euo pipefail

TRAIN_GENOMIC="/Users/nmfuphicsir.co.za/Documents/pHDProject/Data/AFRO_TB/subset_valid_hierarchy/train"
TEST_GENOMIC="/Users/nmfuphicsir.co.za/Documents/pHDProject/Data/AFRO_TB/subset_valid_hierarchy/test"
META="/Users/nmfuphicsir.co.za/Documents/pHDProject/Data/AFRO_TB/metadata/AFRO_dataset_meta_with_test_hierarchy.csv"
REF="/Users/nmfuphicsir.co.za/Documents/pHDProject/Code/VCF_2_Matrix/ref/H37Rv.gbk"
BASE_OUT="/Users/nmfuphicsir.co.za/Documents/pHDProject/Results/All_VCFs/chi2_fdr"

N_JOBS=8
FILTER="chi2_fdr"
QUERY_INPUT_TYPE="vcf"
