#!/bin/bash
#PBS -l select=1:ncpus=24:mpiprocs=24:mem=200GB
#PBS -P RCHPC
#PBS -l walltime=48:00:00
#PBS -o /mnt/lustre/users/
#PBS -e /mnt/lustre/users/USERNAME/OMP_test/test1.err
#PBS -m abe
#PBS -M nmfuphi@csir.co.za
ulimit -s unlimited

module load chpc/BIOMODULES
module load anaconda/3

eval "$(conda shell.bash hook)" 
conda activate networkparser
cd /home/nmfuphi/network_parser/

python -m network_parser.cli 
--genomic /mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_TB_100_VCFs \
--meta    /mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_dataset_meta.csv \
--label   Lineage \
--output-dir testing
