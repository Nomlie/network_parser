#!/bin/bash
#PBS -l select=1:ncpus=24:mpiprocs=24:mem=240GB
#PBS -P RCHPC
#PBS -l walltime=48:00:00
#PBS -o /mnt/lustre/users/nmfuphi/testing_network_parser/LineageBalanced/lineagebalance.out
#PBS -e /mnt/lustre/users/nmfuphi/testing_network_parser/LineageBalanced/lineagebalance.err
#PBS -m abe
#PBS -M nmfuphi@csir.co.za

ulimit -s unlimited

module load chpc/BIOMODULES
module load anaconda/3

eval "$(conda shell.bash hook)"
conda activate networkparser
cd /home/nmfuphi/network_parser/

echo "Starting job at $(date)"
echo "Hostname: $(hostname)"
echo "Free memory: $(free -h)"

python -m network_parser.cli \
    --genomic /mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_TB_LineageBalanced \
    --meta    /mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_dataset_meta.csv \
    --label   Lineage \
    --output-dir /mnt/lustre/users/nmfuphi/testing_network_parser/LineageBalanced/

echo "Job finished at $(date)"