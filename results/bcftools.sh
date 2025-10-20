#!/bin/bash
#PBS -l select=1:ncpus=8:mpiprocs=8:mem=4gb
#PBS -N bcftools_mpileup
#PBS -P CBBI0911
#PBS -l walltime=120:00:00
#PBS -q seriallong	
#PBS -e /mnt/lustre/users/bsekhwela/Redfox_project/trimmed/realigned/bam_final/bcftools.err
#PBS -o /mnt/lustre/users/bsekhwela/Redfox_project/trimmed/realigned/bam_final/bcftools.out
#PBS -m abe
#PBS -M blessingsekhwela09@gmail.com

module load chpc/BIOMODULES
module load bcftools/1.9  # Use latest version

cd /mnt/lustre/users/bsekhwela/Redfox_project/trimmed/realigned/bam_final

# Copy BAM files to local storage (optional, if I/O is a bottleneck)
cp *.bam /tmp/
sed 's|/mnt/lustre/users/bsekhwela/Redfox_project/trimmed/realigned/bam_final|/tmp|' allvulpes.txt > allvulpes_local.txt

# Run mpileup
time bcftools mpileup --threads 8 -b allvulpes_local.txt -f /mnt/lustre/users/bsekhwela/Redfox_project/GCF_018345385.1_ASM1834538v1_genomic.fna -o output.mpileup 2> mpileup.err

# Clean up
rm /tmp/*.bam