#!/usr/bin/env bash
#
# vcf_sanity_checks.sh
# Quick sanity / quality checks for VCF(.gz) files — especially useful for microbial SNP datasets
#
# Usage:
#   ./vcf_sanity_checks.sh file.vcf.gz
#   ./vcf_sanity_checks.sh file.vcf.gz > sanity_report.txt 2>&1
#
# Requirements: bcftools, tabix (and grep, awk, wc, sort, head, tail)
#

set -u
set -e

VCF="$1"

if [[ -z "$VCF" ]]; then
    echo "Error: Please provide a VCF file"
    echo "Usage: $0 yourfile.vcf.gz"
    exit 1
fi

if [[ ! -f "$VCF" ]]; then
    echo "Error: File not found: $VCF"
    exit 1
fi

# Optional: make sure it's indexed (most commands are faster with index)
if [[ ! -f "${VCF}.tbi" ]]; then
    echo "→ Indexing VCF (creating .tbi) ..."
    tabix -p vcf "$VCF" || { echo "Warning: could not index VCF"; }
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  VCF SANITY CHECK REPORT — $(basename "$VCF")"
echo "══════════════════════════════════════════════════════════════"
echo "Date: $(date)"
echo ""

# ──────────────────────────────────────────────────────────────
# 1. Basic file & header information
# ──────────────────────────────────────────────────────────────

echo "1. Number of samples"
bcftools query -l "$VCF" | wc -l
echo ""

echo "2. Number of variants (lines after header)"
bcftools view -H "$VCF" | wc -l
echo ""

echo "3. Number of contigs in header"
bcftools view -h "$VCF" | grep "^##contig=" | wc -l
echo ""

# ──────────────────────────────────────────────────────────────
# 2. Genotype call summary
# ──────────────────────────────────────────────────────────────

echo "4. Haploid ALT calls (\"1\")"
bcftools query -f '[%GT\n]' "$VCF" | grep -c "^1$"
echo ""

echo "5. Diploid heterozygous calls (0/1 or 0|1)"
bcftools query -f '[%GT\n]' "$VCF" | grep -c "0/1\|0|1"
echo ""

echo "6. Missing genotypes (./. or .)"
bcftools query -f '[%GT\n]' "$VCF" | grep -c "\./\.\|\.$"
echo ""

echo "7. Homozygous REF calls (0 or 0/0 or 0|0)"
bcftools query -f '[%GT\n]' "$VCF" | grep -c "^0$\|^0/0\|^0|0"
echo ""

# ──────────────────────────────────────────────────────────────
# 3. Quick quality & coverage checks
# ──────────────────────────────────────────────────────────────

echo "8. Variants with QUAL < 30 (low quality)"
bcftools view -i 'QUAL<30' "$VCF" | grep -v "^#" | wc -l
echo ""

echo "9. Average DP per genotype (if DP tag exists)"
if bcftools view -h "$VCF" | grep -q "##FORMAT=<ID=DP"; then
    bcftools query -f '[%DP ]\n' "$VCF" \
        | awk '{sum+=$1; n++} END {if (n>0) printf "%.1f\n", sum/n; else print "N/A"}'
else
    echo "DP tag not found"
fi
echo ""

echo "10. Genotypes with low depth (DP < 10)"
if bcftools view -h "$VCF" | grep -q "##FORMAT=<ID=DP"; then
    bcftools view -i 'FMT/DP<10' "$VCF" | grep -v "^#" | wc -l
else
    echo "DP tag not found"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# 4. Variant type composition
# ──────────────────────────────────────────────────────────────

echo "11. Number of indels (REF and ALT length differ)"
bcftools query -f '%REF %ALT\n' "$VCF" \
    | awk 'length($1) != length($2) {cnt++} END {print cnt+0}'
echo ""

echo "12. Multi-allelic sites (ALT contains comma)"
bcftools query -f '%ALT\n' "$VCF" | grep -c ","
echo ""

# ──────────────────────────────────────────────────────────────
# 5. Allele frequency rough overview
# ──────────────────────────────────────────────────────────────

echo "13. Lowest 8 MAFs (non-zero)"
bcftools query -f '%AC %AN\n' "$VCF" \
    | awk '$2>0 {printf "%.4f\n", $1/$2}' \
    | sort -n | head -8
echo ""

echo "14. Highest 8 MAFs (should be ≤ 1)"
bcftools query -f '%AC %AN\n' "$VCF" \
    | awk '$2>0 {printf "%.4f\n", $1/$2}' \
    | sort -nr | head -8
echo ""

# ──────────────────────────────────────────────────────────────
# 6. Quick sample-level missingness (first 5 samples only)
# ──────────────────────────────────────────────────────────────

echo "15. Missingness per sample — first 5 samples"
echo "(sample name → missing genotypes %)"
{
    mapfile -t SAMPLES < <(bcftools query -l "$VCF" | head -5)
    for s in "${SAMPLES[@]}"; do
        total=$(bcftools query -s "$s" -f '[%GT\n]' "$VCF" | wc -l)
        miss=$(bcftools query -s "$s" -f '[%GT\n]' "$VCF" | grep -c "\./\.\|\.")
        perc=$(awk -v m="$miss" -v t="$total" 'BEGIN {if(t>0) printf "%.2f", (m/t)*100; else print "N/A"}')
        printf "%-25s  %6s%%\n" "$s" "$perc"
    done
}
echo ""

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  End of report"
echo "══════════════════════════════════════════════════════════════"
echo ""

exit 0
