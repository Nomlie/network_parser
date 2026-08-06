# AI Agent Implementation Checklist Report

Status: **implemented and verified** on 2026-07-20.

This report records the safety contract, implementation evidence, verification
commands, migrations, and remaining scientific limitations for the supplied
15-part checklist.

## Baseline

- Full test suite before checklist work: `107 passed` with 8 warnings.
- Compilation before checklist work: passed.
- Flake8 before checklist work: 1,150 findings with the original defaults; 270
  findings after excluding Black-compatible line-layout rules.
- Black before checklist work: 35 files required formatting.
- Mypy and Pyright were not installed in the active environment.
- The working tree already contained extensive tracked and untracked work. It
  was preserved; no reset, checkout, or unrelated deletion was performed.

## Completion evidence

- [x] **1. Repository reconnaissance.** Training, nested CV, hierarchy,
  bundle, query, annotation, FASTQ, evaluation, visualization, and compatibility
  entry points were inventoried. Duplicate VCF logic and missing-value coercions
  were traced before editing, and baseline tests/static checks were recorded.

- [x] **2. Shared VCF semantics.** `vcf_call_semantics.py` is the shared
  interpreter used by training and query encoding. Missing GT/sample/FORMAT
  data, failed QC/FILTER, mixed calls, incompatible alleles, unsupported ploidy,
  and ambiguous aliases resolve to non-callable states. Variant-only absence is
  unknown by default; the legacy absence-as-reference behavior is opt-in. gVCF
  reference blocks, strict biallelic behavior, explicit contig aliases,
  `.g.vcf.gz` sample IDs, and missing-versus-zero QC values are covered by tests.

- [x] **3. Query callability and abstention.** Global and node-specific gates
  run before inference. Empty/all-missing inputs and inputs below recovery or
  callability thresholds abstain without invoking a model. Outputs include
  reason codes, action, call-state counts, and recovery/callability statistics.
  Registry and bundle query paths use the same engine and gate behavior.

- [x] **4. Missing-value contract.** Matrices now use `0` for callable
  baseline, `1` for callable non-baseline, and `NaN` for non-callable data.
  Sample/feature missingness thresholds are functional. Imputers are fitted on
  training data, serialized with models, and reused unchanged for validation and
  query. Unknown required markers count against callability. Invariant detection
  and visual distances do not reinterpret missing values as reference.

- [x] **5. Nested, group-aware validation.** Repeated nested CV is wired into
  the main pipeline and strict fold-local preprocessing is the default. Outer
  training folds own missingness fitting, invariant/minor-count filters,
  statistical selection, panel selection, model selection, and nested OOF
  thresholds. Outer and inner splitters accept groups and enforce disjointness
  and class/group feasibility. Fold provenance is persisted, failed folds remain
  failures, incomplete runs are not publication-ready, and repeated predictions
  are aggregated by sample before aggregate metrics and intervals.

- [x] **6. Threshold selection.** The default objective is minimum called error
  subject to minimum coverage. Wrong-call and abstention costs are configurable.
  Group-aware OOF inference produces raw support once per fold; cached scores are
  evaluated across the threshold grid. Outputs distinguish support scores from
  calibrated probabilities and record objective, risk, coverage, calibration,
  and OOF provenance. The selected threshold is frozen before deployment fit.

- [x] **7. Model selection and DT triggering.** `decision_tree_candidate_top_k`
  and `decision_tree_candidate_max_margin` are live configuration. A finite DT
  score alone is insufficient; the branch runs only when selected, competitive,
  or explicitly requested. Probe and final DT constraints agree. SVC/MLP are not
  labelled interpretable, model failures are explicit, and clustering diagnostics
  are optional and disabled by default.

- [x] **8. DT stability and interactions.** Bootstrap metadata separates
  requested, successful, and skipped resamples. Successful fits are the stability
  and interval denominator, with a configurable minimum-success check. Pair
  stability requires same-root-to-leaf-path occurrence. Joint support is
  pairwise-complete. MI, Cramér's V, stability, depth, support, p-values, and the
  uncalibrated evidence score remain separate. Results are labelled exploratory
  tree-path interactions rather than epistasis.

- [x] **9. Feature and panel selection.** Missing target labels are removed
  before string conversion. RF-FDR has explicit missing-data preprocessing and
  group/block-aware permutations. Zero permutations are rejected; empirical
  resolution `1/(B+1)` is reported and checked for BH-FDR adequacy. Fallbacks are
  marked exploratory. Panel preprocessing occurs inside panel CV, failures are
  strict by default, tuning estimates are labelled, and nested parallelism uses a
  shared job budget.

- [x] **10. Bundle integrity.** Current schema bundles require one model per
  required successful node, unique IDs, exact ordered features, preprocessing,
  threshold/calibration state, VCF semantics version, baseline alleles, reference
  identity/checksum, and raw-query manifests. Model-only bundles reject raw
  VCF/FASTA input. Strict loading verifies model, manifest, feature-list,
  required-feature, preprocessing, and reference-related content. Missing hashes
  and tampering fail closed. Evaluation consumes the prediction table returned by
  the current query run rather than a recursive first match.

- [x] **11. Annotation.** Exact known-mutation matching requires compatible
  build, contig, position, REF, and ALT. Partial/position-only matches remain
  candidate/unverified, gene matches enforce declared build/contig, and gene/locus
  identifiers are normalized. All overlapping coding features are reported.
  Coordinates are range-checked, circular wrapping is topology-configurable,
  manifests carry reference identity/checksum, and negative-strand allele
  transformation is retained and tested.

- [x] **12. FASTQ processing.** One CPU budget is apportioned across tools and
  samtools sort memory is treated as per-thread. Lanes are paired by parsed sample
  and lane keys, missing mates and sanitized-name collisions fail clearly,
  microbial ploidy is validated, final VCF/gVCF is normalized, and the unused
  depth-callability artifact was removed. Commands, return codes, logs, and tool
  versions are included in provenance.

- [x] **13. Evaluation.** Missing class scores are not fabricated as zero and
  multiclass AUC requires complete genuine vectors. Top-class-only support skips
  AUC with a reason. Arbitrary support is not normalized into probabilities,
  labels retain hyphens, truth samples without predictions remain in audit output,
  and confidence intervals can resample by sample or group. Called-only accuracy,
  coverage, and end-to-end accuracy are reported separately.

- [x] **14. Dead-code cleanup.** Unused imports and clustering work were removed
  or gated, the legacy two-level implementation became a small compatibility shim
  over the hierarchy implementation, current query code uses the shared VCF
  interpreter, the unused FASTQ depth artifact was removed, and visualization/CV
  paths are connected. Compatibility aliases that remain have a documented 2.0
  removal target.

- [x] **15. Mandatory regression tests.** The suite covers empty VCF abstention,
  missing genotypes/sample data, QC failures, gVCF REF blocks and ALT cohort
  baselines, contig identity/aliases, multiallelic policy, callability gates,
  training/query parity, missingness and fold-only imputation, leakage and grouped
  CV, failed-fold reporting, threshold behavior, conditional DT execution,
  successful-bootstrap denominators, bundle completeness/tampering, annotation
  exactness and negative strand, incomplete AUC scores, registry/bundle parity,
  and end-to-end safe VCF/gVCF behavior.

## Changed defaults and migration notes

- Strict fold-local nested CV is enabled by default. Exploratory transductive
  preprocessing must be requested explicitly and is labelled non-publication.
- VCF absence is unknown by default. Legacy absence-as-reference requires an
  explicit option and warning.
- Supported VCF ploidies default to haploid and diploid; other ploidies abstain as
  unresolved unless explicitly supported by future semantics.
- Missing markers remain `NaN` until a training-fitted preprocessor is applied.
- Threshold selection defaults to selective risk under a coverage constraint,
  not end-to-end accuracy.
- Clustering diagnostics are disabled unless requested.
- Raw-query-capable bundle schema 1.2 requires manifests, preprocessing, reference
  identity/checksum, and complete hashes. Older incomplete bundles are model-only
  or rejected by strict loading.
- Deprecated compatibility names remain readable during migration and are
  scheduled for removal in 2.0; new output names describe support/evidence and
  exploratory tree-path interactions without probability or epistasis claims.

## Verification results

Executed from the repository root:

```text
python -m pytest -q
115 passed, 19 warnings in 37.89s

python -m black --check network_parser tests scripts
41 files would be left unchanged

flake8 network_parser tests --count --statistics
0

mypy  # using mypy.ini; isolated mypy 1.17.1 runtime for this run
Success: no issues found in 27 source files

python -m compileall -q network_parser tests scripts
passed

git diff --check
passed

python -m pytest -q \
  tests/test_phase3_tree_evidence.py::TestBootstrapStability::test_deterministic_bootstrap_with_local_rng \
  tests/test_checklist_enforcement.py::TestEndToEndAbstentionParity::test_empty_vcf_abstains_identically_registry_and_bundle
2 passed
```

The 19 warnings are external Matplotlib/PyParsing deprecations and scikit-learn
feature-name warnings in portable bundle fixtures; there are no test failures.

## Remaining scientific and architectural limitations

- The deployed marker contract is binary/biallelic. Mixed, ambiguous,
  multiallelic under strict policy, and unsupported-polyploid calls abstain rather
  than attempting dosage inference.
- Safe reference calls require an explicit genotype or a callable gVCF reference
  block. A variants-only file cannot prove reference callability at absent sites.
- Support scores are not probabilities unless an explicit fitted calibration
  state says otherwise.
- Tree-path interactions and association statistics are exploratory evidence;
  they do not establish biological epistasis or causality.
- Publication claims still depend on representative sampling, correct group
  definitions, adequate independent groups per class, external validation, and a
  reference/catalogue that matches the declared build and checksum.
- FASTQ execution requires external `bwa`, `samtools`, and `bcftools`. The test
  suite validates command construction, budgets, lanes, collisions, and failure
  handling with controlled fixtures; it does not substitute for site-specific
  validation of a production sequencing pipeline.

## Diff review note

The repository was already substantially dirty before this checklist run,
including generated bytecode and existing workflow/script changes. Checklist
work preserved that state and did not revert or delete it. Repository Python
sources and scripts were formatted because formatter compliance was explicitly
part of final verification.
