# Audit of Repository Claims & Discrepancy Log

> **Document**: `docs/readme_corrections.md`
> **Generated**: 2026-11-16T15:53:55.569740+00:00
> **Status**: Audit findings documented. README updates staged for Phase 5.

## Contradictions Identified During Data & Model Audit

| Previous README Claim | Verified Empirical Fact | Discrepancy Severity | Evidence / Reference |
| :--- | :--- | :---: | :--- |
| *'Real-world training corpus combining deepset, JasperLS, curated indirect injection pairs, and augmented Hinglish variants'* | The training CSV contains **only two public datasets**: `deepset/prompt-injections` (660 rows) and `S-Labs/prompt-injection-dataset` (11,023 rows). Curated indirect pairs and augmented Hinglish are **0% present in the training CSV**. | 🔴 High | Normalized SHA-256 hash match against HF datasets (`results/provenance_verification.json`). |
| *'JasperLS/prompt-injections is a distinct source dataset'* | `JasperLS/prompt-injections` is a **100% byte-for-byte clone / mirror** of `deepset/prompt-injections`. Both share identical parquet file SHA-256 hashes (`2e10bc7a...` and `39ac797c...`). | 🟡 Medium | Direct SHA-256 comparison of parquet files from both Hugging Face repos. |
| *'deepset count: 660 samples'* | The training CSV has **659 deepset samples** (one duplicate dropped during initial concatenation). | 🟢 Low | Row-hash matching (`scripts/audit_dataset.py`). |
| *'Classifier AUROC: 0.9993, F1: 99.47%'* | These legacy figures were computed on a held-out split that suffered **data contamination** from previous training runs. | 🔴 High | Splitting protocol audit (Amendment 3). Clean, uncontaminated results require retraining on the verified cluster-isolated splits. |
| *'Utility 100% across all defenses'* | The 50% FPR in `baseline_ablation_report.json` was **100% caused by Layer 2 (DistilBERT classifier)** false alarms on `fetch_page` tool outputs (confidence 0.6775 & 0.9947). Layer 3 policy checker passed both benign cases. Blaming Layer 3 for a '50% FPR on email tasks' was an incorrect misattribution in legacy documentation. However, Layer 3 does suffer severe argument-hijacking and negation vulnerabilities. | 🔴 High | Verified via empirical attribution in `scripts/run_phase1d_evaluation_integrity.py`. |

## Recommended Paper Corrections
1. Disclose exact training provenance: 11,023 rows from S-Labs (Kaggle/HF) + 659 rows from deepset.
2. State that the 500-case indirect prompt injection benchmark is an out-of-distribution (OOD) test set.
3. Explicitly report legacy contaminated numbers alongside newly retrained cluster-isolated numbers.

## Benchmark Taxonomy & Cell Coverage Reality (Audit 2026-11-17)

1. **160-Cell Matrix Sparsity**:
   - The README asserts a 160-cell benchmark (4 injection domains x 8 attack techniques x 5 target actions).
   - **Reality**: Exactly **60 out of 160 cells (37.5%)** are populated. **100 cells (62.5%) are completely empty**.
   - The 450 attacks are generated from only **16 distinct template scenarios** repeated across locations.

2. **Phantom Techniques in README Taxonomy**:
   The following 4 techniques listed in README Section 4 have **zero test cases**:
   - `direct_override` (0 cases in benchmark)
   - `fake_system_delimiter` (0 cases in benchmark)
   - `semantic_camouflage` (0 cases in benchmark)
   - `hypothetical_framing` (0 cases in benchmark)

3. **Unlisted Techniques in Benchmark**:
   The benchmark actually contains:
   - `encoded_payload` (28 cases)
   - `payload_splitting` (28 cases)
   - `context_manipulation` (84 cases)
   These were omitted from the README's 8-technique table.
