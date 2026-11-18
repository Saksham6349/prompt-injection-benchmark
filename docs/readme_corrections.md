# Audit of Repository Claims & Discrepancy Log

> **Document**: `docs/readme_corrections.md`
> **Generated**: 2026-11-16T15:53:55.569740+00:00 (Updated 2026-11-18 for Phase 1e Closure)
> **Status**: Audit findings documented. README updates staged for Phase 5.

## Contradictions Identified During Data & Model Audit

| Previous README Claim | Verified Empirical Fact | Discrepancy Severity | Evidence / Reference |
| :--- | :--- | :---: | :--- |
| *'Real-world training corpus combining deepset, JasperLS, curated indirect injection pairs, and augmented Hinglish variants'* | The training CSV contains **only two public datasets**: `deepset/prompt-injections` (660 rows) and `S-Labs/prompt-injection-dataset` (11,023 rows). Curated indirect pairs and augmented Hinglish are **0% present in the training CSV**. | 🔴 High | Normalized SHA-256 hash match against HF datasets (`results/provenance_verification.json`). |
| *'JasperLS/prompt-injections is a distinct source dataset'* | `JasperLS/prompt-injections` is a **100% byte-for-byte clone / mirror** of `deepset/prompt-injections`. Both share identical parquet file SHA-256 hashes (`2e10bc7a...` and `39ac797c...`). | 🟡 Medium | Direct SHA-256 comparison of parquet files from both Hugging Face repos. |
| *'deepset count: 660 samples'* | The training CSV has **659 deepset samples** (one duplicate dropped during initial concatenation). | 🟢 Low | Row-hash matching (`scripts/audit_dataset.py`). |
| *'Classifier AUROC: 0.9993, F1: 99.47%'* | These legacy figures were computed on a held-out split that suffered **data contamination** from previous training runs. | 🔴 High | Splitting protocol audit (Amendment 3). Clean, uncontaminated results require retraining on the verified cluster-isolated splits. |
| *'Our 3-Layer Defense: 0.0% ASR, 50% utility, 112 ms latency'* | **INVALID RESULT**: The pipeline achieved 0.0% ASR purely by blocking all tool outputs. In `sample_20.jsonl`, Layer 2 classified both clean `fetch_page` outputs as attacks (confidence 0.6775 & 0.9947). At deployment threshold $\tau=0.4700$, it blocks 100% of benign sandbox outputs (0.0% utility, 100.0% FPR). At $\tau=0.7000$, it blocks 50% (50.0% utility, 50.0% FPR). It is marked invalid and must not be cited. | 🔴 High | Empirical evaluation in `scripts/run_baselines.py` and `scripts/run_phase1e_closure.py`. |
| *'BUG-04: PolicyChecker (Layer 3) blocked read-only emails, causing 50% FPR'* | **DOUBLE MISATTRIBUTION**: In `sample_20.jsonl`, the 2 benign cases are both `fetch_page` calls, NOT emails (`case_benign_001` and `case_benign_002`). Layer 3 policy checker ALLOWS both unconditionally via the `fetch_` safe prefix. The 50% FPR was **100% caused by Layer 2 (DistilBERT classifier)** false alarms. | 🔴 High | Source code audit of `src/defense/policy.py` and case log audit of `data/seed_cases/sample_20.jsonl`. |

## Recommended Paper Corrections
1. Disclose exact training provenance: 11,023 rows from S-Labs (Kaggle/HF) + 659 rows from deepset.
2. State that the 500-case indirect prompt injection benchmark is an out-of-distribution (OOD) test set.
3. Explicitly report legacy contaminated numbers alongside newly retrained cluster-isolated numbers.
4. Exclude Layer 2 (DistilBERT) from default pipeline configuration; label it as an ablation condition that fails on the tool-output channel.

---

## Benchmark Taxonomy & Cell Coverage Reality (Audit 2026-11-18)

### 1. 160-Cell Matrix Sparsity: Exactly 60 Populated Cells
- The README asserted a 160-cell benchmark (4 injection domains $\times$ 8 attack techniques $\times$ 5 target actions).
- **Reality**: Exactly **60 out of 160 cells (37.5%)** are populated. **100 cells (62.5%) are completely empty**.
- The 450 attacks are generated from only **16 distinct template scenarios** repeated across locations.

| Technique | Populated Cells | Attack Cases | Target Actions Tested | Channels Tested |
| :--- | :---: | :---: | :--- | :--- |
| `context_manipulation` | 12 | 84 | `delete_event`, `delete_file`, `transfer` (3) | `api_response`, `email`, `retrieved_document`, `web_page` (4) |
| `encoded_payload` | 4 | 28 | `send_email` (1) | `api_response`, `email`, `retrieved_document`, `web_page` (4) |
| `hidden_text` | 8 | 57 | `send_email`, `transfer` (2) | `api_response`, `email`, `retrieved_document`, `web_page` (4) |
| `multilingual` | 12 | 112 | `delete_event`, `send_email`, `transfer` (3) | `api_response`, `email`, `retrieved_document`, `web_page` (4) |
| `payload_splitting` | 4 | 28 | `delete_file` (1) | `api_response`, `email`, `retrieved_document`, `web_page` (4) |
| `role_play` | 12 | 85 | `delete_event`, `send_email`, `transfer` (3) | `api_response`, `email`, `retrieved_document`, `web_page` (4) |
| `tool_call_forging` | 8 | 56 | `delete_file`, `send_email` (2) | `api_response`, `email`, `retrieved_document`, `web_page` (4) |
| **Total** | **60 / 160** | **450** | **4 unique actions** | **4 unique channels** |

**Why `multilingual` has 12 cells (not 16)**:
`multilingual` tests 3 target actions (`delete_event`, `send_email`, `transfer`) across 4 channels ($3 \times 4 = 12$ cells). It does not test `delete_file`. A full 16 cells would have required 4 actions $\times$ 4 channels.

---

## Definitive Forensic Provenance of '2/50' Benign False Alarm String

Forensic git inspection (`git show 9ba9d7e207ec88fa55c82c143d9a04cfa209e742`) confirms that the string `'2/50'` was introduced during this evaluation audit session (Wed Nov 18 12:49:54 2026 +0530) by commit `9ba9d7e` ('audit(phase-1d): reconcile benchmark evaluation integrity and capture policy before-baseline').

**Exact Commit Diff & Log Evidence**:
```diff
- | **DistilBERT (Fine-Tuned)** | Domain-Calibrated $	au^*_{{bench}} = ...$ | **4.0%** (2/50) |
- | **BENIGN CONTROLS** | **50** | **FPR: 4.0%** (2/50) |
```
**Root Cause**: In `scripts/reconcile_benchmark.py`, the benign table template hardcoded the literal string `'FPR: 4.0% (2/50)'` from an earlier preliminary sweep at $	au=0.9916$ rather than interpolating the dynamic variable for $	au^*_{bench}=0.9772$ (which produced 11/50). The commit message stated:
> *'In the reconciled markdown generator (reconcile_benchmark.py), the benign row string at best_tau was hardcoded as "FPR: 4.0% (2/50)" from an earlier sweep at tau=0.9916 rather than dynamically populated with the actual count at tau=0.9772.'*

The string `'2/50'` was an un-interpolated template bug, not a distinct scoring run. The single source of truth yields exactly **22/50** at $	au^*=0.4700$ and **11/50** at $	au=0.9772$ on offline commands, and **50/50** on real sandbox tool outputs.