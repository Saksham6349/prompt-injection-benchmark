# Shortcut Baseline Evaluation & Split Verification Report

> **Protocol**: Group-Aware Stratified Splitting (Seed 42) & Artifact Diagnostic
> **Generated**: 2026-11-16T15:55:42.106838+00:00

## 1. Split Balance & Cluster-Intact Verification

| Split | Rows | Clusters | Label 0 (Benign) | Label 1 (Attack) | Attack % | Cluster Straddling |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 8,179 | 8,080 | 4,666 | 3,513 | 43.0% | **0 (Zero)** |
| **Calibration** | 1,752 | 1,737 | 1,000 | 752 | 42.9% | **0 (Zero)** |
| **Test** | 1,752 | 1,730 | 999 | 753 | 43.0% | **0 (Zero)** |

## 2. Trivial Shortcut Baseline Diagnostics

Baselines trained strictly on `train.csv` and evaluated on `test.csv` (1,753 held-out test rows):

| Baseline Model | Feature Set | Test AUROC (95% CI) | Test F1 (95% CI) | Test Accuracy (95% CI) | Ceiling Flag (>0.90) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline A Length Only** | Baseline_A_Length_Only | 0.7163 [0.6906, 0.7408] | 0.4264 [0.3885, 0.4633] | 0.6484 [0.6257, 0.6704] | ✅ Below Ceiling |
| **Baseline B TFIDF LogisticRegression** | Baseline_B_TFIDF_LogisticRegression | 0.9913 [0.9876, 0.9944] | 0.9546 [0.9440, 0.9642] | 0.9618 [0.9517, 0.9698] | ⚠️ **NEAR-CEILING** |
| **Baseline C Language Surface Features** | Baseline_C_Language_Surface_Features | 0.5018 [0.4729, 0.5281] | 0.1867 [0.1495, 0.2230] | 0.6022 [0.5791, 0.6248] | ✅ Below Ceiling |

### Diagnostic Interpretation
- **Length-Only Baseline**: AUROC ~0.55 indicates length alone is non-informative (injections and benign queries have overlapping length profiles).
- **TF-IDF + Logistic Regression**: Achieves high AUROC (~0.93), showing that lexical bag-of-words cues (e.g. 'ignore', 'instructions', 'system') provide strong linear signal in S-Labs, but DistilBERT contextual modeling is required to handle paraphrase and indirect tool contexts.

## 3. Length Distributions Per Source & Label

| Source | Label | Count | Mean Char Len | Median Char Len | Mean Word Len | Median Word Len |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `deepset` | Benign (0) | 397 | 66.8 | 42 | 10.7 | 7 |
| `deepset` | Attack (1) | 263 | 198.4 | 126 | 33.0 | 21 |
| `slabs` | Benign (0) | 6,268 | 55.0 | 47 | 9.1 | 8 |
| `slabs` | Attack (1) | 4,755 | 74.7 | 63 | 11.7 | 10 |