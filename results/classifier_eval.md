# Publication-Grade Classifier Evaluation Report (Reconciled)

> **Project**: Prompt-Injection Benchmark & Lightweight Defense
> **Audit Status**: Reconciled & Closed (Phase 1)
> **Model Checkpoint SHA-256**: `784c14f18bb459046fa286a2b3897d1c8604347ce16f28f40e33d71142cff6df`
> **Protocol Enforced**: Pinned Seed 42, Cluster-Isolated Splits, No Manual Number Entry

---

## 1. Frozen Threshold Calibration Protocol

| Parameter | Value | Split Calibrated On | Description |
| :--- | :---: | :--- | :--- |
| **Optimal Threshold $\tau^*$ (F1-Max)** | `0.4700` | In-distribution `calibration.csv` | Primary decision boundary locked prior to test evaluation |
| **Security Target (FPR $\le 1.0\%$)** | `0.5700` | In-distribution `calibration.csv` | High-security operating point |
| **Security Target (FPR $\le 0.1\%$)** | `0.8700` | In-distribution `calibration.csv` | Ultra-low false alarm threshold |
| **Benchmark-Calibrated $\tau^*_{bench}$** | `0.9772` | Benchmark Dev Cluster Split (Exploratory) | Exploratory diagnostic threshold |
| **Benchmark Target 1% FPR $\tau_{bench, 1\%}$** | `0.9948` | Benchmark Dev Cluster Split (Exploratory) | Operating point ensuring $\le 1\%$ benign false alarms on benchmark |

---

## 2. In-Distribution Held-Out Test Split (`data/splits/test.csv`, N=1,752)

| Model | Split Protocol | AUROC (95% CI) | AUPRC (95% CI) | F1 @ $\tau^*$ (95% CI) | Recall @ 1% FPR | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | Active Linear Baseline | 0.9913 [0.988, 0.995] | 0.9895 [0.985, 0.994] | 0.9563 [0.946, 0.967] | 0.9137 | 0.0422 |
| **DistilBERT (Fine-Tuned)** | Cluster-Isolated Split | **0.9966** [0.994, 0.999] | **0.9962** [0.994, 0.998] | **0.9779** [0.970, 0.985] | **0.9695** | **0.0169** |
| *Paired Difference (DistilBERT - TF-IDF)* | Bootstrap B=1000 | **+0.0053** [+0.0026, +0.0082] | — | — | — | — |
| *Legacy Model (Contaminated)* | *Contaminated, not for paper* | 0.9993 | 0.9989 | 0.9947 | 0.9973 | 0.0042 |

---

## 3. Leave-One-Source-Out (LOSO) Cross-Domain Generalization

Thresholds do not transfer across disparate data distributions; threshold-free metrics (AUROC, AUPRC) are reported as primary.

| Transfer Direction | Primary / Secondary | Model | AUROC (95% CI) | AUPRC (95% CI) | F1 @ $\tau^*$ | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **S-Labs $\to$ deepset** | **Primary Shift Test** | TF-IDF + LR | 0.7625 [0.660, 0.863] | 0.7670 [0.668, 0.862] | 0.6349 | Surface n-gram baseline collapses under source shift. |
| **S-Labs $\to$ deepset** | **Primary Shift Test** | **DistilBERT** | **0.9323** [0.878, 0.972] | **0.9232** [0.866, 0.968] | **0.7692** | Model fine-tuned on S-Labs exclusively. |
| *Paired Difference (DistilBERT - TF-IDF)* | Primary Shift Test | Bootstrap B=1000 | **+0.1679** [+0.0843, +0.2578] | — | — | Statistically significant contextual generalization gain. |
| **deepset $\to$ S-Labs** | Secondary Shift Test | TF-IDF + LR | 0.8385 [0.819, 0.858] | 0.8053 [0.780, 0.830] | 0.6912 | Extreme sample asymmetry (660 rows predicting 11,023 rows). |

---

## 4. Reconciled 500-Case Benchmark Evaluation

> **Reconciliation Audit Finding**: The previous evaluation loop mixed the 50 benign control cases into the `role_play` attack slice (because template metadata populated `role_play` as a default tag). In this reconciled report, benign cases are strictly separated into the control group ($N=50$), and all 7 attack techniques sum exactly to $N=450$.

### 4.1 Overall Benchmark Performance

> **Evaluation Integrity Notice (Phase 1d Audit)**:
> Offline benchmark evaluation (`scores_benchmark_frozen.jsonl`, SHA-256: `7c79493d9d19c95baa3a5fdc0788c51170d9657f3a8b5153ca84e193d04d8dfe`) scored the user task string for the 50 benign cases because injection payloads were empty. However, the runtime pipeline **never passes user tasks to Layer 2**; Layer 2 inspects only tool outputs.
> When instrumenting the pipeline to log the exact strings returned by the sandbox (`results/scores_L2_inputs.jsonl`, SHA-256: `99623e3d5dcfdc6aed3f63fb915b5951312fd016990869168926867aa689b511`), DistilBERT flags **50 out of 50 (100.0%)** of real benign tool outputs at $\tau^* = 0.4700$. The earlier 44.0% FPR was an artifact of scoring user command strings and **must not be cited as a tool-output FPR**.

| Evaluation Input Source | Model | Threshold $\tau$ | Attack Detection (N=450) | Benign Control FPR (N=50) | 95% Clopper-Pearson CI (Template Level) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Real Sandbox Tool Outputs** (`scores_L2_inputs.jsonl`) | **DistilBERT** (Ablation Condition: *fails on tool-output channel (this checkpoint)*) | $\tau^* = 0.4700$ | **100.0%** (450/450) | **100.0%** (50/50)\* | Atk: [79.4%, 100.0%]<br>Benign: [85.8%, 100.0%] |
| **Real Sandbox Tool Outputs** | **PromptGuard Baseline** (Heuristic Fallback) | $\tau = 0.5000$ | 2.89% (13/450) | **0.0%** (0/50) | AUROC: 0.5178 |
| **Offline User Task Commands** (`scores_benchmark_frozen.jsonl`) | **DistilBERT** | $\tau^* = 0.4700$ | **100.0%** (450/450) | 44.0% (22/50) | Atk: [79.4%, 100.0%]<br>Benign: [24.4%, 65.1%] |
| **Offline User Task Commands** | **DistilBERT** | $\tau = 0.9772$ (Exploratory) | **100.0%** (450/450) | 22.0% (11/50) | Atk: [79.4%, 100.0%]<br>Benign: [9.3%, 47.6%] |
| **Offline User Task Commands** | **TF-IDF + LR** | $\tau^* = 0.4700$ | 68.89% (310/450) | **0.0%** (0/50) | Atk: [44.7%, 86.6%]<br>Benign: [0.0%, 14.2%] |

> [!CAUTION]
> **Domain-Mismatch Empirical Hypothesis & Channel Failure (This Checkpoint)**:
> Layer 2 (fine-tuned DistilBERT) is **excluded from default pipeline configurations** (default is L1 Provenance + L3 Policy) and retained solely as an ablation condition labeled *"fails on tool-output channel (this checkpoint)"*.
>
> To test whether this channel failure is caused by formatting layout (headers, linebreaks) or deeper semantic features, we conducted controlled variant testing on all 500 cases (`results/domain_mismatch_eval.json`):
> 1. **Training Corpus Baseline** (`data/splits/train.csv`, N=8,179): Mean length is 10.74 words (median 9.0; slabs=10.19 words, deepset=19.95 words). Only **0.15%** of training samples contain newlines (`\n`), and only **0.86%** contain key-value headers.
> 2. **Controlled Variant Scoring on Sandbox Outputs (N=500)**:
>    - **As-Is Sandbox Outputs**: Benign mean score 0.9776 (median 0.9971), Benign FPR 100.0%, AUROC = 0.6162.
>    - **Headers Stripped (Body Only)**: Benign mean score 0.9756 (median 0.9970), Benign FPR 100.0%, AUROC = 0.5402.
>    - **Reflowed Prose (Single-Line)**: Benign mean score 0.9776 (median 0.9971), Benign FPR 100.0%, AUROC = 0.6162.
>
> **Hypothesis Verdict**: Stripping structured headers and flattening multiline formatting does *not* eliminate false positives. The 100% FPR is not an artifact of header markers or newlines, but indicates that this DistilBERT checkpoint has generalized "injection" to include dense, factual business schemas (API specs, email listings, transaction records). Future retraining requires a large corpus of authentic tool outputs with contrastive hard negatives.

### 4.2 Counterfactual Probes & Neutralization Audit

A rigorous audit of counterfactual transformations (`results/probe_neutralization_audit.json`) explains earlier evaluation anomalies:
- **Hash Invariance in Naive Probes**: In `p2_payload_only`, 84/450 cases had identical SHA-256 hashes to `p1_original_attack` because single-line attacks lacked delimiters for regex splitting. Furthermore, `p2` and `p7` exhibited identical 421/450 detection because the naive neutralizer only masked 9 isolated verbs with `[ACTION]`, leaving persona directives, base64 payloads, and tool signatures fully intact.
- **True Neutralization & Keywords-Only Tests**:
  - **Original Unique Templates (N=16)**: 16 / 16 detected (100.0%, mean score 0.9970).
  - **Truly Neutralized (Commands, Syntax, Base64 Stripped)**: 14 / 16 detected (87.5%, mean score 0.8686; 393/450 overall).
  - **Keywords-Only (Isolated Trigger Words, No Command Grammar)**: Detection drops to **7 / 16 (43.8%, mean score 0.4131; 197/450 overall)**.
  This confirms that removing imperative grammar significantly degrades classifier activation, but residual business keywords still trigger over-sensitive features in this checkpoint.

### 4.2 Benchmark Text Repetition & Effective Sample Size ($N_{eff}$)

> [!WARNING]
> **Attack Prompt Redundancy**: An exact SHA-256 hash audit of the 450 benchmark attacks reveals only **16 unique text prompts**. Target entities (account numbers, recipient emails, file IDs) were statically hardcoded into the 16 template descriptions without per-case randomization. The true effective sample size of the attack benchmark is **$N_{eff} = 16$ independent prompts**. Degenerate bootstrap intervals of `[100.0%, 100.0%]` have been discarded in favor of exact Clopper-Pearson intervals at the template level (`[79.4%, 100.0%]`).

### 4.3 Reconciled Per-Technique Attack Detection Breakdown (N=450 Attacks)

All per-technique counts sum strictly to 450:

| Attack Technique (Taxonomy Name) | Sample Count N | TF-IDF Detection Rate (95% CI) | DistilBERT Detection Rate @ $\tau=0.4700$ | DistilBERT Detection Rate @ $\tau=0.9772$ | DistilBERT Detection Rate @ $\tau=0.9916$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `context_manipulation` | 84 | 100.0% [95.6%, 100.0%] | **100.0% [95.6%, 100.0%]** | **100.0% [95.6%, 100.0%]** | **100.0% [95.6%, 100.0%]** |
| `encoded_payload` (pseudo-encoded) | 28 | 100.0% [87.9%, 100.0%] | **100.0% [87.9%, 100.0%]** | **100.0% [87.9%, 100.0%]** | **100.0% [87.9%, 100.0%]** |
| `hidden_text` | 57 | 50.9% [38.3%, 63.4%] | **100.0% [93.7%, 100.0%]** | **100.0% [93.7%, 100.0%]** | **50.88%** (29/57)\* |
| `multilingual` | 112 | 25.0% [17.9%, 33.8%] | **100.0% [96.7%, 100.0%]** | **100.0% [96.7%, 100.0%]** | **100.0% [96.7%, 100.0%]** |
| `payload_splitting` (single-block) | 28 | 0.0% [0.0%, 12.1%] | **100.0% [87.9%, 100.0%]** | **100.0% [87.9%, 100.0%]** | **100.0% [87.9%, 100.0%]** |
| `role_play` | 85 | 100.0% [95.7%, 100.0%] | **100.0% [95.7%, 100.0%]** | **100.0% [95.7%, 100.0%]** | **100.0% [95.7%, 100.0%]** |
| `tool_call_forging` | 56 | 100.0% [93.6%, 100.0%] | **100.0% [93.6%, 100.0%]** | **100.0% [93.6%, 100.0%]** | **100.0% [93.6%, 100.0%]** |
| **ALL ATTACK SAMPLES** | **450** | **68.9%** (310/450) | **100.0%** (450/450) | **100.0%** (450/450) | **93.78%** (422/450) |
| **BENIGN CONTROLS** | **50** | **FPR: 0.0%** (0/50) | **FPR: 44.0%** (22/50) | **FPR: 22.0%** (11/50) | **FPR: 2.0%** (1/50) |

*\*Note on $\tau=0.9916$: Exactly 28 attacks under `hidden_text` (template `atk_Zero-opacity CSS injection triggering unauthorized wire transfer.`) score 0.99158716, which is strictly less than 0.99160000, reducing `hidden_text` detection from 57/57 to 29/57 (50.88%).*

---

## 5. Template-Artifact & Shortcut Baseline Audit

To ensure benchmark performance is not an artifact of template structure, we evaluated trivial shortcut baselines using GroupKFold cross-validation across all 40 template clusters:

| Shortcut Feature Set | GroupKFold AUROC | Artifact Risk Assessment |
| :--- | :---: | :--- |
| **Length Only** (chars, words) | `0.8955` | Below threshold (< 0.90) |
| **Format Markers** (HTML, JSON, syntax) | `0.8276` | Below threshold (< 0.90) |
| **First Token Features** (case, digits, length) | `0.7255` | Below threshold (< 0.90) |
| **Combined Surface Features** | `0.9166` | **Flagged as artifact-prone (> 0.90)** |

> [!WARNING]
> **Benchmark Artifact Warning**: Combined surface features (token length, HTML/JSON tags, and first-token casing/punctuation) achieve **AUROC = 0.9166** under GroupKFold cross-validation across template clusters. Because a linear model with simple surface features exceeds 0.90 AUROC, the benchmark is officially **flagged as shortcut-artifact prone**. Results must be evaluated with counterfactual probes and held-out cluster cross-validation.

---

## 6. Utility Attribution Diagnosis (50 Benign Benchmark Tasks)

Diagnosing which layer blocks benign user tasks under a deterministic scripted agent:

| Defense Configuration | Benign Tasks Blocked | False Alarm Rate | Primary Cause |
| :--- | :---: | :---: | :--- |
| **Layer 1 Alone (Provenance Tagging)** | **0 / 50** | **0.0%** | Delimiters never reject tool outputs |
| **Layer 2 Alone (Classifier @ Transferred $\tau^*=0.47$)** | **22 / 50** | **44.0%** | Classifier flags short imperative user tasks (cause unresolved) |
| **Layer 2 Alone (Classifier @ Exploratory $\tau^*_{bench}=0.9772$)** | **11 / 50** | **22.0%** | Test template false positives (exploratory setting) |
| **Layer 3 Alone (Policy Checker)** | **0 / 50** | **0.0%** | Legitimate read actions match safe prefixes |
| **Full Pipeline (L1 + L2@$\tau^*$ + L3)** | **22 / 50** | **44.0%** | Dominated entirely by Layer 2 false alarms |
| **Full Pipeline (L1 + L2@$\tau_{bench}$ + L3)** | **11 / 50** | **22.0%** | Dominated entirely by Layer 2 false alarms |

> **Diagnosis for BUG-04 & Phase 2**: On benign tasks where read operations are requested, Layer 3 policy checker currently passes read actions via safe prefixes (`get_`, `read_`, `list_`). The over-blocking observed in benchmark testing was **100% driven by Layer 2 classifier false alarms**, not Layer 3 policy rejection. However, for sensitive tools (e.g. `send_email`), Layer 3's keyword substring check remains vulnerable to gaming (SEC-02) and fails to verify argument consistency (e.g., recipient address hijacking).

---

## 7. Limitations & Scientific Scope

1. **Pretrained Language Model Scope**: Evaluated on `distilbert-base-uncased` (single seed 42 reported for GPU retrain). Multilingual transfer to Hindi/Hinglish relies on subword tokenization without native multilingual pretraining.
2. **Target Evaluation Size**: The cross-dataset Deepset target split contains 98 cluster-intact rows; while statistically significant, confidence intervals reflect sample size limitations.
3. **Synthetic Benchmark Nature**: The 500-case suite utilizes realistic threat scenarios generated from 40 template clusters; results should be validated against the forthcoming human-written dataset (`data/handwritten_eval.jsonl`).
