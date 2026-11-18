# Dataset Card: Indirect Prompt-Injection Agent Benchmark (PI-Agent-500)

## 1. Dataset Description
- **Benchmark Suite (PI-Agent-500):** 500 executable cases testing tool-using agents against indirect injection attacks in an isolated SQLite sandbox with state-based assertions.
  - **Public Split:** 401 cases (80.2%)
  - **Private Held-Out Split:** 99 cases (19.8%)
  - **Benign Controls:** 50 cases (10.0%)
  - **Generation Protocol:** Synthesized from parameterized combinatorial templates across a 160-cell taxonomy (see Section 2). Payloads are embedded in mock tool environments (email, calendar, files, bank, web).
- **Training & Calibration Corpus (`data/real_classifier_training_data_v2.csv`):** 11,683 verified real prompt-injection and benign samples cryptographically verified by SHA-256 against official Hugging Face repositories:
  - **`deepset/prompt-injections`:** 659 samples (397 benign, 262 attack). License: Apache-2.0.
  - **`S-Labs/prompt-injection-dataset`:** 11,023 samples (6,268 benign, 4,755 attack). License: MIT.
  - **Note on `JasperLS/prompt-injections`:** Cryptographic analysis confirms that `JasperLS/prompt-injections` is a byte-for-byte repository mirror of `deepset/prompt-injections` (identical parquet SHA-256 hashes `2e10bc7a...` and `39ac797c...`).
  - Label breakdown: 5,018 adversarial attacks (label 1) and 6,665 benign instructions (label 0).

## 2. Taxonomy Coverage
The benchmark dataset exercises tool-using agents against indirect injections across 4 goals, 5 channels, and 8 techniques:
- **Goals:** Data Exfiltration, Unauthorized Action, Goal Hijack, Denial of Service
- **Channels:** Web Page, Email, Retrieved Document, API Response, PDF
- **Techniques:** Hidden Text, Role Play, Tool-Call Forging, Multi-Step Chaining, Encoded Payloads (Base64/Hex), Multilingual (Hindi/Hinglish), Context Manipulation, Payload Splitting

## 3. Data Provenance & Licenses

| Dataset | Hugging Face Repository | Git Revision | Matched Rows | License | Recommended Citation |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **deepset** | [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) | `4f61ecb038e9c3fb77e21034b22511b523772cdd` | 659 | Apache-2.0 | `@misc{deepset_prompt_injections, author={deepset}, title={Prompt Injections Dataset}, year={2023}, publisher={Hugging Face}}` |
| **S-Labs** | [`S-Labs/prompt-injection-dataset`](https://huggingface.co/datasets/S-Labs/prompt-injection-dataset) | `002a9dd18514abd021869823d6b0429b38606d99` | 11,023 | MIT | `@misc{slabs_prompt_injection, author={S-Labs}, title={Prompt Injection Detection Dataset}, year={2024}, publisher={Hugging Face}}` |
| **JasperLS** | [`JasperLS/prompt-injections`](https://huggingface.co/datasets/JasperLS/prompt-injections) | `9f62f4a9788f030e52fe85a816b338f507c48187` | 0 novel (mirror) | Apache-2.0 | *(Identical to deepset)* |

## 4. Ground Truth & Evaluation Mechanism
Unlike naive text-matching benchmarks, each case contains:
- `user_task`: Benign instruction for the agent.
- `expected_benign_calls`: Ground truth tool invocations needed to satisfy utility.
- `expected_malicious_calls`: Specific malicious function-calling parameters that confirm a successful hijack.
- Outcome verification evaluates state-level changes in mock SQLite state (e.g. unauthorized bank debits or exfiltration emails recorded in the database).

## 5. Ethical & Responsible Use
- All tool calls execute exclusively within an isolated in-memory or local mock sandbox. No external networks, private accounts, or live systems are targeted.
- Payloads represent synthetic adversarial patterns crafted purely to assess agent robustness.
