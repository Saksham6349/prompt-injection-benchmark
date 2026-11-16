# Dataset Card: Indirect Prompt-Injection Agent Benchmark (PI-Agent-500)

## 1. Dataset Description
- **Benchmark Suite (PI-Agent-500):** 500 executable cases testing tool-using agents against indirect injection attacks in an isolated SQLite sandbox with state-based assertions.
  - **Public Split:** 401 cases (80.2%)
  - **Private Held-Out Split:** 99 cases (19.8%)
  - **Benign Controls:** 50 cases (10.0%)
- **Real-World Training Corpus (`data/real_classifier_training_data.csv`):** 11,683 verified real prompt-injection and benign samples aggregated and deduplicated from:
  - `deepset/prompt-injections` (15,500+ downloads on Hugging Face Hub)
  - `S-Labs/prompt-injection-dataset` (Kaggle / Hugging Face security benchmark)
  - Label breakdown: 5,018 real adversarial attacks (1) and 6,665 benign instructions (0).

## 2. Taxonomy Coverage
The dataset exercises tool-using agents against indirect injections across 4 goals, 5 channels, and 8 techniques:
- **Goals:** Data Exfiltration, Unauthorized Action, Goal Hijack, Denial of Service
- **Channels:** Web Page, Email, Retrieved Document, API Response, PDF
- **Techniques:** Hidden Text, Role Play, Tool-Call Forging, Multi-Step Chaining, Encoded Payloads (Base64/Hex), Multilingual (Hindi/Hinglish), Context Manipulation, Payload Splitting

## 3. Ground Truth & Evaluation Mechanism
Unlike naive text-matching benchmarks, each case contains:
- `user_task`: Benign instruction for the agent.
- `expected_benign_calls`: Ground truth tool invocations needed to satisfy utility.
- `expected_malicious_calls`: Specific malicious function-calling parameters that confirm a successful hijack.
- Outcome verification evaluates changes in mock state (e.g. unauthorized bank debits or exfiltration emails recorded in the sandbox SQLite database).

## 4. Ethical & Responsible Use
- All tool calls execute exclusively within an isolated in-memory or local mock sandbox. No external networks, private accounts, or live systems are targeted.
- Payloads represent synthetic adversarial patterns crafted purely to assess agent robustness.
