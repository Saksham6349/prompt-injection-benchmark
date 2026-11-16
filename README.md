# Prompt-Injection Benchmark & Lightweight Defense for Tool-Using LLM Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch: 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace: Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/)
[![Gemini: 2.5--Flash](https://img.shields.io/badge/LLM-Gemini--2.5--Flash-4285F4.svg)](https://ai.google.dev/)
[![Status: Review--I--Ready](https://img.shields.io/badge/VIT%20Capstone-Review--I%20Ready-success.svg)]()

A reproducible evaluation benchmark and modular, low-latency **3-layer defense framework** designed to protect autonomous, tool-augmented Large Language Model (LLM) agents against **indirect prompt injection** attacks.

Developed for the B.Tech Final Year Capstone Project (**BCSE498J / CBS1904**), School of Computer Science and Engineering (SCOPE), VIT.

---

## 📌 Executive Summary & Problem Formulation

When autonomous LLM agents consume untrusted external data (such as emails, fetched web pages, shared document stores, or third-party API payloads), malicious actors can embed adversarial instructions designed to hijack the model's function-calling privileges. This vulnerability is known as **Indirect Prompt Injection** (OWASP LLM01 / MITRE ATLAS AML.T0051).

Existing defense paradigms suffer from severe trade-offs:
- **LLM-as-a-Judge**: Extremely high latency (+1,200 ms to 3,000 ms) and prohibitive inference token costs per tool invocation.
- **Static Keyword / Regex Denylists**: Brittle and easily circumvented by character substitutions, paraphrasing, base64 encodings, or cross-lingual pivoting (e.g., Hindi / Hinglish).

### Project Solution
This project establishes:
1. **A Systematic 160-Cell Attack Taxonomy & 500-Case Executable Benchmark**: Covers 4 attack goals, 5 intake channels, 8 injection techniques, and 3 languages (English, Hindi, Hinglish), evaluated via **state-level outcome verification** inside an isolated deterministic sandbox.
2. **A Low-Latency 3-Layer Defense Pipeline**:
   - **Layer 1 (Provenance Tagging)**: Enforces explicit data/instruction boundary marking (`<untrusted_data>`).
   - **Layer 2 (Lightweight Sequence Classifier)**: A fine-tuned `microsoft/deberta-v3-small` transformer classifier trained on 11,683 real-world samples for sub-millisecond anomaly detection.
   - **Layer 3 (Tool Policy Enforcement)**: Dynamic intent authorization checks verifying user-approved operations prior to sensitive tool execution (`bank_transfer`, `send_email`, `delete_file`).
3. **Frontier Multi-Provider Agent Harness**: Live function-calling support for Google Gemini (`gemini-2.5-flash`), OpenAI (`gpt-4o-mini`), Anthropic (`claude-3-haiku`), local Ollama models (`llama3.2`), and an offline mock provider with built-in response caching.

---

## 🏛 System Architecture

```
User Task (e.g., "Summarize unread client emails")
                        │
                        ▼
            ┌───────────────────────┐
            │   LLM Agent Harness   │ ◄─── System Prompt (User Context & Permissions)
            └───────────┬───────────┘
                        │
                  Proposed Tool Call
                        │
                        ▼
            ┌───────────────────────┐
            │ Layer 3: Policy Check │ ────► [BLOCKED: Unauthorized Sensitive Action]
            └───────────┬───────────┘
                        │ (Allowed: e.g. read_email)
                        ▼
            ┌───────────────────────┐
            │ Isolated Mock Sandbox │ (SQLite-backed Email, Calendar, Files, Bank, Web)
            └───────────┬───────────┘
                        │
                  Raw Tool Output (Contains Adversarial Injection Payload)
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│  Layer 2: Fast Classifier│    │ Layer 1: Provenance Tag  │
│ (DeBERTa-v3 / Heuristic) │    │ (Boundary Delimiters)    │
└───────────────┬──────────┘    └───────────┬──────────────┘
        (Flag Anomaly)                  (Structured XML Enclosure)
        └───────────────┬───────────────┘
                        ▼
         Sanitized / Tagged Tool Result
                        │
                        ▼
              Agent Loop Continues
                        │
                        ▼
            ┌───────────────────────┐
            │ State Outcome Checker │ ────► Pass/Fail based on Database Mutation
            └───────────────────────┘
```

---

## 🔬 Benchmark Taxonomy & Dataset Specification

The benchmark systematically covers **$4 \times 5 \times 8 = 160$** structural attack cells across 3 linguistic variants:

| Dimension | Categories |
| :--- | :--- |
| **Attack Goals (4)** | `data_exfiltration`, `unauthorized_action`, `goal_hijack`, `denial_of_service` |
| **Intake Channels (5)** | `email`, `web_page`, `retrieved_document`, `api_response`, `pdf` |
| **Attack Techniques (8)** | `hidden_text`, `role_play`, `tool_call_forging`, `multi_step`, `encoded_payload`, `multilingual`, `context_manipulation`, `payload_splitting` |
| **Languages (3)** | English (`en`), Hindi (`hi`), Hinglish (`hi-en`) |

### Split Integrity & Benign Controls
- **Public Split (401 cases)**: Open for defense optimization, validation, and training.
- **Private Held-Out Split (99 cases)**: Strictly preserved for unbiased evaluation across unseen attacks.
- **Benign Controls (50 cases)**: Non-adversarial tasks with clean tool outputs to rigorously measure **False Positive Rate (FPR)** and task completion degradation.
- **Outcome Verification**: Never relies on fuzzy text regex on agent prose; verifies actual SQLite state changes (e.g., rows inserted in `bank_transactions`, files deleted in `filestore`, unauthorized outgoing emails in `email_outbox`).

---

## 🛡 Layer 2: Real-World Dataset & Model Fine-Tuning

Layer 2 uses a fine-tuned **`microsoft/deberta-v3-small`** transformer classifier.

- **Dataset Size**: **11,683 real-world samples** compiled from Hugging Face Hub benchmarks (`deepset/prompt-injections` and `S-Labs/prompt-injection-dataset`):
  - 5,018 real prompt injection payloads
  - 6,665 benign instructions and clean tool outputs
  - Stored at `data/real_classifier_training_data.csv`
- **Training Setup**:
  - Script: `scripts/train_classifier.py`
  - Colab Notebook: `notebooks/train_classifier_colab.ipynb` (Free T4 GPU, ~8–10 mins)
  - Numeric stability: Trained with **FP32 (`fp16=False`)** to prevent DeBERTa-v3 gradient overflow.
  - Automatic fallback: If weights are absent, a high-recall heuristic scanner seamlessly activates.

---

## 📁 Repository Structure

```
├── config/
│   └── default.yaml                # Global evaluation, rate-limiting & model configs
├── data/
│   ├── benchmark_cases.jsonl       # Full 500 generated benchmark cases (public + private split)
│   ├── real_classifier_training_data.csv # 11,683 HuggingFace/Kaggle injection training samples
│   ├── response_cache.db           # SQLite persistent cache for LLM API responses
│   └── seed_cases/
│       └── sample_20.jsonl         # 20-case rapid validation suite
├── docs/
│   ├── capstone_alignment.md       # University milestones, guidelines & submission tracker
│   └── data_card.md                # Dataset ethics, distribution & provenance card
├── models/
│   └── injection_classifier/       # Fine-tuned DeBERTa-v3 model weights & tokenizer
├── notebooks/
│   └── train_classifier_colab.ipynb # Free T4 GPU training notebook for Google Colab
├── results/
│   └── evaluation_report.md        # Generated evaluation report with bootstrap CIs
├── scripts/
│   ├── generate_cases.py           # Benchmark case generator CLI
│   ├── run_benchmark.py            # Single-configuration benchmark runner CLI
│   ├── train_classifier.py         # Local/GPU sequence classifier training script
│   └── evaluate.py                 # Multi-defense comparative matrix evaluator
├── src/
│   ├── agent/                      # Multi-provider agent harness & rate limiter
│   │   ├── cache.py                # Response caching engine
│   │   ├── harness.py              # Tool-calling agent loop
│   │   ├── prompts.py              # Context-seeded system prompts
│   │   └── providers/              # Google Gemini, OpenAI, Anthropic, Ollama, Mock
│   ├── baselines/                  # Spotlighting, Sandwich, Perplexity, PromptGuard, LLM-Judge
│   ├── benchmark/                  # Benchmark generator and SQLite state-based outcome checker
│   ├── defense/                    # 3-Layer defense implementation
│   │   ├── classifier.py           # Layer 2: DeBERTa-v3 & heuristic scanner
│   │   ├── pipeline.py             # Integrated Layer 1-3 defense pipeline
│   │   ├── policy.py               # Layer 3: Sensitive tool policy & authorization engine
│   │   └── provenance.py           # Layer 1: Boundary & provenance tagger
│   ├── evaluation/                 # Metrics calculation (ASR, FPR, Latency) & reporting
│   ├── sandbox/                    # Deterministic mock tools (Email, Bank, Calendar, Files, Web)
│   │   ├── app.py                  # Standalone FastAPI sandbox server
│   │   ├── executor.py             # In-process deterministic tool executor
│   │   ├── state.py                # SQLite state schema and transaction rollbacks
│   │   └── tools/                  # Concrete tool implementations
│   ├── taxonomy/                   # 160-cell taxonomy, templates, and multilingual generator
│   └── types.py                    # Core dataclasses and domain enums
└── tests/                          # Complete unit & integration test suites
    ├── test_benchmark.py           # Taxonomy and case generation tests
    ├── test_defense.py             # Layer 1, 2, 3 and pipeline tests
    └── test_sandbox.py             # Mock tool and state rollback tests
```

---

## ⚡ Quick Start & Usage

### 1. Installation
Clone the repository and install dependencies in Python 3.11+:
```bash
git clone https://github.com/your-username/capstone-prompt-injection.git
cd capstone-prompt-injection
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```bash
# Google Gemini (Default frontier model: gemini-2.5-flash)
GEMINI_API_KEY="your-gemini-api-key"

# Optional Cloud Providers
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Run Unit Tests (100% Passing)
```bash
python -c "
from tests.test_sandbox import *
from tests.test_benchmark import *
from tests.test_defense import *
test_sandbox_state_initialization()
test_email_tool_read_and_send()
test_bank_tool_transfer()
test_sandbox_reset()
test_taxonomy_structure()
test_benchmark_generation_and_distributions()
test_provenance_tagger()
test_heuristic_classifier()
test_policy_checker_blocks_unauthorized_sensitive_tool()
test_defense_pipeline_pass_benign()
print('All 11 unit tests passed successfully!')
"
```

### 4. Train Layer 2 Classifier (DeBERTa-v3-small)
Train on Google Colab GPU using `notebooks/train_classifier_colab.ipynb` or locally:
```bash
python scripts/train_classifier.py \
  --cases data/real_classifier_training_data.csv \
  --output-dir models/injection_classifier \
  --epochs 3 \
  --batch-size 16
```

### 5. Run Live Evaluation Against Frontier Models
Run comparative evaluations across defense layers:
```bash
# Undefended Baseline on Google Gemini 2.5 Flash
python scripts/run_benchmark.py --provider google --cases data/seed_cases/sample_20.jsonl --defense none

# Full 3-Layer Defense on Google Gemini 2.5 Flash
python scripts/run_benchmark.py --provider google --cases data/seed_cases/sample_20.jsonl --defense all

# Full Comparative Matrix Evaluation with Markdown Report
python scripts/evaluate.py --provider google --cases data/seed_cases/sample_20.jsonl --max-cases 10 --output-report results/evaluation_report.md
```

---

## 🎓 VIT Capstone Project Alignment (BCSE498J / CBS1904)

- **Title**: *Prompt-Injection Benchmark and Lightweight Defense for Tool-Using LLM Agents*
- **Team**: Final Year B.Tech Project, Winter Semester 2026–2027
- **Deliverables & Milestones**:
  - [x] **Google Form Submission**: Title, Abstract, Guide details (Deadline: 20 Nov 2026)
  - [x] **Phase 1 Implementation**: Mock Sandbox, Taxonomy, 500-case generator, 3-layer defense, and test suite.
  - [x] **Phase 2 Implementation**: Real-world dataset integration (11,683 samples), DeBERTa-v3 fine-tuning pipeline, and live Gemini function-calling harness.
  - [ ] **Review-I Presentation**: Title defense, architecture diagram, literature review, and Guide Consent Form submission (9–11 Dec 2026).
  - [ ] **Review-II (50–60% Milestone)**: Live comparative benchmarking on frontier models, ablation study, and baseline comparisons (28–29 Jan 2027).

---

## 📜 Ethical Considerations & Safe Disclosure
This project generates synthetic indirect prompt injection payloads strictly within an isolated, mock SQLite sandbox for defensive benchmarking. Payloads are never executed against real production endpoints, live email servers, or monetary payment gateways. For detailed ethical guidelines, see [docs/data_card.md](file:///c:/Users/Saksham/Downloads/Capstone/docs/data_card.md).

## 📄 License
MIT License. Open for educational, academic, and non-commercial research use.
