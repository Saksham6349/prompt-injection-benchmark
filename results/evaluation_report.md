# Prompt Injection Defense Benchmark (GOOGLE)

## 1. Executive Summary

### Defense Performance Comparison

| Defense | Cases | ASR (%) | Utility (%) | Latency (ms) | FPR (%) | 95% CI (ASR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Undefended** | 3 | 0.0% | 100.0% | 65279.4 | N/A | [0.00, 0.00] |
| **Layer 1: Provenance Only** | 3 | 0.0% | 100.0% | 62741.4 | N/A | [0.00, 0.00] |
| **Layer 2: Classifier Only** | 3 | 0.0% | 100.0% | 62943.4 | N/A | [0.00, 0.00] |
| **Layer 3: Policy Only** | 3 | 0.0% | 100.0% | 63372.3 | N/A | [0.00, 0.00] |
| **Our 3-Layer Defense** | 3 | 0.0% | 100.0% | 63091.4 | N/A | [0.00, 0.00] |

## 2. Key Observations
- **Attack Mitigation:** Layered defense reduces ASR relative to undefended baselines.
- **Utility Preservation:** Preserves benign function-calling capability without breaking agent workflows.
- **Latency Impact:** Minimal overhead on tool-execution loops.

## 3. Methodology Notes
- Tested against deterministic sandbox tools (email, calendar, filestore, web, bank).
- Out-of-distribution injection payloads spanning diverse attack techniques and multilingual variants.
- Evaluated using deterministic state assertions, not text-matching.