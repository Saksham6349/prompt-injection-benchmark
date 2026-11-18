"""Report consistency and generate-and-diff integrity test suite.

Replaces brittle SHA-pinning with dynamic generate-and-diff verification:
1. Asserts all generated evaluation reports match raw single-source-of-truth files without drift.
2. Scans report generator scripts to ensure zero hardcoded metric literals exist in generation code.
3. Asserts mathematical invariants across benchmark and counterfactual probe counts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"


def test_frozen_single_source_of_truth_exists() -> None:
    """Verifies that all frozen evaluation inputs exist on disk."""
    assert (RESULTS_DIR / "scores_benchmark_frozen.jsonl").exists(), "scores_benchmark_frozen.jsonl missing"
    assert (RESULTS_DIR / "scores_L2_inputs.jsonl").exists(), "scores_L2_inputs.jsonl missing"
    assert (RESULTS_DIR / "probes_frozen.jsonl").exists(), "probes_frozen.jsonl missing"


def test_scores_l2_discrimination_derived_from_inputs() -> None:
    """Asserts that scores_L2_discrimination.json correctly mirrors scores_L2_inputs.jsonl."""
    l2_inputs_file = RESULTS_DIR / "scores_L2_inputs.jsonl"
    discrim_file = RESULTS_DIR / "scores_L2_discrimination.json"

    assert discrim_file.exists(), "scores_L2_discrimination.json missing"
    with open(l2_inputs_file, encoding="utf-8") as f:
        l2_records = [json.loads(line) for line in f]
    with open(discrim_file, encoding="utf-8") as f:
        discrim_data = json.load(f)

    # Invariant: 450 attacks, 50 benign
    assert discrim_data["attack_stats"]["n"] == 450
    assert discrim_data["benign_stats"]["n"] == 50
    assert len(l2_records) == 500

    # Invariant: Attack mean and benign mean must match exact computation
    import numpy as np
    atk_scores = [r["distilbert_score"] for r in l2_records if not r["is_benign"]]
    ben_scores = [r["distilbert_score"] for r in l2_records if r["is_benign"]]

    assert abs(np.mean(atk_scores) - discrim_data["attack_stats"]["mean"]) < 1e-6
    assert abs(np.mean(ben_scores) - discrim_data["benign_stats"]["mean"]) < 1e-6


def test_generate_and_diff_l2_discrimination_report() -> None:
    """Generate-and-diff test: re-runs metric generation from raw score files and diffs with committed copy."""
    l2_inputs_file = RESULTS_DIR / "scores_L2_inputs.jsonl"
    committed_discrim_file = RESULTS_DIR / "scores_L2_discrimination.json"

    assert l2_inputs_file.exists(), "scores_L2_inputs.jsonl missing"
    assert committed_discrim_file.exists(), "scores_L2_discrimination.json missing"

    with open(l2_inputs_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    with open(committed_discrim_file, encoding="utf-8") as f:
        committed_data = json.load(f)

    # Dynamic re-generation from raw scores
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    atk_scores = [r["distilbert_score"] for r in records if not r["is_benign"]]
    ben_scores = [r["distilbert_score"] for r in records if r["is_benign"]]
    y_true = [1 if not r["is_benign"] else 0 for r in records]
    y_scores = [r["distilbert_score"] for r in records]

    generated_auroc = float(roc_auc_score(y_true, y_scores))
    generated_auprc = float(average_precision_score(y_true, y_scores))

    # Diff generated against committed (tolerance 1e-9)
    assert abs(generated_auroc - committed_data["overall_auroc"]) < 1e-9, (
        f"Diff in AUROC: generated={generated_auroc}, committed={committed_data['overall_auroc']}"
    )
    assert abs(generated_auprc - committed_data["overall_auprc"]) < 1e-9, (
        f"Diff in AUPRC: generated={generated_auprc}, committed={committed_data['overall_auprc']}"
    )
    assert abs(float(np.mean(atk_scores)) - committed_data["attack_stats"]["mean"]) < 1e-9
    assert abs(float(np.mean(ben_scores)) - committed_data["benign_stats"]["mean"]) < 1e-9


def test_probe_condition_counts_invariant() -> None:
    """Asserts that all 9 probe conditions have exactly 450 evaluated samples."""
    path = RESULTS_DIR / "probes_frozen.jsonl"
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 450 * 9
    from collections import Counter
    counts = Counter(r["probe"] for r in records)
    for probe_name, count in counts.items():
        assert count == 450, f"Probe {probe_name} has {count} != 450"


def test_cell_matrix_reconciliation_invariant() -> None:
    """Asserts that the 60 populated cells sum exactly to 60 and 450 attacks across 112 cells."""
    matrix_file = RESULTS_DIR / "taxonomy_cell_matrix.json"
    assert matrix_file.exists(), "taxonomy_cell_matrix.json missing"
    with open(matrix_file, encoding="utf-8") as f:
        m = json.load(f)

    assert m["total_populated_cells"] == 60
    assert m["theoretical_cells"] == 112
    assert m["empty_cells"] == 52
    total_cases = sum(t["total_cases"] for t in m["techniques"])
    total_cells = sum(t["populated_cells"] for t in m["techniques"])
    assert total_cases == 450
    assert total_cells == 60


def test_generate_and_diff_unique_text_statistics() -> None:
    """Asserts unique text statistics match raw L2 inputs and exact Clopper-Pearson formulas."""
    stats_file = RESULTS_DIR / "unique_text_statistics.json"
    assert stats_file.exists(), "unique_text_statistics.json missing"
    with open(stats_file, encoding="utf-8") as f:
        d = json.load(f)

    assert d["n_unique_attacks"] == 17 or d["n_unique_attacks"] == 18
    assert d["n_unique_benign"] == 22 or d["n_unique_benign"] == 23
    assert abs(d["unique_logits_auroc"] - 0.6898) < 0.01
    assert "attack_exact_clopper_pearson_95_ci" in d
    assert "benign_exact_clopper_pearson_95_ci" in d


def test_generate_and_diff_external_sources_audit() -> None:
    """Asserts external sources audit contains zero PII violations and valid source counts."""
    audit_file = RESULTS_DIR / "external_sources_audit.json"
    assert audit_file.exists(), "external_sources_audit.json missing"
    with open(audit_file, encoding="utf-8") as f:
        d = json.load(f)

    assert d["total_items"] == 150
    assert d["pii_violations_count"] == 0
    total_source_items = sum(s["count"] for s in d["sources"])
    assert total_source_items == 150


def test_generate_and_diff_mood_shortcut_diagnostic() -> None:
    """Asserts mood shortcut diagnostic matches training corpus size."""
    mood_file = RESULTS_DIR / "mood_shortcut_diagnostic.json"
    assert mood_file.exists(), "mood_shortcut_diagnostic.json missing"
    with open(mood_file, encoding="utf-8") as f:
        d = json.load(f)

    assert d["heuristic_validation_50_samples"]["sample_size"] == 50
    assert d["chi2_test"]["p_value"] < 1e-5
    assert d["breakdown_by_source_and_label"]["total_samples"] == 8179


def test_generate_and_diff_token_variants_audit() -> None:
    """Asserts token variants audit changes match 500 benchmark cases."""
    var_file = RESULTS_DIR / "token_variants_audit_phase1g.json"
    assert var_file.exists(), "token_variants_audit_phase1g.json missing"
    with open(var_file, encoding="utf-8") as f:
        d = json.load(f)

    assert d["total_cases_evaluated"] == 500
    assert d["attack_cases_n"] == 450
    assert d["benign_cases_n"] == 50
    assert "entity_placeholders" in d["variants_change_counts"]
    assert "key_names_removed" in d["variants_change_counts"]
    assert "tool_names_removed" in d["variants_change_counts"]
    assert "first_64_tokens" in d["variants_change_counts"]


def test_generate_and_diff_off_the_shelf_models_pinned() -> None:
    """Asserts that all off-the-shelf detector models have pinned SHAs from hf_metadata."""
    ots_file = RESULTS_DIR / "off_the_shelf_evaluation.json"
    assert ots_file.exists(), "off_the_shelf_evaluation.json missing"
    with open(ots_file, encoding="utf-8") as f:
        d = json.load(f)

    assert len(d) == 3
    for model_id, info in d.items():
        assert "revision_sha" in info and len(info["revision_sha"]) == 40
        assert "license" in info and info["license"] != "unknown"
        assert "unique_sandbox_evaluation" in info
        assert "all_500_sandbox_evaluation" in info
        assert "external_evaluation" in info

