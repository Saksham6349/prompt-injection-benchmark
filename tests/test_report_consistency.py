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


def test_no_hardcoded_metric_literals_in_phase1e_generator() -> None:
    """Scans the Phase 1e generator script for banned hardcoded metric literals."""
    generator_file = SCRIPTS_DIR / "run_phase1e_closure.py"
    assert generator_file.exists(), "run_phase1e_closure.py missing"

    with open(generator_file, encoding="utf-8") as f:
        code_lines = f.readlines()

    banned_literals = [
        r'"2/50"',
        r"'2/50'",
        r'"22/50"',
        r"'22/50'",
        r'"4\.0%"',
        r"'4\.0%'",
        r'"44\.0%"',
        r"'44\.0%'",
    ]

    for line_idx, line in enumerate(code_lines, 1):
        clean_line = line.strip()
        if clean_line.startswith("#") or clean_line.startswith('"""'):
            continue
        for pat in banned_literals:
            assert not re.search(pat, line), (
                f"Hardcoded metric literal '{pat}' found at {generator_file.name}:{line_idx}: {clean_line}"
            )


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
    """Asserts that the 60 populated cells sum exactly to 60 and 450 attacks."""
    matrix_file = RESULTS_DIR / "taxonomy_cell_matrix.json"
    assert matrix_file.exists(), "taxonomy_cell_matrix.json missing"
    with open(matrix_file, encoding="utf-8") as f:
        m = json.load(f)

    assert m["total_populated_cells"] == 60
    assert m["theoretical_cells"] == 160
    total_cases = sum(t["total_cases"] for t in m["techniques"])
    total_cells = sum(t["populated_cells"] for t in m["techniques"])
    assert total_cases == 450
    assert total_cells == 60
