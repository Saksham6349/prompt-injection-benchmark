"""Integrity test asserting exact cryptographic hashes and single source of truth.

Fails if any report or probe record diverges from frozen ground truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

EXPECTED_BENCHMARK_FROZEN_SHA = "7c79493d9d19c95baa3a5fdc0788c51170d9657f3a8b5153ca84e193d04d8dfe"
EXPECTED_PROBES_FROZEN_SHA = "cf9e78df61647c355437321158321496af4a7e111569e78f3c8c315a6f984d4d"
EXPECTED_L2_INPUTS_SHA = "99623e3d5dcfdc6aed3f63fb915b5951312fd016990869168926867aa689b511"


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_frozen_benchmark_scores_hash() -> None:
    path = RESULTS_DIR / "scores_benchmark_frozen.jsonl"
    assert path.exists(), "scores_benchmark_frozen.jsonl missing"
    assert compute_file_sha256(path) == EXPECTED_BENCHMARK_FROZEN_SHA


def test_frozen_probes_hash() -> None:
    path = RESULTS_DIR / "probes_frozen.jsonl"
    assert path.exists(), "probes_frozen.jsonl missing"
    assert compute_file_sha256(path) == EXPECTED_PROBES_FROZEN_SHA


def test_scores_l2_inputs_hash() -> None:
    path = RESULTS_DIR / "scores_L2_inputs.jsonl"
    assert path.exists(), "scores_L2_inputs.jsonl missing"
    assert compute_file_sha256(path) == EXPECTED_L2_INPUTS_SHA


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
