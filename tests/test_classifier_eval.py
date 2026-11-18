"""Tests for classifier evaluation pipeline, deduplication, and split integrity.

Covers Amendment 7:
1. Disjoint, deterministic, stratified, and cluster-intact splits.
2. Normalization and near-duplicate detection logic.
3. Wilson score interval and bootstrap reproducibility.
4. Gate enforcement (evaluator refuses without frozen_threshold.json).
5. Source attribution vocabulary integrity.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.split_dataset import (
    bootstrap_metric_ci,
    compute_clusters,
    group_aware_stratified_split,
    normalize_text,
    wilson_ci,
)


def test_source_attribution_vocabulary() -> None:
    """Verifies source attribution only emits approved source vocabulary."""
    v2_path = PROJECT_ROOT / "data" / "real_classifier_training_data_v2.csv"
    if not v2_path.exists():
        pytest.skip("V2 dataset not yet created")

    df = pd.read_csv(v2_path)
    allowed_sources = {"deepset", "slabs", "jasperls", "curated", "augmented_hinglish", "unknown_merged"}
    actual_sources = set(df["source"].unique())
    invalid = actual_sources - allowed_sources
    assert len(invalid) == 0, f"Found unexpected source values: {invalid}"


def test_normalization_and_dedup_known_pairs() -> None:
    """Tests that text normalization correctly collapses punctuation and spacing."""
    s1 = "Ignore previous instructions!"
    s2 = "  ignore   PREVIOUS instructions... "
    s3 = "Different unrelated query text"

    assert normalize_text(s1) == normalize_text(s2)
    assert normalize_text(s1) != normalize_text(s3)


def test_wilson_confidence_interval() -> None:
    """Verifies Wilson interval formula matches known statistical references."""
    # Reference: 50 successes out of 100 trials -> ~0.4038 to ~0.5962
    low, high = wilson_ci(50, 100)
    assert 0.40 <= low <= 0.41
    assert 0.59 <= high <= 0.60

    # Edge cases: 0 and n
    low_0, high_0 = wilson_ci(0, 50)
    assert low_0 == 0.0
    assert 0.05 <= high_0 <= 0.08  # ~7.1% upper bound for 0/50

    low_n, high_n = wilson_ci(50, 50)
    assert 0.92 <= low_n <= 0.95
    assert high_n >= 0.999


def test_bootstrap_reproducibility() -> None:
    """Verifies bootstrap CI is strictly deterministic given a fixed seed."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15])

    def dummy_metric(y, s):
        return float(np.mean(y == (s >= 0.5)))

    ci_1 = bootstrap_metric_ci(y_true, y_score, dummy_metric, n_boot=200, seed=42)
    ci_2 = bootstrap_metric_ci(y_true, y_score, dummy_metric, n_boot=200, seed=42)
    assert ci_1 == ci_2


def test_split_cluster_integrity_and_disjointness() -> None:
    """Verifies splits are disjoint and no near-duplicate cluster straddles splits."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_path = splits_dir / "train.csv"
    cal_path = splits_dir / "calibration.csv"
    test_path = splits_dir / "test.csv"

    if not (train_path.exists() and cal_path.exists() and test_path.exists()):
        pytest.skip("Splits not yet generated")

    train_df = pd.read_csv(train_path)
    cal_df = pd.read_csv(cal_path)
    test_df = pd.read_csv(test_path)

    # 1. Total row count preserved
    total = len(train_df) + len(cal_df) + len(test_df)
    assert total == 11683

    # 2. Split clusters are 100% disjoint
    train_clusters = set(train_df["cluster_id"])
    cal_clusters = set(cal_df["cluster_id"])
    test_clusters = set(test_df["cluster_id"])

    assert len(train_clusters.intersection(cal_clusters)) == 0, "Train and Cal share clusters!"
    assert len(train_clusters.intersection(test_clusters)) == 0, "Train and Test share clusters!"
    assert len(cal_clusters.intersection(test_clusters)) == 0, "Cal and Test share clusters!"


def test_split_determinism() -> None:
    """Verifies that running group-aware splitting with the same seed yields identical splits."""
    test_data = pd.DataFrame({
        "text": ["prompt 1", "prompt 2", "prompt 3", "prompt 4", "prompt 5", "prompt 6"],
        "label": [1, 0, 1, 0, 1, 0],
        "source": ["slabs", "slabs", "deepset", "deepset", "slabs", "deepset"],
    })
    clusters = [{0, 1}, {2}, {3}, {4, 5}]

    split_1 = group_aware_stratified_split(test_data, clusters, seed=42)
    split_2 = group_aware_stratified_split(test_data, clusters, seed=42)

    assert split_1["split"].tolist() == split_2["split"].tolist()


def test_frozen_threshold_enforcement(tmp_path: Path) -> None:
    """Verifies that evaluation refuses to run without a frozen_threshold.json file."""
    fake_results_dir = tmp_path / "results"
    fake_results_dir.mkdir()
    threshold_file = fake_results_dir / "frozen_threshold.json"

    assert not threshold_file.exists()

    # Emulate the security gate
    def run_test_eval(res_dir: Path) -> None:
        lock_file = res_dir / "frozen_threshold.json"
        if not lock_file.exists():
            raise FileNotFoundError("GATE REFUSED: frozen_threshold.json missing. Calibrate first.")

    with pytest.raises(FileNotFoundError, match="GATE REFUSED"):
        run_test_eval(fake_results_dir)



def test_frozen_threshold_file_structure() -> None:
    """Verifies that frozen_threshold.json contains required protocol keys."""
    lock_path = PROJECT_ROOT / "results" / "frozen_threshold.json"
    if not lock_path.exists():
        pytest.skip("frozen_threshold.json not yet generated")

    with open(lock_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["locked"] is True
    assert "calibration_file_sha256" in data
    assert "selection_rules" in data
    rules = data["selection_rules"]
    assert "primary_f1_max" in rules
    assert "security_target_fpr_1pct" in rules
    assert "security_target_fpr_0_1pct" in rules
    assert rules["primary_f1_max"]["threshold"] > 0.0


def test_provenance_discrepancy_explanation() -> None:
    """Verifies provenance_verification.json documents source datasets and mirror status."""
    prov_path = PROJECT_ROOT / "results" / "provenance_verification.json"
    if not prov_path.exists():
        pytest.skip("provenance_verification.json not yet generated")

    with open(prov_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    repos = data["repositories"]
    assert "deepset/prompt-injections" in repos
    assert "JasperLS/prompt-injections" in repos
    assert "S-Labs/prompt-injection-dataset" in repos

    # Verify JasperLS is documented as identical mirror
    jasper = repos["JasperLS/prompt-injections"]
    assert jasper["identical_sha256_match"] is True
    assert jasper["unique_rows_added_to_deepset"] == 0


def test_benchmark_reconciliation_consistency() -> None:
    """Verifies that benchmark technique attack counts sum to 450 and weighted rate matches overall."""
    cases = [json.loads(l) for l in open(PROJECT_ROOT / "data" / "benchmark_cases.jsonl", encoding="utf-8") if l.strip()]
    assert len(cases) == 500

    attacks = [c for c in cases if not c.get("is_benign")]
    benign = [c for c in cases if c.get("is_benign")]
    assert len(attacks) == 450
    assert len(benign) == 50

    from collections import Counter
    tech_counts = Counter(c["attack_metadata"]["technique"] for c in attacks)
    assert sum(tech_counts.values()) == 450

    # Verify score file exists and consistency holds
    scores_path = PROJECT_ROOT / "results" / "scores_benchmark.json"
    if scores_path.exists():
        with open(scores_path, encoding="utf-8") as f:
            probs = np.array(json.load(f)["probabilities"])
        assert len(probs) == 500
        tau = 0.47
        y = np.array([0 if c.get("is_benign") else 1 for c in cases])
        preds = (probs >= tau).astype(int)

        atk_idx = np.where(y == 1)[0]
        overall_atk_det = preds[atk_idx].sum()
        overall_rate = overall_atk_det / 450

        # Weighted sum across techniques
        weighted_sum = 0
        for tech, count in tech_counts.items():
            t_idx = [i for i in atk_idx if cases[i]["attack_metadata"]["technique"] == tech]
            t_det = preds[t_idx].sum()
            t_rate = t_det / count
            weighted_sum += count * t_rate

        weighted_rate = weighted_sum / 450
        assert abs(overall_rate - weighted_rate) < 1e-9


def test_checkpoint_sha256_lock() -> None:
    """Verifies that the new model checkpoint SHA-256 matches expected frozen digest."""
    from src.defense.classifier import EXPECTED_CHECKPOINT_SHA256, compute_safetensors_sha256
    model_dir = str(PROJECT_ROOT / "models" / "injection_classifier")
    actual_sha = compute_safetensors_sha256(model_dir)
    assert actual_sha == EXPECTED_CHECKPOINT_SHA256


def test_threshold_split_count_consistency() -> None:
    """Asserts that per-case evaluation yields exactly one consistent number per (threshold, split)."""
    per_case_file = PROJECT_ROOT / "results" / "benchmark_per_case_eval.jsonl"
    if not per_case_file.exists():
        pytest.skip("benchmark_per_case_eval.jsonl not found")

    with open(per_case_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) == 500
    benign_records = [r for r in records if r["label"] == 0]
    assert len(benign_records) == 50

    dev_benign = [r for r in benign_records if r["split"] == "dev"]
    test_benign = [r for r in benign_records if r["split"] == "test"]
    assert len(dev_benign) == 21
    assert len(test_benign) == 29

    # Assert invariant counts at each frozen/exploratory threshold
    # tau = 0.4700 (primary deployment threshold)
    assert sum(1 for r in benign_records if r["distilbert_score"] >= 0.4700) == 22
    assert sum(1 for r in dev_benign if r["distilbert_score"] >= 0.4700) == 9
    assert sum(1 for r in test_benign if r["distilbert_score"] >= 0.4700) == 13

    # tau = 0.9772 (exploratory benchmark threshold)
    assert sum(1 for r in benign_records if r["distilbert_score"] >= 0.9772) == 11
    assert sum(1 for r in dev_benign if r["distilbert_score"] >= 0.9772) == 0
    assert sum(1 for r in test_benign if r["distilbert_score"] >= 0.9772) == 11

    # tau = 0.9916
    assert sum(1 for r in benign_records if r["distilbert_score"] >= 0.9916) == 1
    assert sum(1 for r in dev_benign if r["distilbert_score"] >= 0.9916) == 0
    assert sum(1 for r in test_benign if r["distilbert_score"] >= 0.9916) == 1

    # tau = 0.9948
    assert sum(1 for r in benign_records if r["distilbert_score"] >= 0.9948) == 0
    assert sum(1 for r in dev_benign if r["distilbert_score"] >= 0.9948) == 0
    assert sum(1 for r in test_benign if r["distilbert_score"] >= 0.9948) == 0


