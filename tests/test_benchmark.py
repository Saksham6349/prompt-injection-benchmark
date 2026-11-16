"""Unit tests for attack taxonomy and benchmark generator."""

from __future__ import annotations

from src.benchmark.generator import BenchmarkGenerator
from src.taxonomy.taxonomy import get_taxonomy
from src.types import AttackGoal, AttackChannel, AttackTechnique


def test_taxonomy_structure():
    tax = get_taxonomy()
    assert len(tax) == 4
    for goal_key, channels in tax.items():
        assert len(channels) == 5
        for ch_key, techniques in channels.items():
            assert len(techniques) == 8


def test_benchmark_generation_and_distributions():
    gen = BenchmarkGenerator(seed=123)
    cases = gen.generate(total_cases=50, public_ratio=0.8, benign_ratio=0.1)

    assert len(cases) == 50
    benign_cases = [c for c in cases if c.is_benign]
    injected_cases = [c for c in cases if not c.is_benign]

    assert len(benign_cases) == 5
    assert len(injected_cases) == 45

    # Check splits
    public_cases = [c for c in cases if c.split == "public"]
    private_cases = [c for c in cases if c.split == "private"]
    assert len(public_cases) > len(private_cases)

    # Check case fields
    for c in cases:
        assert c.id.startswith("case_")
        assert len(c.user_task) > 0
        assert c.attack_metadata is not None
        if not c.is_benign:
            assert len(c.injection_payload) > 0


def test_case_serialization_roundtrip(tmp_path):
    gen = BenchmarkGenerator(seed=42)
    cases = gen.generate(total_cases=10)
    file_path = tmp_path / "test_cases.jsonl"

    gen.save_cases(cases, file_path)
    loaded = gen.load_cases(file_path)

    assert len(loaded) == len(cases)
    for orig, load in zip(cases, loaded):
        assert orig.id == load.id
        assert orig.user_task == load.user_task
        assert orig.is_benign == load.is_benign
        assert orig.split == load.split
        assert orig.attack_metadata.goal == load.attack_metadata.goal
