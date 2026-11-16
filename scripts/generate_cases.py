"""CLI tool to generate benchmark cases and output distributions."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.generator import BenchmarkGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark prompt injection cases.")
    parser.add_argument("--output", type=str, default="data/benchmark_cases.jsonl", help="Output path")
    parser.add_argument("--total", type=int, default=500, help="Total number of cases to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--public-ratio", type=float, default=0.8, help="Ratio for public split")
    args = parser.parse_args()

    print(f"Generating {args.total} benchmark cases (seed={args.seed})...")
    gen = BenchmarkGenerator(seed=args.seed)
    cases = gen.generate(total_cases=args.total, public_ratio=args.public_ratio)

    gen.save_cases(cases, args.output)
    print(f"Successfully saved {len(cases)} cases to {args.output}\n")

    # Display distributions
    goals = Counter(c.attack_metadata.goal.value for c in cases if not c.is_benign)
    channels = Counter(c.attack_metadata.channel.value for c in cases if not c.is_benign)
    techniques = Counter(c.attack_metadata.technique.value for c in cases if not c.is_benign)
    langs = Counter(c.attack_metadata.language.value for c in cases if not c.is_benign)
    splits = Counter(c.split for c in cases)
    benign_count = sum(1 for c in cases if c.is_benign)

    print("=== Attack Goal Distribution ===")
    for g, count in goals.items():
        print(f"  {g:25s}: {count}")

    print("\n=== Attack Channel Distribution ===")
    for ch, count in channels.items():
        print(f"  {ch:25s}: {count}")

    print("\n=== Technique Distribution ===")
    for t, count in techniques.items():
        print(f"  {t:25s}: {count}")

    print("\n=== Language Distribution ===")
    for l, count in langs.items():
        print(f"  {l:25s}: {count}")

    print(f"\n=== Splits & Controls ===")
    print(f"  Public Split             : {splits['public']}")
    print(f"  Private Split            : {splits['private']}")
    print(f"  Benign Controls          : {benign_count}")


if __name__ == "__main__":
    main()
