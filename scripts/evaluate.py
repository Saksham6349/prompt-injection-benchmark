"""Full Evaluation Pipeline Script.

Executes a comparative benchmark across defenses (none, provenance, classifier, policy, all)
and generates a structured comparison report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.providers.mock_provider import MockProvider
from src.agent.providers.openai_provider import OpenAIProvider
from src.agent.providers.anthropic_provider import AnthropicProvider
from src.agent.providers.google_provider import GoogleProvider
from src.agent.providers.ollama_provider import OllamaProvider
from src.benchmark.generator import BenchmarkGenerator
from src.defense.pipeline import DefensePipeline
from src.evaluation.report import ReportGenerator
from src.evaluation.runner import EvalRunner
from src.types import DefenseConfig, EvalResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete comparative defense evaluation.")
    parser.add_argument("--cases", type=str, default="data/benchmark_cases.jsonl", help="Cases JSONL path")
    parser.add_argument("--provider", type=str, default="mock", help="Provider (mock, openai, anthropic, google, ollama)")
    parser.add_argument("--max-cases", type=int, default=50, help="Number of cases to evaluate per condition")
    parser.add_argument("--output-report", type=str, default="results/evaluation_report.md", help="Path for markdown report")
    args = parser.parse_args()

    cases = BenchmarkGenerator.load_cases(args.cases)
    if args.max_cases:
        cases = cases[:args.max_cases]

    print(f"Loaded {len(cases)} cases for comparative evaluation.")

    # Select provider
    provider_map = {
        "mock": MockProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
    }
    provider = provider_map.get(args.provider, MockProvider)()

    # Setup defenses to compare
    defenses = {
        "Undefended": None,
        "Layer 1: Provenance Only": DefensePipeline(DefenseConfig(enable_provenance=True, enable_classifier=False, enable_policy=False)),
        "Layer 2: Classifier Only": DefensePipeline(DefenseConfig(enable_provenance=False, enable_classifier=True, enable_policy=False)),
        "Layer 3: Policy Only": DefensePipeline(DefenseConfig(enable_provenance=False, enable_classifier=False, enable_policy=True)),
        "Our 3-Layer Defense": DefensePipeline(DefenseConfig(enable_provenance=True, enable_classifier=True, enable_policy=True)),
    }

    runner = EvalRunner()
    results_by_defense: dict[str, list[EvalResult]] = {}

    for name, pipeline in defenses.items():
        print(f"\nEvaluating condition: {name}...")
        res = runner.run_benchmark(cases, provider, defense_pipeline=pipeline, show_progress=True)
        results_by_defense[name] = res

    # Generate Report
    report = ReportGenerator.generate_full_report(results_by_defense, title=f"Prompt Injection Defense Benchmark ({args.provider.upper()})")
    ReportGenerator.save_report(report, args.output_report)

    print("\n" + "="*60)
    print(f"Report generated successfully at: {args.output_report}")
    print("="*60)
    print("\n" + ReportGenerator.generate_summary_table(results_by_defense))


if __name__ == "__main__":
    main()
