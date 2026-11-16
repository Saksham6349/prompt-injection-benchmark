"""CLI to run benchmark evaluation across models and defenses."""

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
from src.evaluation.metrics import aggregate_metrics
from src.evaluation.report import ReportGenerator
from src.evaluation.runner import EvalRunner
from src.types import DefenseConfig


def get_provider(name: str):
    name = name.lower()
    if name == "mock":
        return MockProvider()
    elif name == "openai":
        return OpenAIProvider()
    elif name == "anthropic":
        return AnthropicProvider()
    elif name == "google":
        return GoogleProvider()
    elif name == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unknown provider '{name}'. Available: mock, openai, anthropic, google, ollama")


def get_defense(name: str):
    name = name.lower()
    if name == "none":
        return None
    elif name == "all" or name == "our_defense":
        cfg = DefenseConfig(enable_provenance=True, enable_classifier=True, enable_policy=True)
        p = DefensePipeline(cfg)
        setattr(p, "name", "Layered Defense (All)")
        return p
    elif name == "provenance":
        cfg = DefenseConfig(enable_provenance=True, enable_classifier=False, enable_policy=False)
        p = DefensePipeline(cfg)
        setattr(p, "name", "Provenance Only")
        return p
    elif name == "classifier":
        cfg = DefenseConfig(enable_provenance=False, enable_classifier=True, enable_policy=False)
        p = DefensePipeline(cfg)
        setattr(p, "name", "Classifier Only")
        return p
    elif name == "policy":
        cfg = DefenseConfig(enable_provenance=False, enable_classifier=False, enable_policy=True)
        p = DefensePipeline(cfg)
        setattr(p, "name", "Policy Only")
        return p
    else:
        raise ValueError(f"Unknown defense '{name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt injection benchmark evaluation.")
    parser.add_argument("--cases", type=str, default="data/benchmark_cases.jsonl", help="Cases JSONL path")
    parser.add_argument("--provider", type=str, default="mock", help="LLM provider (mock, openai, anthropic, google, ollama)")
    parser.add_argument("--defense", type=str, default="none", help="Defense (none, all, provenance, classifier, policy)")
    parser.add_argument("--max-cases", type=int, default=None, help="Evaluate on subset of cases")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save evaluation results")
    args = parser.parse_args()

    print(f"Loading cases from {args.cases}...")
    cases = BenchmarkGenerator.load_cases(args.cases)
    if args.max_cases:
        cases = cases[:args.max_cases]
    print(f"Loaded {len(cases)} cases.")

    provider = get_provider(args.provider)
    defense = get_defense(args.defense)

    runner = EvalRunner(output_dir=args.output_dir)
    print(f"Running benchmark with provider='{args.provider}', defense='{args.defense}'...")
    results = runner.run_benchmark(cases, provider, defense_pipeline=defense)

    filename = f"eval_{args.provider}_{args.defense}.jsonl"
    saved_path = runner.save_results(results, filename)
    print(f"\nResults saved to {saved_path}")

    # Print summary metrics
    metrics = aggregate_metrics(results)
    print("\n" + "="*50)
    print(f"Evaluation Summary: {args.provider.upper()} | Defense: {args.defense}")
    print(f"  Total Cases          : {metrics.total_cases}")
    print(f"  Attack Success Rate  : {metrics.attack_success_rate * 100:.2f}%")
    print(f"  Benign Utility Score : {metrics.utility_score * 100:.2f}%")
    print(f"  Mean Latency         : {metrics.mean_latency_ms:.1f} ms")
    if metrics.confidence_interval_95:
        print(f"  95% CI for ASR       : [{metrics.confidence_interval_95[0]:.2f}, {metrics.confidence_interval_95[1]:.2f}]")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
