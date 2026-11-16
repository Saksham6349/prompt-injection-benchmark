"""Baseline defenses for comparison."""

from src.baselines.base import BaseDefense
from src.baselines.spotlighting import SpotlightingDefense
from src.baselines.sandwich import SandwichDefense
from src.baselines.perplexity_filter import PerplexityFilter
from src.baselines.promptguard import PromptGuardDefense
from src.baselines.llm_judge import LLMJudgeDefense

__all__ = [
    "BaseDefense",
    "SpotlightingDefense",
    "SandwichDefense",
    "PerplexityFilter",
    "PromptGuardDefense",
    "LLMJudgeDefense",
]
