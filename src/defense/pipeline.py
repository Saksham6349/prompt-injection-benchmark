"""Combined three-layer defense pipeline.

Orchestrates provenance tagging (Layer 1), classifier (Layer 2), and
policy checking (Layer 3) into a single entry point.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.types import DefenseConfig, DefenseVerdict, ToolCall, ToolOutput
from src.defense.provenance import ProvenanceTagger
from src.defense.classifier import InjectionClassifier
from src.defense.policy import PolicyChecker

log = logging.getLogger(__name__)


class DefensePipeline:
    """End-to-end defense pipeline combining all three layers.

    Each layer can be independently enabled or disabled through
    :class:`DefenseConfig`.
    """

    def __init__(self, config: DefenseConfig | None = None) -> None:
        self.config = config or DefenseConfig()
        self._tagger = ProvenanceTagger(fmt="structured")
        self._classifier = InjectionClassifier(
            model_name=self.config.classifier_model,
            threshold=self.config.classifier_threshold,
        )
        self._policy = PolicyChecker(self.config)

    # ── Tool-output processing (Layers 1 + 2) ─────────────────────────

    def process_tool_output(
        self,
        tool_output: ToolOutput,
        user_task: str,
    ) -> tuple[str, DefenseVerdict]:
        """Process raw tool output through Layers 1 and 2.

        Args:
            tool_output: The raw :class:`ToolOutput` from the sandbox.
            user_task: The original user request.

        Returns:
            A tuple of ``(tagged_text, verdict)``.
            *tagged_text* is the provenance-tagged version of the output.
            *verdict* records whether Layer 2 flagged the content.
        """
        raw_text = tool_output.content
        verdict = DefenseVerdict()

        # Layer 1 — Provenance tagging
        if self.config.enable_provenance:
            tagged_text = self._tagger.tag_output(tool_output)
            log.debug("Provenance tag applied for tool '%s'.", tool_output.tool_name)
        else:
            tagged_text = raw_text

        # Layer 2 — Classifier
        if self.config.enable_classifier:
            is_injection, confidence = self._classifier.predict(raw_text)
            verdict = DefenseVerdict(
                is_blocked=is_injection,
                layer="classifier",
                confidence=confidence,
                explanation=(
                    f"Classifier flagged output from '{tool_output.tool_name}' "
                    f"as {'injection' if is_injection else 'benign'} "
                    f"(confidence={confidence:.3f})."
                ),
                raw_scores={"classifier_confidence": confidence},
            )
            if is_injection:
                log.warning(
                    "Layer 2 BLOCKED output from '%s' (confidence=%.3f).",
                    tool_output.tool_name,
                    confidence,
                )
            else:
                log.debug(
                    "Layer 2 passed output from '%s' (confidence=%.3f).",
                    tool_output.tool_name,
                    confidence,
                )

        return tagged_text, verdict

    # ── Tool-call checking (Layer 3) ───────────────────────────────────

    def check_tool_call(
        self,
        tool_call: ToolCall,
        user_task: str,
        history: list[dict[str, Any]] | None = None,
    ) -> DefenseVerdict:
        """Check a proposed tool call through Layer 3 (policy).

        Args:
            tool_call: The :class:`ToolCall` the agent wants to execute.
            user_task: The original user request.
            history: Optional conversation history.

        Returns:
            A :class:`DefenseVerdict`.
        """
        if not self.config.enable_policy:
            return DefenseVerdict(
                is_blocked=False,
                layer="policy",
                explanation="Policy layer is disabled.",
            )

        verdict = self._policy.check(tool_call, user_task, history)
        if verdict.is_blocked:
            log.warning(
                "Layer 3 BLOCKED tool call '%s': %s",
                tool_call.name,
                verdict.explanation,
            )
        else:
            log.debug("Layer 3 allowed tool call '%s'.", tool_call.name)
        return verdict

    def __call__(
        self,
        tool_call: ToolCall,
        tool_output: ToolOutput,
        user_task: str = "",
    ) -> DefenseVerdict:
        """Callable adapter for AgentHarness integration."""
        # 1. Policy check on tool call
        policy_verdict = self.check_tool_call(tool_call, user_task)
        if policy_verdict.is_blocked:
            return policy_verdict

        # 2. Classifier check on tool output
        tagged_text, out_verdict = self.process_tool_output(tool_output, user_task)
        tool_output.content = tagged_text
        if out_verdict.is_blocked:
            return out_verdict

        return DefenseVerdict(is_blocked=False, layer="pipeline", explanation="Passed all layers")

    # ── Helpers ────────────────────────────────────────────────────────

    def get_system_instruction(self) -> str:
        """Return the provenance-related system instruction (Layer 1)."""
        if self.config.enable_provenance:
            return self._tagger.get_system_instruction()
        return ""
