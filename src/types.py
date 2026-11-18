"""Shared types used across all modules."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Attack Taxonomy Enums ──────────────────────────────────────────────

class AttackGoal(str, enum.Enum):
    DATA_EXFILTRATION = "data_exfiltration"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    GOAL_HIJACK = "goal_hijack"
    DENIAL_OF_SERVICE = "denial_of_service"


class AttackChannel(str, enum.Enum):
    WEB_PAGE = "web_page"
    EMAIL = "email"
    PDF = "pdf"
    API_RESPONSE = "api_response"
    RETRIEVED_DOCUMENT = "retrieved_document"


class AttackTechnique(str, enum.Enum):
    HIDDEN_TEXT = "hidden_text"
    ROLE_PLAY = "role_play"
    TOOL_CALL_FORGING = "tool_call_forging"
    MULTI_STEP = "multi_step"
    ENCODED_PAYLOAD = "encoded_payload"
    MULTILINGUAL = "multilingual"
    CONTEXT_MANIPULATION = "context_manipulation"
    PAYLOAD_SPLITTING = "payload_splitting"


class Language(str, enum.Enum):
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hi-en"


# ── Tool-Related Types ─────────────────────────────────────────────────

@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: list[ToolParameter]
    category: str = "general"

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function-calling schema."""
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class ToolCall:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_malicious: bool | None = None  # Ground truth label


@dataclass
class ToolOutput:
    """Raw output from a tool, before any defense processing."""
    tool_name: str
    content: str
    channel: AttackChannel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Benchmark Types ────────────────────────────────────────────────────

@dataclass
class AttackMetadata:
    goal: AttackGoal
    channel: AttackChannel
    technique: AttackTechnique
    language: Language = Language.ENGLISH
    expected_malicious_calls: list[dict[str, Any]] = field(default_factory=list)
    description: str = ""


@dataclass
class BenchmarkCase:
    id: str
    user_task: str                        # The benign user request
    user_task_id: str                     # Links back to the base benign task
    injection_payload: str                # The injected instruction
    injection_location: str               # Where the injection is placed
    attack_metadata: AttackMetadata
    expected_benign_calls: list[dict[str, Any]] = field(default_factory=list)
    is_benign: bool = False               # True for benign-only control cases
    split: str = "public"                 # "public" or "private"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_task": self.user_task,
            "user_task_id": self.user_task_id,
            "injection_payload": self.injection_payload,
            "injection_location": self.injection_location,
            "attack_metadata": {
                "goal": self.attack_metadata.goal.value,
                "channel": self.attack_metadata.channel.value,
                "technique": self.attack_metadata.technique.value,
                "language": self.attack_metadata.language.value,
                "expected_malicious_calls": self.attack_metadata.expected_malicious_calls,
                "description": self.attack_metadata.description,
            },
            "expected_benign_calls": self.expected_benign_calls,
            "is_benign": self.is_benign,
            "split": self.split,
        }


# ── Defense Types ──────────────────────────────────────────────────────

@dataclass
class DefenseConfig:
    enable_provenance: bool = True
    enable_classifier: bool = False   # Disabled by default: Layer 2 fails on tool-output channel (ablation condition only)
    enable_policy: bool = True
    classifier_model: str = "microsoft/deberta-v3-small"
    classifier_threshold: float = 0.5
    policy_allow_list: dict[str, list[str]] = field(default_factory=dict)
    # Maps user-intent keywords to allowed tool names


@dataclass
class DefenseVerdict:
    is_blocked: bool = False
    layer: str = ""                       # Which layer triggered the block
    confidence: float = 0.0
    explanation: str = ""
    raw_scores: dict[str, float] = field(default_factory=dict)


# ── Evaluation Types ───────────────────────────────────────────────────

@dataclass
class EvalResult:
    case_id: str
    model: str
    defense: str                          # "none", "provenance", "classifier", etc.
    attack_succeeded: bool
    utility_preserved: bool               # Did the benign task also complete?
    tool_calls_made: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    defense_verdict: DefenseVerdict | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "model": self.model,
            "defense": self.defense,
            "attack_succeeded": self.attack_succeeded,
            "utility_preserved": self.utility_preserved,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "result": tc.result}
                for tc in self.tool_calls_made
            ],
            "latency_ms": self.latency_ms,
            "defense_verdict": {
                "is_blocked": self.defense_verdict.is_blocked,
                "layer": self.defense_verdict.layer,
                "confidence": self.defense_verdict.confidence,
                "explanation": self.defense_verdict.explanation,
            } if self.defense_verdict else None,
            "error": self.error,
        }


@dataclass
class AggregateMetrics:
    total_cases: int = 0
    attack_success_rate: float = 0.0      # ASR
    utility_score: float = 0.0            # % of benign tasks completed
    mean_latency_ms: float = 0.0
    classifier_auroc: float | None = None
    classifier_fpr: float | None = None   # False positive rate
    confidence_interval_95: tuple[float, float] | None = None
