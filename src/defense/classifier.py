"""Layer 2: Small-Model Classifier for prompt-injection detection.

Provides ``InjectionClassifier`` (transformer-based) with a
``HeuristicDetector`` fallback that works without a trained model.
"""

from __future__ import annotations

import logging
import math
import re
import sys
import os
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

log = logging.getLogger(__name__)


# ── Heuristic Detector (fallback) ─────────────────────────────────────

class HeuristicDetector:
    """Rule-based injection detector using regex patterns.

    Good enough for initial testing before the transformer classifier is
    fine-tuned.
    """

    # Each pattern is a (compiled regex, weight) pair.
    _PATTERNS: list[tuple[re.Pattern, float]] = [
        # Direct instruction-override attempts
        (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.I), 0.9),
        (re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.I), 0.85),
        (re.compile(r"forget\s+(all\s+)?(previous|prior|above|everything)", re.I), 0.85),
        # Role-play / persona hijack
        (re.compile(r"you\s+are\s+now\s+", re.I), 0.8),
        (re.compile(r"act\s+as\s+(if\s+you\s+are\s+|a\s+)?", re.I), 0.6),
        (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I), 0.7),
        (re.compile(r"new\s+(system\s+)?instructions?:", re.I), 0.9),
        # Fake system / assistant markers
        (re.compile(r"^\s*system\s*:", re.I | re.M), 0.85),
        (re.compile(r"^\s*assistant\s*:", re.I | re.M), 0.5),
        (re.compile(r"\[SYSTEM\]", re.I), 0.85),
        (re.compile(r"\[INST\]", re.I), 0.7),
        # Tool-call forging
        (re.compile(r'"function_call"\s*:', re.I), 0.75),
        (re.compile(r'"tool_calls?"\s*:', re.I), 0.75),
        (re.compile(r'"name"\s*:\s*"(send_email|delete|execute|run_code|transfer)"', re.I), 0.8),
        # Encoded payloads
        (re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"), 0.4),  # base64-ish blocks
        (re.compile(r"\\x[0-9a-fA-F]{2}", re.I), 0.5),
        # Data exfiltration cues
        (re.compile(r"(send|forward|exfiltrate|post)\s+(all\s+)?(data|info|credentials|password|token|key)", re.I), 0.85),
        (re.compile(r"curl\s+https?://", re.I), 0.6),
        # Prompt-leaking
        (re.compile(r"(show|print|reveal|output)\s+(your\s+)?(system\s+)?(prompt|instructions)", re.I), 0.7),
        # Multilingual overrides (Hindi / Hinglish examples)
        (re.compile(r"(pichle|pehle)\s+(nirdesh|instructions?)\s+(bhool|ignore)", re.I), 0.8),
    ]

    def detect(self, text: str) -> tuple[bool, float]:
        """Return ``(is_injection, confidence)`` for *text*.

        Confidence is the maximum pattern weight among all matches, so it
        stays in ``[0, 1]``.
        """
        max_weight = 0.0
        for pattern, weight in self._PATTERNS:
            if pattern.search(text):
                max_weight = max(max_weight, weight)
        return (max_weight >= 0.5, max_weight)


# ── Transformer-Based Classifier ──────────────────────────────────────

FROZEN_TAU_STAR = 0.4700
EXPECTED_CHECKPOINT_SHA256 = "784c14f18bb459046fa286a2b3897d1c8604347ce16f28f40e33d71142cff6df"


def compute_safetensors_sha256(path: str) -> str:
    """Compute SHA-256 digest of model.safetensors if it exists."""
    import hashlib
    safetensor_f = os.path.join(path, "model.safetensors") if os.path.isdir(path) else path
    if not os.path.exists(safetensor_f):
        return ""
    hasher = hashlib.sha256()
    with open(safetensor_f, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class InjectionClassifier:
    """Transformer classifier for prompt-injection detection.

    Falls back to :class:`HeuristicDetector` when the model is not
    available (weights not downloaded or not yet fine-tuned).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        model_path: str | None = None,
        threshold: float = FROZEN_TAU_STAR,
        verify_checkpoint: bool = True,
    ) -> None:
        if model_path is None:
            default_local = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "injection_classifier"))
            if os.path.exists(os.path.join(default_local, "model.safetensors")):
                model_path = default_local

        self.model_name = model_name
        self.model_path = model_path
        self.threshold = threshold
        self.verify_checkpoint = verify_checkpoint
        self._model = None
        self._tokenizer = None
        self._fallback = HeuristicDetector()
        self._using_fallback = True

        # Try loading the transformer model; silently fall back on failure.
        self._try_load()

    # ── model loading ──────────────────────────────────────────────────

    def _try_load(self) -> None:
        """Attempt to load the transformer model + tokenizer."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
            import torch  # type: ignore

            path = self.model_path or self.model_name
            if self.verify_checkpoint and os.path.exists(os.path.join(path, "model.safetensors")):
                actual_sha = compute_safetensors_sha256(path)
                if actual_sha != EXPECTED_CHECKPOINT_SHA256:
                    log.warning(
                        "SECURITY GATE: Checkpoint at %s has SHA-256 %s, which does not match "
                        "expected publication checkpoint %s. Refusing to load legacy/unverified weights.",
                        path, actual_sha, EXPECTED_CHECKPOINT_SHA256
                    )
                    self._using_fallback = True
                    return

            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForSequenceClassification.from_pretrained(path)
            self._model.eval()
            self._using_fallback = False
            log.info("Loaded classifier model from %s (SHA-256 verified)", path)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Could not load transformer model (%s); using heuristic fallback.",
                exc,
            )
            self._using_fallback = True

    def load_model(self, path: str) -> None:
        """Load fine-tuned weights from *path*.

        Raises:
            RuntimeError: If loading fails.
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForSequenceClassification.from_pretrained(path)
            self._model.eval()
            self._using_fallback = False
            log.info("Loaded fine-tuned model from %s", path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {path}: {exc}") from exc

    # ── prediction ─────────────────────────────────────────────────────

    def predict(self, text: str) -> tuple[bool, float]:
        """Return ``(is_injection, confidence)`` for a single text.

        Uses the transformer model if available, otherwise the heuristic
        fallback.
        """
        if self._using_fallback:
            return self._fallback.detect(text)
        return self._predict_transformer(text)

    def predict_batch(self, texts: list[str]) -> list[tuple[bool, float]]:
        """Batch prediction over a list of texts."""
        if self._using_fallback:
            return [self._fallback.detect(t) for t in texts]
        return self._predict_transformer_batch(texts)

    # ── transformer internals ──────────────────────────────────────────

    def _predict_transformer(self, text: str) -> tuple[bool, float]:
        import torch  # type: ignore

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        prob = torch.softmax(logits, dim=-1)[0, 1].item()  # P(injection)
        return (prob >= self.threshold, prob)

    def _predict_transformer_batch(
        self, texts: list[str]
    ) -> list[tuple[bool, float]]:
        import torch  # type: ignore

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
        return [(p >= self.threshold, p) for p in probs]
