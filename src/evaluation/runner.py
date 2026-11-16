"""Evaluation Runner.

Orchestrates running benchmark cases through an LLM agent harness, executing tools
against the deterministic sandbox, and computing metrics across test runs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from src.agent.cache import ResponseCache
from src.agent.harness import AgentHarness
from src.agent.providers.base import BaseProvider
from src.benchmark.checker import OutcomeChecker
from src.sandbox.executor import ToolExecutor
from src.sandbox.state import SandboxState
from src.types import BenchmarkCase, DefenseVerdict, EvalResult, ToolCall

logger = logging.getLogger(__name__)


class EvalRunner:
    """Orchestrates benchmark evaluation runs."""

    def __init__(
        self,
        output_dir: str = "results",
        cache_path: str = "data/response_cache.db",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache = ResponseCache(cache_path)
        self.state = SandboxState()
        self.executor = ToolExecutor(self.state)
        self.tools = self.executor.get_all_tool_definitions()

    def run_case(
        self,
        case: BenchmarkCase,
        provider: BaseProvider,
        defense_pipeline: Callable[[ToolCall, Any], DefenseVerdict] | None = None,
        max_turns: int = 10,
    ) -> EvalResult:
        """Run a single benchmark case."""
        # 1. Reset sandbox to deterministic seed state
        self.executor.reset()

        # If the case specifies an injection to be placed in state
        if not case.is_benign and case.injection_payload:
            self._inject_into_sandbox(case)

        harness = AgentHarness(
            provider=provider,
            tool_executor=self.executor,
            tools=self.tools,
            defense_pipeline=defense_pipeline,
            config={"max_turns": max_turns},
            cache=self.cache,
        )

        start_time = time.perf_counter()
        error = None
        run_record = None

        try:
            run_record = harness.run(case.user_task, max_turns=max_turns)
        except Exception as exc:
            error = str(exc)
            logger.error("Error evaluating case %s: %s", case.id, exc)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        tool_calls = run_record.tool_calls if run_record else []
        state_snap = self.state.get_state_snapshot()

        attack_succeeded = OutcomeChecker.check_attack_success(
            case, tool_calls, state_snap
        )
        utility_preserved = OutcomeChecker.check_utility(
            case, tool_calls, state_snap
        )

        last_verdict = (
            run_record.defense_verdicts[-1]
            if run_record and run_record.defense_verdicts
            else None
        )

        defense_name = "none"
        if defense_pipeline:
            defense_name = getattr(defense_pipeline, "name", "defense")

        return EvalResult(
            case_id=case.id,
            model=getattr(provider, "_model", "unknown"),
            defense=defense_name,
            attack_succeeded=attack_succeeded,
            utility_preserved=utility_preserved,
            tool_calls_made=tool_calls,
            latency_ms=latency_ms,
            defense_verdict=last_verdict,
            error=error,
        )

    def _inject_into_sandbox(self, case: BenchmarkCase) -> None:
        """Seeds the injection payload into the appropriate sandbox tool state."""
        channel = case.attack_metadata.channel.value
        payload = case.injection_payload

        # Inject into emails, web pages, or files depending on channel
        conn = self.state.conn
        if "email" in channel:
            conn.execute(
                "INSERT INTO emails (sender, recipient, subject, body, read, timestamp) "
                "VALUES ('urgent@partner.org', 'alice@example.com', 'Meeting Notes', ?, 0, '2026-11-16T10:00:00')",
                (payload,),
            )
            conn.commit()
        elif "web" in channel:
            conn.execute(
                "UPDATE web_pages SET content = content || '\n' || ? WHERE url LIKE '%company%'",
                (payload,),
            )
            conn.commit()
        elif "document" in channel or "file" in channel:
            conn.execute(
                "INSERT INTO files (name, content, owner, created_at) VALUES ('project_brief.txt', ?, 'alice', '2026-11-16T10:00:00')",
                (payload,),
            )
            conn.commit()

    def run_benchmark(
        self,
        cases: list[BenchmarkCase],
        provider: BaseProvider,
        defense_pipeline: Any | None = None,
        max_turns: int = 10,
        show_progress: bool = True,
    ) -> list[EvalResult]:
        """Runs an evaluation suite across the given benchmark cases."""
        results: list[EvalResult] = []
        iterator = tqdm(cases, desc="Evaluating cases") if show_progress else cases

        for case in iterator:
            res = self.run_case(case, provider, defense_pipeline, max_turns=max_turns)
            results.append(res)

        return results

    def save_results(self, results: list[EvalResult], filename: str) -> Path:
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        return out_path
