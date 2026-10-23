"""
PromptHacker orchestrator — Iterative Reflective Search Loop.
Planning -> Composition -> Evaluation -> Revision
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from agents.composer import ComposerAgent
from agents.evaluator import EvaluatorAgent
from agents.planner import PlannerAgent
from agents.reflector import ReflectorAgent
from agents.memory_summarizer import QueryFailureSummarizer
from core.config import PromptHackerConfig, DEFAULT_CONFIG
from core.model_config import ModelRegistry
from core.types import AttemptRecord, HackResult, StrategyProgram
from memory.query_memory import QueryMemoryStore
from memory.strategy_memory import StrategyMemoryStore


class PromptHackerOrchestrator:
    """Main search controller for PromptHacker."""

    def __init__(
        self,
        aux_llm,
        target_llm,
        config: Optional[PromptHackerConfig] = None,
        query_memory: Optional[QueryMemoryStore] = None,
        strategy_memory: Optional[StrategyMemoryStore] = None,
        embedder=None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.aux_llm = aux_llm
        self.target_llm = target_llm

        self.planner = PlannerAgent(aux_llm, max_retries=self.config.agent_max_retries)
        self.composer = ComposerAgent(aux_llm, max_retries=self.config.agent_max_retries)
        self.evaluator = EvaluatorAgent(aux_llm, max_retries=self.config.agent_max_retries)
        self.reflector = ReflectorAgent(aux_llm, max_retries=self.config.agent_max_retries)
        self.memory_summarizer = QueryFailureSummarizer(
            aux_llm, max_retries=self.config.agent_max_retries
        )

        if embedder is None and (query_memory is None or strategy_memory is None):
            embedder = ModelRegistry("embedding", self.config.embedding_config_path).create_embedder(
                self.config.embedding_model, self.config.embedding_path
            )
        self.query_memory = query_memory if query_memory is not None else QueryMemoryStore(
            path=self.config.query_memory_path,
            embedder=embedder,
            similarity_threshold=self.config.query_memory_similarity_threshold,
            top_k=self.config.query_memory_top_k,
            summarizer=self.memory_summarizer,
        )
        if self.query_memory.summarizer is None:
            self.query_memory.summarizer = self.memory_summarizer
        self.strategy_memory = strategy_memory if strategy_memory is not None else StrategyMemoryStore(
            path=self.config.strategy_memory_path,
            embedder=embedder,
            similarity_threshold=self.config.strategy_memory_similarity_threshold,
            top_k=self.config.strategy_memory_top_k,
        )

    def hack(self, target_query: str, budget: Optional[int] = None) -> HackResult:
        budget = budget or self.config.budget
        print(f"[PromptHacker] Starting search, budget={budget}")

        # Stage 1: Planning (once)
        retrieved = self.query_memory.retrieve(target_query)
        if retrieved:
            print(f"[PromptHacker] Query memory hit, similarity={retrieved.get('similarity', 0):.3f}")
        else:
            print("[PromptHacker] No query memory hit, planning from scratch")

        program = self.planner.plan(target_query, retrieved_experience=retrieved)
        print(f"[PromptHacker] Initial program: {program.strategies}")

        attempts: List[AttemptRecord] = []
        failure_types_seen: List[str] = []

        # Iterative Reflective Search Loop
        for t in range(budget):
            print(f"\n{'=' * 40}\nIteration {t + 1}/{budget}\n{'=' * 40}")

            # Stage 2: Composition
            prompt = self.composer.compose(target_query, program)
            print(f"[Composer] Prompt generated ({len(prompt.prompt_text)} chars)")

            # Target model call
            target_messages = [{"role": "user", "content": prompt.prompt_text}]
            target_response = self.target_llm.generate(target_messages)
            if not target_response or target_response == "$ERROR$":
                target_response = "I'm sorry, I can't generate a response for that prompt."
            print(f"[Target] Response preview: {target_response[:200]}...")

            # Stage 3: Evaluation
            evaluation = self.evaluator.evaluate(target_query, prompt, target_response)
            print(
                f"[Evaluator] tc={evaluation.topic_consistency_score}, "
                f"harm={evaluation.raw_harmfulness_score}, "
                f"final={evaluation.final_score:.3f}, "
                f"failure={evaluation.failure_type}"
            )

            attempt = AttemptRecord(
                attempt_id=f"att_{uuid.uuid4().hex[:10]}",
                iteration=t + 1,
                program=program,
                prompt=prompt,
                target_response=target_response,
                evaluation=evaluation,
            )
            attempts.append(attempt)

            if evaluation.failure_type and evaluation.failure_type not in failure_types_seen:
                failure_types_seen.append(evaluation.failure_type)

            if evaluation.success_label:
                print(f"[PromptHacker] Success at iteration {t + 1}")
                break

            if t == budget - 1:
                print("[PromptHacker] Budget exhausted")
                break

            # Stage 4: Revision
            advice = self.strategy_memory.retrieve(
                failure_type=evaluation.failure_type,
                failure_reason=evaluation.failure_reason,
            )
            if advice:
                print(f"[Strategy Memory] Retrieved {len(advice)} revision advice entries")

            program = self.reflector.revise(
                query=target_query,
                failed_program=program,
                previous_prompt=prompt.prompt_text,
                evaluation=evaluation,
                memory_revision_advice=advice or None,
            )
            print(f"[Reflector] Revised program: {program.strategies}")
            print(f"[Reflector] revision_note: {program.revision_note}")

        # Memory update
        self.query_memory.update(target_query, attempts)
        self.strategy_memory.update_from_attempts(target_query, attempts)

        best_attempt = max(attempts, key=lambda a: a.evaluation.final_score)
        is_successful = best_attempt.evaluation.final_score >= self.config.success_threshold

        search_stats: Dict[str, Any] = {
            "budget": budget,
            "attempts": len(attempts),
            "best_score": best_attempt.evaluation.final_score,
            "best_strategies": best_attempt.program.strategies,
            "query_memory_used": retrieved is not None,
        }

        return HackResult(
            target_query=target_query,
            jailbreak_query=best_attempt.prompt.prompt_text,
            target_response=best_attempt.target_response,
            success_score=best_attempt.evaluation.final_score,
            is_successful=is_successful,
            iterations_used=len(attempts),
            best_strategy_program=best_attempt.program,
            attempt_history=attempts,
            failure_types_seen=failure_types_seen,
            search_stats=search_stats,
        )
