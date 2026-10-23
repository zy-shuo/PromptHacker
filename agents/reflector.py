"""Reflector agent (Appendix D.5)."""

from typing import List, Optional

from agents.base_agent import BaseAgent, render_prompt
from core.types import EvaluationResult, StrategyProgram
from prompts.reflector import REFLECTOR_PROMPT
from taxonomy.loader import format_hierarchy_for_prompt, validate_strategies


class ReflectorAgent(BaseAgent):
    """Revises a failed Strategy Program for the next iteration."""

    def revise(
        self,
        query: str,
        failed_program: StrategyProgram,
        previous_prompt: str,
        evaluation: EvaluationResult,
        memory_revision_advice: Optional[List[str]] = None,
    ) -> StrategyProgram:
        advice_text = "None"
        if memory_revision_advice:
            advice_text = "\n".join(f"- {a}" for a in memory_revision_advice)

        user_prompt = render_prompt(
            REFLECTOR_PROMPT,
            query=query,
            failed_strategies=", ".join(failed_program.strategies),
            failed_brief_guidance=failed_program.brief_guidance,
            failed_rationale=failed_program.rationale,
            previous_prompt=previous_prompt,
            topic_consistency_score=evaluation.topic_consistency_score,
            harmfulness_score=evaluation.raw_harmfulness_score,
            failure_type=evaluation.failure_type,
            failure_reason=evaluation.failure_reason or "",
            strategy_hierarchy=format_hierarchy_for_prompt(),
            memory_revision_advice=advice_text,
        )

        data = self.call_json(user_prompt)
        strategies = validate_strategies(data.get("strategies", []))
        if not strategies:
            raise ValueError("Reflector returned no valid strategies")

        return StrategyProgram(
            strategies=strategies,
            brief_guidance=data.get("brief_guidance", ""),
            rationale=data.get("rationale", ""),
            source="reflector",
            parent_program_id=failed_program.program_id,
            revision_note=data.get("revision_note"),
        )
