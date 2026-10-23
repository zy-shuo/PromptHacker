"""Planner agent (Appendix D.2)."""

from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent, render_prompt
from core.types import StrategyProgram
from prompts.planner import PLANNER_PROMPT_WITH_MEMORY, PLANNER_PROMPT_WITHOUT_MEMORY
from taxonomy.loader import format_hierarchy_for_prompt, validate_strategies


class PlannerAgent(BaseAgent):
    """Selects atomic strategies and produces an initial Strategy Program."""

    def plan(
        self,
        query: str,
        retrieved_experience: Optional[Dict[str, Any]] = None,
    ) -> StrategyProgram:
        hierarchy_text = format_hierarchy_for_prompt()

        if retrieved_experience and retrieved_experience.get("best_strategy_pattern"):
            pattern = retrieved_experience["best_strategy_pattern"]
            prompt = render_prompt(
                PLANNER_PROMPT_WITH_MEMORY,
                query=query,
                strategy_hierarchy=hierarchy_text,
                best_pattern_strategies=", ".join(pattern.get("strategies", [])),
                best_pattern_description=pattern.get("description", ""),
                failed_experience=retrieved_experience.get("failed_experience", ""),
            )
        else:
            prompt = render_prompt(
                PLANNER_PROMPT_WITHOUT_MEMORY,
                query=query,
                strategy_hierarchy=hierarchy_text,
            )

        data = self.call_json(prompt)
        strategies = validate_strategies(data.get("strategies", []))
        if not strategies:
            raise ValueError("Planner returned no valid strategies")

        return StrategyProgram(
            strategies=strategies,
            brief_guidance=data.get("brief_guidance", ""),
            rationale=data.get("rationale", ""),
            source="planner",
            parent_program_id=None,
            revision_note=None,
        )
