"""Composer agent (Appendix D.3)."""

from core.types import PromptCandidate, StrategyProgram
from agents.base_agent import BaseAgent, render_prompt
from prompts.composer import COMPOSER_PROMPT


class ComposerAgent(BaseAgent):
    """Instantiates a Strategy Program into a concrete jailbreak prompt."""

    def compose(self, query: str, program: StrategyProgram) -> PromptCandidate:
        prompt = render_prompt(
            COMPOSER_PROMPT,
            query=query,
            strategies=", ".join(program.strategies),
            brief_guidance=program.brief_guidance,
            rationale=program.rationale,
        )
        data = self.call_json(prompt)
        prompt_text = data.get("prompt_text", "").strip()
        if not prompt_text:
            raise ValueError("Composer returned empty prompt_text")

        return PromptCandidate(
            prompt_id=f"prompt_{program.program_id}",
            program_id=program.program_id,
            prompt_text=prompt_text,
        )
