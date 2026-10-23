"""Evaluator agent (Appendix D.4)."""

from core.types import EvaluationResult, PromptCandidate
from agents.base_agent import BaseAgent, render_prompt
from prompts.evaluator import EVALUATOR_PROMPT


class EvaluatorAgent(BaseAgent):
    """Evaluates topic consistency, harmfulness, and failure diagnosis."""

    def evaluate(
        self,
        query: str,
        prompt: PromptCandidate,
        target_response: str,
    ) -> EvaluationResult:
        user_prompt = render_prompt(
            EVALUATOR_PROMPT,
            query=query,
            prompt_text=prompt.prompt_text,
            target_response=target_response,
        )
        data = self.call_json(user_prompt)
        return EvaluationResult.from_evaluator_output(data)
