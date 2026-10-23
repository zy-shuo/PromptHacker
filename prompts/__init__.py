"""Prompt templates for PromptHacker agents (Appendix D)."""

from prompts.shared_system import SHARED_SYSTEM_PROMPT
from prompts.planner import PLANNER_PROMPT_WITH_MEMORY, PLANNER_PROMPT_WITHOUT_MEMORY
from prompts.composer import COMPOSER_PROMPT
from prompts.evaluator import EVALUATOR_PROMPT
from prompts.reflector import REFLECTOR_PROMPT
from prompts.memory_summarizer import QUERY_FAILURE_SUMMARIZER_PROMPT

__all__ = [
    "SHARED_SYSTEM_PROMPT",
    "PLANNER_PROMPT_WITH_MEMORY",
    "PLANNER_PROMPT_WITHOUT_MEMORY",
    "COMPOSER_PROMPT",
    "EVALUATOR_PROMPT",
    "REFLECTOR_PROMPT",
    "QUERY_FAILURE_SUMMARIZER_PROMPT",
]
