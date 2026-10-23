"""
Base agent with shared system prompt and JSON parsing utilities.
"""

import json
import re
from typing import Any, Dict, List, Optional

from prompts.shared_system import SHARED_SYSTEM_PROMPT


def render_prompt(template: str, **kwargs: Any) -> str:
    """
    Replace only explicit {key} placeholders.

    Unlike str.format(), this leaves JSON examples with braces untouched.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


class BaseAgent:
    """Common LLM agent wrapper for PromptHacker modules."""

    def __init__(self, llm, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    def _build_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        # Try to locate outermost JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text)

    def call_json(self, user_prompt: str) -> Dict[str, Any]:
        """Call LLM and parse a JSON object with retries."""
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.llm.generate(self._build_messages(user_prompt))
                if not response or response == "$ERROR$":
                    raise ValueError("LLM returned empty or error response")
                return self._extract_json(response)
            except Exception as exc:
                last_error = exc
                print(f"[{self.__class__.__name__}] JSON parse attempt {attempt + 1}/{self.max_retries} failed: {exc}")
        raise RuntimeError(
            f"{self.__class__.__name__} failed after {self.max_retries} attempts: {last_error}"
        )
