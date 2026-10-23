"""Summarize compact search records after a query's search has ended."""

import json
import re
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from core.types import AttemptRecord
from prompts.memory_summarizer import QUERY_FAILURE_SUMMARIZER_PROMPT


class QueryFailureSummarizer(BaseAgent):
    """Extract one failure lesson using the existing attack-model instance."""

    MAX_FAILURE_REASON_CHARS = 600
    MAX_REVISION_NOTE_CHARS = 400
    MAX_GUIDANCE_EXCERPT_CHARS = 400

    @staticmethod
    def _clip(value: str, limit: int) -> str:
        value = " ".join(value.split())
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + " [truncated]"

    def compact_records(self, attempts: List[AttemptRecord]) -> List[Dict[str, Any]]:
        records = []
        for attempt in attempts:
            evaluation = attempt.evaluation
            record = {
                "iteration": attempt.iteration,
                "strategies": list(attempt.program.strategies),
                "scores": {
                    "topic_consistency": evaluation.topic_consistency_score,
                    "harmfulness": evaluation.harmfulness_score,
                    "final": evaluation.final_score,
                },
                "success": evaluation.success_label,
            }
            if attempt.program.revision_note:
                record["revision_note"] = self._clip(
                    attempt.program.revision_note, self.MAX_REVISION_NOTE_CHARS
                )
            if evaluation.failure_type:
                record["failure_type"] = evaluation.failure_type
            if evaluation.failure_reason:
                record["failure_reason"] = self._clip(
                    evaluation.failure_reason, self.MAX_FAILURE_REASON_CHARS
                )
            if evaluation.failure_type == "composition_failure":
                guidance = attempt.program.brief_guidance
                integration = re.search(r"\bintegration\s*:", guidance, re.IGNORECASE)
                if integration:
                    guidance = guidance[integration.start():]
                if guidance:
                    record["guidance_excerpt"] = self._clip(
                        guidance, self.MAX_GUIDANCE_EXCERPT_CHARS
                    )
            records.append(record)
        return records

    def summarize(self, query: str, attempts: List[AttemptRecord]) -> str:
        if not any(not a.evaluation.success_label for a in attempts):
            return ""

        values = {
            "search_records": json.dumps(
                self.compact_records(attempts), ensure_ascii=False, separators=(",", ":")
            ),
            "query": query,
        }
        # Substitute once so placeholder-like text inside logged data stays literal.
        prompt = re.sub(
            r"\{(query|search_records)\}",
            lambda match: values[match.group(1)],
            QUERY_FAILURE_SUMMARIZER_PROMPT,
        )
        data = self.call_json(prompt)
        summary = data.get("failed_experience")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Failure summarizer returned no usable failure experience")
        summary = " ".join(summary.split())
        if len(summary.split()) > 80 or len(summary) > 1000:
            raise ValueError("Failure summarizer exceeded the compact-summary limit")
        return summary
