"""
Query-level memory (Appendix C.1).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.types import AttemptRecord, QueryExperienceSummary
from memory.embedder import Embedder, default_embedder
from memory.store import load_json, save_json

if TYPE_CHECKING:
    from agents.memory_summarizer import QueryFailureSummarizer


class QueryMemoryStore:
    """Stores and retrieves query-level search experience."""

    def __init__(
        self,
        path: str = "memory/query_memory.json",
        embedder: Optional[Embedder] = None,
        similarity_threshold: float = 0.7,
        top_k: int = 1,
        summarizer: Optional["QueryFailureSummarizer"] = None,
    ):
        self.path = path
        self.embedder = embedder if embedder is not None else default_embedder()
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.summarizer = summarizer
        self.entries: List[Dict[str, Any]] = load_json(path, {"entries": []}).get("entries", [])

    def save(self) -> None:
        save_json(self.path, {"entries": self.entries})

    def retrieve(self, query: str) -> Optional[Dict[str, Any]]:
        if not self.entries:
            return None

        query_vec = self.embedder.embed(query)
        cache_key = getattr(self.embedder, "cache_key", None)
        cache_updated = False
        scored: List[tuple[float, Dict[str, Any]]] = []
        for entry in self.entries:
            rep = entry.get("query_representation", "")
            if not rep:
                continue
            entry_vec = entry.get("embedding")
            if (not isinstance(entry_vec, list) or not entry_vec
                    or len(entry_vec) != len(query_vec)
                    or (cache_key and entry.get("embedding_model") != cache_key)):
                entry_vec = self.embedder.embed(rep)
                entry["embedding"] = entry_vec
                entry["embedding_model"] = cache_key
                cache_updated = True
            sim = self.embedder.cosine_similarity(query_vec, entry_vec)
            scored.append((sim, entry))

        if cache_updated:
            self.save()

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored or scored[0][0] < self.similarity_threshold:
            return None

        best_entry = scored[0][1]
        summary = best_entry.get("query_experience_summary", {})
        return {
            "similarity": scored[0][0],
            "best_strategy_pattern": summary.get("best_strategy_pattern"),
            "failed_experience": summary.get("failed_experience", ""),
        }

    def _summarize_attempts(
        self, attempts: List[AttemptRecord], query: str = ""
    ) -> QueryExperienceSummary:
        best = max(attempts, key=lambda a: a.evaluation.final_score)
        failed_attempts = [a for a in attempts if not a.evaluation.success_label]
        failed_experience = ""
        if failed_attempts:
            # Preserve diagnoses if no summarizer is configured or its call fails.
            failed_experience = "; ".join(
                dict.fromkeys(
                    f"{a.evaluation.failure_type or 'unspecified_failure'}: "
                    f"{a.evaluation.failure_reason or 'No detailed diagnosis recorded.'}"
                    for a in failed_attempts
                )
            )
            if self.summarizer is not None:
                try:
                    summary = self.summarizer.summarize(query, attempts)
                    if not isinstance(summary, str) or not summary.strip():
                        raise ValueError("Empty failure-experience summary")
                    failed_experience = summary.strip()
                except Exception as exc:
                    print(
                        "[Query Memory] Failure summarization unavailable; "
                        f"preserving recorded diagnoses ({type(exc).__name__})"
                    )

        return QueryExperienceSummary(
            best_strategy_pattern={
                "strategies": best.program.strategies,
                "description": best.program.brief_guidance,
                "final_score": best.evaluation.final_score,
            },
            failed_experience=failed_experience,
            success_label=any(a.evaluation.success_label for a in attempts),
        )

    def update(self, query: str, attempts: List[AttemptRecord]) -> None:
        if not attempts:
            return

        best = max(attempts, key=lambda a: a.evaluation.final_score)
        summary = self._summarize_attempts(attempts, query=query)
        query_id = f"q_{uuid.uuid4().hex[:12]}"

        entry = {
            "query_id": query_id,
            "query_representation": query,
            "embedding": self.embedder.embed(query),
            "embedding_model": getattr(self.embedder, "cache_key", None),
            "attempts": [a.to_dict() for a in attempts],
            "best_attempt_id": best.attempt_id,
            "best_program_id": best.program.program_id,
            "best_final_score": best.evaluation.final_score,
            "success_label": summary.success_label,
            "total_iterations": len(attempts),
            "query_experience_summary": summary.to_dict(),
            "updated_at": time.time(),
        }
        self.entries.append(entry)
        self.save()
