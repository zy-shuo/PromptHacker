"""
Strategy-level memory (Appendix C.2).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.types import AttemptRecord, ImprovementNote
from memory.embedder import Embedder, default_embedder
from memory.store import load_json, save_json


class StrategyMemoryStore:
    """Index records by combination; retrieve notes by failure descriptions."""

    def __init__(
        self,
        path: str = "memory/strategy_memory.json",
        embedder: Optional[Embedder] = None,
        similarity_threshold: float = 0.5,
        top_k: int = 3,
    ):
        self.path = path
        self.embedder = embedder if embedder is not None else default_embedder()
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.records: Dict[str, Dict[str, Any]] = load_json(path, {"records": {}}).get("records", {})

    def save(self) -> None:
        save_json(self.path, {"records": self.records})

    @staticmethod
    def _combo_key(strategies: List[str]) -> str:
        return json.dumps(sorted(strategies), ensure_ascii=False)

    def _ensure_record(self, strategies: List[str]) -> Dict[str, Any]:
        key = self._combo_key(strategies)
        if key not in self.records:
            self.records[key] = {
                "strategies": sorted(strategies),
                "performance_statistics": {
                    "usage_count": 0,
                    "success_count": 0,
                    "success_rate": 0.0,
                    "average_score": 0.0,
                },
                "improvement_notes": [],
            }
        return self.records[key]

    def retrieve(
        self,
        *,
        failure_type: Optional[str],
        failure_reason: Optional[str],
    ) -> List[str]:
        if (
            not self.records or not failure_type or self.top_k <= 0
            or not isinstance(failure_reason, str) or not failure_reason.strip()
        ):
            return []

        # Filter individual notes before any vector work. Legacy query text or
        # strategy-combination embeddings cannot substitute for a failure reason.
        eligible_notes = []
        for record in self.records.values():
            for note in record.get("improvement_notes", []):
                reason = note.get("failure_reason")
                advice = note.get("revision_advice")
                if (
                    note.get("failure_type") == failure_type
                    and isinstance(reason, str) and reason.strip()
                    and isinstance(advice, str) and advice.strip()
                ):
                    eligible_notes.append(note)
        if not eligible_notes:
            return []

        try:
            failure_vec = self.embedder.embed(failure_reason.strip())
        except Exception as exc:
            print(
                "[Strategy Memory] Failure-reason retrieval unavailable "
                f"({type(exc).__name__}); revising without memory"
            )
            return []

        candidates = []
        cache_updated = False
        cache_key = getattr(self.embedder, "cache_key", None)
        for note in eligible_notes:
            note_vec = note.get("failure_reason_embedding")
            if (not isinstance(note_vec, list) or not note_vec or len(note_vec) != len(failure_vec)
                    or (cache_key and note.get("failure_reason_embedding_model") != cache_key)):
                # Backfill notes with real diagnoses but no usable cached vector.
                try:
                    note_vec = self.embedder.embed(note["failure_reason"].strip())
                except Exception as exc:
                    print(
                        "[Strategy Memory] Skipping an unavailable note embedding "
                        f"({type(exc).__name__})"
                    )
                    continue
                note["failure_reason_embedding"] = note_vec
                note["failure_reason_embedding_model"] = cache_key
                cache_updated = True

            similarity = self.embedder.cosine_similarity(failure_vec, note_vec)
            if similarity >= self.similarity_threshold:
                candidates.append((
                    similarity,
                    bool(note.get("success_after_revision", False)),
                    float(note.get("revised_final_score") or 0.0),
                    note,
                ))

        if cache_updated:
            self.save()

        # Lexicographic ordering keeps similarity primary; outcome signals only
        # break similarity ties instead of overpowering semantic relevance.
        candidates.sort(key=lambda item: item[:3], reverse=True)
        advice: List[str] = []
        for _, _, _, note in candidates:
            advice_text = note["revision_advice"].strip()
            if advice_text not in advice:
                advice.append(advice_text)
            if len(advice) >= self.top_k:
                break
        return advice

    def update_from_attempts(self, query: str, attempts: List[AttemptRecord]) -> None:
        for i, attempt in enumerate(attempts):
            strategies = attempt.program.strategies
            record = self._ensure_record(strategies)
            stats = record["performance_statistics"]
            stats["usage_count"] += 1
            stats["average_score"] = (
                stats["average_score"] * (stats["usage_count"] - 1) + attempt.evaluation.final_score
            ) / stats["usage_count"]
            if attempt.evaluation.success_label:
                stats["success_count"] += 1
            stats["success_rate"] = stats["success_count"] / stats["usage_count"]

            if i == 0 or attempt.program.source != "reflector":
                continue
            prev = attempts[i - 1]
            if prev.evaluation.success_label or not (
                attempt.evaluation.success_label
                or attempt.evaluation.final_score > prev.evaluation.final_score
            ):
                continue

            # The note describes how the previous failed program was revised.
            # Store it under that program's combination, not the revised one.
            reason = (prev.evaluation.failure_reason or "").strip()
            reason_embedding = None
            if reason:
                try:
                    reason_embedding = self.embedder.embed(reason)
                except Exception as exc:
                    print(
                        "[Strategy Memory] Retaining revision note without an embedding "
                        f"({type(exc).__name__}); retrieval will retry when eligible"
                    )
            note = ImprovementNote(
                source_query_id=query[:64],
                failure_type=prev.evaluation.failure_type or "unknown",
                failure_reason=reason,
                failure_reason_embedding=reason_embedding,
                revision_advice=attempt.program.revision_note or attempt.program.brief_guidance,
                revised_final_score=attempt.evaluation.final_score,
                success_after_revision=attempt.evaluation.success_label,
                failure_reason_embedding_model=(
                    getattr(self.embedder, "cache_key", None) if reason_embedding else None
                ),
            )
            source_record = self._ensure_record(prev.program.strategies)
            source_record["improvement_notes"].append(note.to_dict())

        self.save()
