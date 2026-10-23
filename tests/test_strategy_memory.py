"""Offline checks for failure-reason retrieval and effective revision notes."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from core.types import AttemptRecord, EvaluationResult, PromptCandidate, StrategyProgram
from memory.embedder import OpenAIEmbedder
from memory.strategy_memory import StrategyMemoryStore


def vector(similarity):
    return [similarity, math.sqrt(1.0 - similarity ** 2)]


class RecordingEmbedder:
    cosine_similarity = staticmethod(OpenAIEmbedder.cosine_similarity)

    def __init__(self, vectors=None):
        self.vectors = vectors or {"current diagnosis": [1.0, 0.0]}
        self.calls = []
        self.unavailable = set()

    def embed(self, text):
        self.calls.append(text)
        if text in self.unavailable:
            raise RuntimeError("Embedding temporarily unavailable")
        return list(self.vectors[text])


def note(reason, similarity, advice, failure_type="off_topic", success=False, score=.5):
    return {
        "failure_type": failure_type,
        "failure_reason": reason,
        "failure_reason_embedding": vector(similarity),
        "revision_advice": advice,
        "success_after_revision": success,
        "revised_final_score": score,
    }


def attempt(iteration, strategies, score, success=False, reason="A different topic was discussed."):
    program = StrategyProgram(
        strategies=strategies, brief_guidance="Fixture coordination guidance.",
        rationale="Fixture rationale.",
        source="planner" if iteration == 1 else "reflector",
        revision_note=None if iteration == 1 else f"Recorded modification {iteration}.",
    )
    return AttemptRecord(
        attempt_id=f"attempt_{iteration}", iteration=iteration, program=program,
        prompt=PromptCandidate(f"prompt_{iteration}", program.program_id, "Fixture prompt."),
        target_response="Fixture response.",
        evaluation=EvaluationResult(
            topic_consistency_score=1, raw_harmfulness_score=1 + round(score * 4),
            harmfulness_score=score, final_score=score,
            failure_type=None if success else "off_topic",
            failure_reason=None if success else reason,
            success_label=success,
        ),
    )


class StrategyMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = str(Path(self.temporary.name) / "strategy_memory.json")
        self.embedder = RecordingEmbedder()
        self.store = StrategyMemoryStore(self.path, embedder=self.embedder)

    def record(self, strategies, notes):
        record = self.store._ensure_record(strategies)
        record["improvement_notes"] = notes
        return record

    def retrieve(self, reason="current diagnosis", failure_type="off_topic"):
        return self.store.retrieve(failure_type=failure_type, failure_reason=reason)

    def test_ranks_individual_notes_across_combinations_and_applies_threshold(self):
        self.record(["A"], [note("first", .95, "first advice"),
                             note("below threshold", .49, "must be excluded"),
                             note("third", .75, "third advice")])
        self.record(["B"], [note("second", .85, "second advice"),
                             note("fourth", .65, "fourth advice")])
        self.record(["C"], [note("wrong failure", 1.0, "wrong advice", failure_type="refusal")])
        self.assertEqual(self.retrieve(), ["first advice", "second advice", "third advice"])
        self.assertEqual(self.embedder.calls, ["current diagnosis"])

    def test_outcome_scores_cannot_override_higher_similarity(self):
        self.record(["A"], [note("lower", .9, "lower similarity", success=True, score=1.0),
                             note("higher", .95, "higher similarity", success=False, score=.25)])
        self.assertEqual(self.retrieve(), ["higher similarity", "lower similarity"])

    def test_success_and_score_break_similarity_ties(self):
        self.record(["A"], [note("unsuccessful", .9, "unsuccessful", success=False, score=1.0),
                             note("successful low", .9, "successful low", success=True, score=.5),
                             note("successful high", .9, "successful high", success=True, score=.75)])
        self.assertEqual(self.retrieve(), ["successful high", "successful low", "unsuccessful"])

    def test_exact_threshold_is_included_and_duplicate_advice_is_not_repeated(self):
        self.record(["A"], [note("close", .95, "shared advice"),
                             note("same advice", .9, "shared advice"),
                             note("threshold", .5, "threshold advice")])
        self.assertEqual(self.retrieve(), ["shared advice", "threshold advice"])

    def test_filter_and_missing_input_skip_vector_calls(self):
        self.record(["A"], [note("wrong failure", .95, "advice", failure_type="refusal")])
        self.assertEqual(self.retrieve(), [])
        self.assertEqual(self.retrieve(reason=""), [])
        self.assertEqual(self.retrieve(failure_type=None), [])
        self.assertEqual(self.embedder.calls, [])

    def test_legacy_missing_reason_is_preserved_without_query_or_combo_matching(self):
        legacy = {"source_query_representation": "An old query.",
                  "failure_type": "off_topic", "revision_advice": "Legacy advice.",
                  "revised_final_score": 1.0, "success_after_revision": True}
        record = self.record(["Legacy strategy"], [legacy])
        record["embedding"] = [1.0, 0.0]
        self.store.save()
        original = Path(self.path).read_text(encoding="utf-8")
        self.assertEqual(self.retrieve(), [])
        self.assertEqual(self.embedder.calls, [])
        self.assertEqual(Path(self.path).read_text(encoding="utf-8"), original)
        self.assertEqual(record["improvement_notes"], [legacy])

    def test_missing_reason_vector_is_backfilled_persisted_and_reused(self):
        self.embedder.vectors["historical diagnosis"] = vector(.8)
        historical = note("historical diagnosis", .8, "Historical advice.")
        del historical["failure_reason_embedding"]
        record = self.record(["A"], [historical])
        record["embedding"] = [0.0, 1.0]  # Old combination vector is ignored.
        self.store.save()

        self.assertEqual(self.retrieve(), ["Historical advice."])
        self.assertEqual(self.embedder.calls, ["current diagnosis", "historical diagnosis"])
        saved = json.loads(Path(self.path).read_text(encoding="utf-8"))
        cached = next(iter(saved["records"].values()))["improvement_notes"][0]
        self.assertEqual(cached["failure_reason_embedding"], vector(.8))

        self.embedder.calls.clear()
        self.store = StrategyMemoryStore(self.path, embedder=self.embedder)
        self.assertEqual(self.retrieve(), ["Historical advice."])
        self.assertEqual(self.embedder.calls, ["current diagnosis"])

    def test_updates_cache_previous_failure_under_its_original_combination(self):
        self.embedder.vectors.update({"failure A": vector(.8), "failure B": vector(.9)})
        attempts = [attempt(1, ["A"], .25, reason="failure A"),
                    attempt(2, ["B"], .5, reason="failure B"),
                    attempt(3, ["C"], 1.0, success=True)]
        self.store.update_from_attempts("Query provenance only.", attempts)

        records = {tuple(r["strategies"]): r for r in self.store.records.values()}
        first = records[("A",)]["improvement_notes"][0]
        second = records[("B",)]["improvement_notes"][0]
        self.assertEqual(first["failure_reason"], "failure A")
        self.assertEqual(first["failure_reason_embedding"], vector(.8))
        self.assertEqual(first["revised_final_score"], .5)
        self.assertFalse(first["success_after_revision"])
        self.assertEqual(first["revision_advice"], attempts[1].program.revision_note)
        self.assertEqual(second["failure_reason"], "failure B")
        self.assertTrue(second["success_after_revision"])
        self.assertEqual(records[("C",)]["improvement_notes"], [])
        self.assertEqual(self.embedder.calls, ["failure A", "failure B"])
        self.assertNotIn("source_query_representation", first)
        for record in records.values():
            self.assertEqual(record["performance_statistics"]["usage_count"], 1)

    def test_non_improving_revisions_update_statistics_without_adding_notes(self):
        self.store.update_from_attempts("A query.", [attempt(1, ["A"], .5),
                                                    attempt(2, ["B"], .25),
                                                    attempt(3, ["B"], .25)])
        self.assertTrue(all(not r["improvement_notes"] for r in self.store.records.values()))
        record_b = self.store.records[self.store._combo_key(["B"])]
        self.assertEqual(record_b["performance_statistics"]["usage_count"], 2)
        self.assertEqual(self.embedder.calls, [])

    def test_embedding_outage_retains_note_and_later_retrieval_repairs_cache(self):
        self.embedder.vectors["past failure"] = vector(.8)
        self.embedder.unavailable.add("past failure")
        self.store.update_from_attempts("A query.", [
            attempt(1, ["A"], .25, reason="past failure"),
            attempt(2, ["B"], .5),
        ])
        stored_note = self.store.records[self.store._combo_key(["A"])]["improvement_notes"][0]
        self.assertIsNone(stored_note["failure_reason_embedding"])
        self.assertTrue(Path(self.path).is_file())

        self.embedder.unavailable.add("current diagnosis")
        self.assertEqual(self.retrieve(), [])
        self.embedder.unavailable.clear()
        self.assertEqual(self.retrieve(), ["Recorded modification 2."])
        self.assertEqual(stored_note["failure_reason_embedding"], vector(.8))


if __name__ == "__main__":
    unittest.main()
