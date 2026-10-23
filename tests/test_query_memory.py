"""Offline checks for post-search failure summaries and memory integration."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.memory_summarizer import QueryFailureSummarizer
from core.config import PromptHackerConfig
from core.orchestrator import PromptHackerOrchestrator
from core.types import AttemptRecord, EvaluationResult, PromptCandidate, StrategyProgram
from memory.query_memory import QueryMemoryStore


class StubLLM:
    def __init__(self, output=None):
        self.output = output
        self.messages = []
        self.call_count = 0

    def generate(self, messages):
        self.call_count += 1
        self.messages.append(messages)
        if isinstance(self.output, Exception):
            raise self.output
        return json.dumps(self.output)


class StubEmbedder:
    def embed(self, text):
        return [1.0, 0.0]

    def cosine_similarity(self, first, second):
        return 1.0


class StubStrategyMemory:
    def __init__(self):
        self.updates = []
        self.retrieval_requests = []

    def retrieve(self, *, failure_type, failure_reason):
        self.retrieval_requests.append((failure_type, failure_reason))
        return []

    def update_from_attempts(self, query, attempts):
        self.updates.append((query, attempts))


def make_attempt(iteration, score=.25, success=False, failure_type="off_topic",
                 reason="The answer discusses a different topic.", revision_note=None,
                 guidance="Use the supplied task description."):
    program = StrategyProgram(
        strategies=[f"Strategy {iteration}"],
        brief_guidance=guidance,
        rationale="Fixture rationale.",
        source="planner" if iteration == 1 else "reflector",
        revision_note=revision_note,
    )
    return AttemptRecord(
        attempt_id=f"attempt_{iteration}", iteration=iteration, program=program,
        prompt=PromptCandidate(f"prompt_{iteration}", program.program_id,
                               f"RAW_PROMPT_{iteration}"),
        target_response=f"RAW_RESPONSE_{iteration}",
        evaluation=EvaluationResult(
            topic_consistency_score=1, raw_harmfulness_score=1 + round(4 * score),
            harmfulness_score=score, final_score=score,
            failure_type=None if success else failure_type,
            failure_reason=None if success else reason,
            success_label=success,
        ),
    )


class QueryMemorySummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def store(self, summarizer=None, name="memory.json"):
        return QueryMemoryStore(
            str(Path(self.temporary.name) / name), embedder=StubEmbedder(),
            summarizer=summarizer,
        )

    def test_update_preserves_history_best_attempt_and_retrievable_summary(self):
        lesson = "One attempt deviated from the topic; the later evaluation improved."
        llm = StubLLM({"failed_experience": lesson})
        store = self.store(QueryFailureSummarizer(llm, max_retries=1))
        attempts = [make_attempt(1), make_attempt(2, 1.0, True)]
        store.update("Summarize a paragraph.", attempts)

        self.assertEqual(llm.call_count, 1)
        entry = store.entries[0]
        self.assertEqual(entry["attempts"], [a.to_dict() for a in attempts])
        self.assertEqual(entry["best_attempt_id"], attempts[1].attempt_id)
        pattern = entry["query_experience_summary"]["best_strategy_pattern"]
        self.assertEqual(pattern["strategies"], attempts[1].program.strategies)
        self.assertEqual(pattern["description"], attempts[1].program.brief_guidance)
        self.assertEqual(pattern["final_score"], 1.0)
        self.assertEqual(store.retrieve("Summarize a paragraph.")["failed_experience"], lesson)
        self.assertIn("RAW_RESPONSE_1", Path(store.path).read_text(encoding="utf-8"))
        self.assertNotIn("RAW_RESPONSE_1", llm.messages[0][1]["content"])
        self.assertNotIn("RAW_PROMPT_1", llm.messages[0][1]["content"])

    def test_compact_input_bounds_diagnoses_and_keeps_later_outcomes(self):
        llm = StubLLM({"failed_experience": "The recorded coordination problem remained unresolved."})
        summarizer = QueryFailureSummarizer(llm, max_retries=1)
        attempts = [
            make_attempt(1, failure_type="composition_failure", reason="x" * 5000,
                         guidance="UNNEEDED_GUIDANCE " * 200 + "Integration: " + "z" * 5000),
            make_attempt(2, .5, revision_note="r" * 5000),
        ]
        summarizer.summarize("Literal {search_records} in query.", attempts)
        content = llm.messages[0][1]["content"]
        self.assertIn("Literal {search_records} in query.", content)
        records = json.loads(content.split("<search_records>\n", 1)[1].split("\n</search_records>", 1)[0])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["scores"]["final"], .5)
        self.assertTrue(records[0]["guidance_excerpt"].startswith("Integration:"))
        self.assertNotIn("UNNEEDED_GUIDANCE", records[0]["guidance_excerpt"])
        self.assertLess(len(records[0]["failure_reason"]), 650)
        self.assertLess(len(records[1]["revision_note"]), 450)
        self.assertLess(len(records[0]["guidance_excerpt"]), 450)
        self.assertNotIn("guidance_excerpt", records[1])

    def test_success_only_and_empty_search_skip_summary_calls(self):
        llm = StubLLM({"failed_experience": "Must not be requested."})
        summarizer = QueryFailureSummarizer(llm, max_retries=1)
        store = self.store(summarizer)
        store.update("Summarize a paragraph.", [])
        self.assertEqual(store.entries, [])
        store.update("Summarize a paragraph.", [make_attempt(1, 1.0, True)])
        self.assertEqual(llm.call_count, 0)
        self.assertEqual(store.entries[0]["query_experience_summary"]["failed_experience"], "")
        self.assertEqual(summarizer.summarize("Any query.", []), "")

    def test_failed_summary_falls_back_without_losing_memory(self):
        invalid_outputs = [RuntimeError("offline"), {"failed_experience": []},
                           {"failed_experience": ""}, {"failed_experience": "word " * 81}]
        for index, output in enumerate(invalid_outputs):
            with self.subTest(output_type=type(output).__name__):
                llm = StubLLM(output)
                store = self.store(QueryFailureSummarizer(llm, max_retries=1), f"fallback_{index}.json")
                store.update("Summarize a paragraph.", [make_attempt(1), make_attempt(2)])
                expected = "off_topic: The answer discusses a different topic."
                self.assertEqual(store.entries[0]["query_experience_summary"]["failed_experience"], expected)
                self.assertTrue(Path(store.path).is_file())
                self.assertEqual(llm.call_count, 1)

    def test_no_summarizer_preserves_deterministic_fallback(self):
        store = self.store()
        store.update("Summarize a paragraph.", [make_attempt(1)])
        self.assertEqual(store.entries[0]["best_final_score"], .25)
        self.assertTrue(store.entries[0]["query_experience_summary"]["failed_experience"].startswith("off_topic:"))

    def test_orchestrator_reuses_aux_model_and_summarizes_after_search(self):
        llm = StubLLM({"failed_experience": "The first attempt deviated from the topic."})
        target = StubLLM("A fixture response.")
        store = self.store()
        strategy_memory = StubStrategyMemory()
        orchestrator = PromptHackerOrchestrator(
            aux_llm=llm, target_llm=target,
            config=PromptHackerConfig(budget=2), query_memory=store,
            strategy_memory=strategy_memory,
        )
        self.assertIs(store.summarizer.llm, llm)
        self.assertIs(orchestrator.planner.llm, llm)
        first, last = make_attempt(1), make_attempt(2, 1.0, True)
        observed_target_calls = []
        original_summary = orchestrator.memory_summarizer.summarize

        def observe_summary(query, attempts):
            observed_target_calls.append(target.call_count)
            return original_summary(query, attempts)

        with patch.object(orchestrator.planner, "plan", return_value=first.program), \
             patch.object(orchestrator.composer, "compose", side_effect=[first.prompt, last.prompt]), \
             patch.object(orchestrator.evaluator, "evaluate", side_effect=[first.evaluation, last.evaluation]), \
             patch.object(orchestrator.reflector, "revise", return_value=last.program), \
             patch.object(orchestrator.memory_summarizer, "summarize", side_effect=observe_summary):
            result = orchestrator.hack("Summarize a paragraph.", budget=2)

        self.assertTrue(result.is_successful)
        self.assertEqual(target.call_count, 2)
        self.assertEqual(observed_target_calls, [2])
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(len(strategy_memory.updates), 1)
        self.assertEqual(strategy_memory.retrieval_requests, [
            (first.evaluation.failure_type, first.evaluation.failure_reason)
        ])
        self.assertEqual(len(store.entries[0]["attempts"]), 2)


if __name__ == "__main__":
    unittest.main()
