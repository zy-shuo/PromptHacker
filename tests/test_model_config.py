"""Offline model selection and local-loading checks; no weights or API calls."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from core.config import PromptHackerConfig
from core.model_config import ModelRegistry, prepare_models
from core.orchestrator import PromptHackerOrchestrator
from llm_wrapper import LLMWrapper
from memory.embedder import LocalEmbedder, OpenAIEmbedder
from memory.query_memory import QueryMemoryStore
from memory.strategy_memory import StrategyMemoryStore
import jailbreak_single_query
import main


class ModelConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "weights").mkdir()
        self.paths = {role: self.root / f"{role}_models.json" for role in ("target", "aux", "embedding")}
        self.target = {
            "default_model": "chat",
            "models": {
                "chat": {"model": "remote-chat", "api_key": "fixture-target",
                         "base_url": "https://target.example/v1"},
                "local-chat": {"backend": "local", "local_path": "weights"},
            },
        }
        self.aux = {
            "default_model": "chat",
            "models": {"chat": {"model": "remote-chat", "api_key": "fixture-aux",
                                "base_url": "https://aux.example/v1"}},
        }
        self.embedding = {
            "default_model": "bge-m3",
            "models": {
                "bge-m3": {"model": "BAAI/bge-m3", "backend": "local", "local_path": "weights"},
                "embed-api": {"model": "embed-id", "api_key": "fixture-embedding",
                              "base_url": "https://embed.example/v1"},
            },
        }

    def registry(self, role="target"):
        for kind, data in (("target", self.target), ("aux", self.aux), ("embedding", self.embedding)):
            self.paths[kind].write_text(json.dumps(data), encoding="utf-8")
        return ModelRegistry(role, str(self.paths[role]))

    def test_generation_defaults_and_role_specific_api_routing(self):
        registry = self.registry()
        target = registry.create_llm("chat")
        aux = self.registry("aux").create_llm("chat")
        self.assertEqual(target.max_tokens, 2048)
        self.assertEqual(target.temperature, 0)
        self.assertEqual(aux.temperature, 1)
        self.assertEqual(target.api_key, "fixture-target")
        self.assertEqual(aux.api_key, "fixture-aux")
        self.assertEqual(registry.select("remote-chat"), "chat")

    def test_per_model_length_and_credentials_have_no_provider_fallback(self):
        self.target["models"]["chat"].update(max_tokens=512, temperature=.7, api_key="explicit")
        self.target["models"]["chat"].pop("base_url")
        wrapper = self.registry().create_llm("chat")
        self.assertEqual(wrapper.max_tokens, 512)
        self.assertEqual(wrapper.temperature, 0.0)
        self.assertEqual(wrapper.api_key, "explicit")
        self.assertIsNone(wrapper.base_url)

    def test_decoding_policy_is_uniform_within_each_role(self):
        self.target["models"]["chat"].update(temperature=.7, top_p=.2)
        self.aux["models"]["chat"].update(temperature=.2, top_p=.1)
        target = self.registry("target").create_llm("chat")
        aux = self.registry("aux").create_llm("chat")
        self.assertEqual((target.temperature, target.top_p), (0.0, 1.0))
        self.assertEqual((aux.temperature, aux.top_p), (1.0, .9))

    def test_missing_target_key_never_uses_auxiliary_model_credentials(self):
        self.target["models"]["chat"]["api_key"] = ""
        with self.assertRaisesRegex(ValueError, "api_key"):
            self.registry("target").create_llm("chat")
        self.assertEqual(self.registry("aux").create_llm("chat").api_key, "fixture-aux")

    def test_api_embeddings_use_their_profile(self):
        with patch("memory.embedder.OpenAI") as client:
            embedder = self.registry("embedding").create_embedder("embed-api")
        client.assert_called_once_with(api_key="fixture-embedding", base_url="https://embed.example/v1")
        self.assertIsInstance(embedder, OpenAIEmbedder)
        self.assertEqual(embedder.model, "embed-id")

    def test_local_path_is_relative_to_config_and_can_be_overridden(self):
        registry = self.registry("embedding")
        embedder = registry.create_embedder()
        self.assertIsInstance(embedder, LocalEmbedder)
        self.assertEqual(embedder.local_path, str(self.root / "weights"))
        self.assertIsNone(embedder._encoder)
        other = self.root / "other"
        other.mkdir()
        self.assertEqual(registry.create_embedder(local_path=str(other)).local_path, str(other))
        self.assertNotEqual(embedder.cache_key, registry.create_embedder(local_path=str(other)).cache_key)

    def test_missing_local_path_fails_without_downloading(self):
        self.embedding["models"]["bge-m3"]["local_path"] = ""
        with self.assertRaisesRegex(ValueError, "No weights will be downloaded"):
            self.registry("embedding").create_embedder()
        with self.assertRaises(FileNotFoundError):
            self.registry("embedding").create_embedder(local_path=str(self.root / "absent"))

    def test_invalid_model_kind_and_length_fail_early(self):
        registry = self.registry()
        with self.assertRaisesRegex(ValueError, "Unknown or ambiguous"):
            registry.select("absent")
        self.target["models"]["chat"]["kind"] = "embedding"
        with self.assertRaisesRegex(ValueError, "not a chat"):
            self.registry().create_llm("chat")
        self.target["models"]["chat"].pop("kind")
        self.target["models"]["chat"]["max_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.registry().create_llm("chat")

    def test_model_list_contains_names_but_no_credentials(self):
        output = StringIO()
        with redirect_stdout(output):
            self.registry("embedding").list_models()
        self.assertIn("bge-m3", output.getvalue())
        self.assertNotIn("fixture-target", output.getvalue())
        self.assertNotIn("https://", output.getvalue())

    def test_both_memories_share_one_selected_embedder(self):
        self.registry()
        config = PromptHackerConfig(
            target_config_path=str(self.paths["target"]), aux_config_path=str(self.paths["aux"]),
            embedding_config_path=str(self.paths["embedding"]), query_memory_path=str(self.root / "query.json"),
            strategy_memory_path=str(self.root / "strategy.json"),
        )
        aux, target, embedder = prepare_models(config)
        controller = PromptHackerOrchestrator(aux, target, config=config, embedder=embedder)
        self.assertIs(controller.query_memory.embedder, embedder)
        self.assertIs(controller.strategy_memory.embedder, embedder)
        for component in (controller.planner, controller.composer, controller.evaluator,
                          controller.reflector, controller.memory_summarizer):
            self.assertIs(component.llm, aux)
        self.assertIsNone(embedder._encoder)

    def test_cli_lists_models_without_any_inference(self):
        self.embedding["models"]["bge-m3"]["local_path"] = ""
        self.registry()
        for module in (main, jailbreak_single_query):
            with self.subTest(module=module.__name__), \
                 patch.object(sys, "argv", [module.__name__, "--target-config", str(self.paths["target"]),
                                           "--aux-config", str(self.paths["aux"]), "--embedding-config",
                                           str(self.paths["embedding"]), "--list-models"]), \
                 patch.object(module, "prepare_models") as factory, redirect_stdout(StringIO()):
                module.cli()
                factory.assert_not_called()

    def test_cli_exposes_one_shared_auxiliary_model_option(self):
        for module in (main, jailbreak_single_query):
            output = StringIO()
            with self.subTest(module=module.__name__), \
                 patch.object(sys, "argv", [module.__name__, "--help"]), redirect_stdout(output):
                with self.assertRaises(SystemExit) as exit_result:
                    module.cli()
            self.assertEqual(exit_result.exception.code, 0)
            self.assertIn("--aux-model", output.getvalue())
            self.assertNotIn("--attack-model", output.getvalue())
            self.assertNotIn("--judge-model", output.getvalue())

    def test_entry_points_sum_all_auxiliary_and_target_calls(self):
        for module, function, arguments in (
            (jailbreak_single_query, jailbreak_single_query.jailbreak_attack,
             {"target_query": "Explain sorting."}),
            (main, main.run_prompthacker_campaign,
             {"file_path": "fixture.csv", "shuffle_order": False}),
        ):
            aux, target, embedder = MagicMock(), MagicMock(), MagicMock()
            aux.call_count, target.call_count = 7, 2
            result = MagicMock()
            result.best_strategy_program = None
            result.to_result_dict.return_value = {
                "success_score": 1.0, "is_successful": True, "iterations_used": 2,
            }
            with self.subTest(module=module.__name__), \
                 patch.object(module, "prepare_models", return_value=(aux, target, embedder)), \
                 patch.object(module, "PromptHackerOrchestrator") as controller, \
                 patch("utils.get_prompts", return_value=[{"target_query": "Explain sorting."}]), \
                 patch("utils.save_result"), patch("utils.save_final_statistics"), \
                 redirect_stdout(StringIO()):
                controller.return_value.hack.return_value = result
                function(**arguments)
            self.assertIs(controller.call_args.kwargs["aux_llm"], aux)
            self.assertIs(controller.call_args.kwargs["target_llm"], target)
            self.assertEqual(result.to_result_dict.call_args.kwargs["api_call_count"], 9)
            aux.clean_up.assert_called_once()
            target.clean_up.assert_called_once()

    def test_each_independent_role_call_increments_the_shared_counter(self):
        self.registry()
        config = PromptHackerConfig(
            query_memory_path=str(self.root / "query.json"),
            strategy_memory_path=str(self.root / "strategy.json"),
        )
        aux = LLMWrapper("fixture-aux", api_key="fixture-key")
        target = LLMWrapper("fixture-target", api_key="fixture-key", api_role="target")
        controller = PromptHackerOrchestrator(aux, target, config=config, embedder=MagicMock())
        roles = (controller.planner, controller.composer, controller.evaluator,
                 controller.reflector, controller.memory_summarizer)
        with patch.object(aux, "_generate_api", return_value='{"ok": true}') as auxiliary_requests, \
             patch.object(target, "_generate_api", return_value="Fixture response.") as target_requests:
            for expected_count, role in enumerate(roles, start=1):
                role.call_json("Fixture request.")
                self.assertEqual(aux.call_count, expected_count)
            target.generate([{"role": "user", "content": "Fixture target request."}])
        self.assertEqual(auxiliary_requests.call_count, 5)
        self.assertEqual(target_requests.call_count, 1)
        self.assertEqual(aux.call_count + target.call_count, 6)

    def test_local_embeddings_load_once_only_from_local_files(self):
        constructor = MagicMock()
        encoder = constructor.return_value
        encoder.max_seq_length = 8192
        encoder.encode.return_value.tolist.return_value = [1.0, 0.0]
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = constructor
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            embedder = self.registry("embedding").create_embedder()
            constructor.assert_not_called()
            self.assertEqual(embedder.embed("  "), [])
            self.assertEqual(embedder.embed("First sample."), [1.0, 0.0])
            embedder.embed("Second sample.")
        constructor.assert_called_once_with(str(self.root / "weights"), device=None,
                                            local_files_only=True, trust_remote_code=False)
        self.assertEqual(encoder.max_seq_length, 2048)
        self.assertEqual(embedder.call_count, 2)
        self.assertTrue(encoder.encode.call_args.kwargs["normalize_embeddings"])

    def test_local_generation_honors_path_and_loads_offline_on_first_call(self):
        registry = self.registry()
        with patch("llm_wrapper.AutoTokenizer.from_pretrained") as tokenizer, \
             patch("llm_wrapper.AutoModelForCausalLM.from_pretrained") as model, \
             patch("llm_wrapper.transformers.pipeline") as pipeline, redirect_stdout(StringIO()):
            pipeline.return_value.tokenizer.eos_token_id = 2
            pipeline.return_value.tokenizer.convert_tokens_to_ids.return_value = None
            pipeline.return_value.return_value = [{"generated_text": [{"content": "Fixture output."}]}]
            wrapper = registry.create_llm("local-chat")
            tokenizer.assert_not_called()
            self.assertEqual(wrapper.generate([{"role": "user", "content": "Fixture."}]), "Fixture output.")
            self.assertEqual(model.call_args.args[0], str(self.root / "weights"))
            self.assertTrue(model.call_args.kwargs["local_files_only"])
            self.assertTrue(tokenizer.call_args.kwargs["local_files_only"])
            self.assertFalse(pipeline.return_value.call_args.kwargs["do_sample"])
            self.assertNotIn("temperature", pipeline.return_value.call_args.kwargs)
            wrapper.release_local_model()


class CacheIdentityTests(unittest.TestCase):
    def test_same_dimension_vectors_are_refreshed_when_model_changes(self):
        embedder = MagicMock()
        embedder.cache_key = "new-embedding-model"
        embedder.embed.return_value = [1.0, 0.0]
        embedder.cosine_similarity = OpenAIEmbedder.cosine_similarity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query = QueryMemoryStore(str(root / "q.json"), embedder=embedder)
            query.entries = [{"query_representation": "Old query", "embedding": [0.0, 1.0],
                              "embedding_model": "old-embedding-model"}]
            query.retrieve("New query")
            self.assertEqual(query.entries[0]["embedding"], [1.0, 0.0])
            self.assertEqual(query.entries[0]["embedding_model"], embedder.cache_key)
            strategy = StrategyMemoryStore(str(root / "s.json"), embedder=embedder)
            note = {"failure_type": "off_topic", "failure_reason": "Old reason",
                    "failure_reason_embedding": [0.0, 1.0],
                    "failure_reason_embedding_model": "old-embedding-model",
                    "revision_advice": "Fixture advice."}
            strategy._ensure_record(["Fixture"])["improvement_notes"] = [note]
            self.assertEqual(strategy.retrieve(failure_type="off_topic", failure_reason="Current reason"),
                             ["Fixture advice."])
            self.assertEqual(note["failure_reason_embedding_model"], embedder.cache_key)
            # A second retrieval reuses the freshly saved historical vectors.
            embedder.embed.reset_mock()
            query.retrieve("New query")
            strategy.retrieve(failure_type="off_topic", failure_reason="Current reason")
            self.assertEqual([c.args[0] for c in embedder.embed.call_args_list], ["New query", "Current reason"])


if __name__ == "__main__":
    unittest.main()
