"""Offline checks for target/auxiliary/embedding API routing."""

from contextlib import redirect_stdout
from io import StringIO
import os
import unittest
from unittest.mock import MagicMock, patch

from core.api_config import resolve_api_endpoint
from llm_wrapper import LLMWrapper
from memory.embedder import OpenAIEmbedder
import jailbreak_single_query
import main


class APIConfigTests(unittest.TestCase):
    def test_named_profiles_route_target_auxiliary_and_embeddings(self):
        settings = {
            "TARGET_API_KEY": "target-fixture-key", "TARGET_BASE_URL": "https://target.example/v1",
            "AUX_API_KEY": "aux-fixture-key", "AUX_BASE_URL": "https://aux.example/v1",
            "EMBEDDING_API_KEY": "embed-fixture-key", "EMBEDDING_BASE_URL": "https://embed.example/v1",
        }
        with patch.dict(os.environ, settings, clear=True):
            for role, prefix in [("target", "TARGET"), ("aux", "AUX"), ("embedding", "EMBEDDING")]:
                with self.subTest(role=role):
                    endpoint = resolve_api_endpoint(role)
                    self.assertEqual(endpoint.api_key, settings[f"{prefix}_API_KEY"])
                    self.assertEqual(endpoint.base_url, settings[f"{prefix}_BASE_URL"])

    def test_embedding_defaults_to_auxiliary_service(self):
        with patch.dict(os.environ, {"AUX_API_KEY": "aux-key", "AUX_BASE_URL": "https://aux.example/v1"}, clear=True):
            self.assertEqual(resolve_api_endpoint("embedding"), resolve_api_endpoint("aux"))

    def test_legacy_numbered_profiles_keep_their_original_meaning(self):
        with patch.dict(os.environ, {
            "API_KEY_1": "old-embedding-key", "BASE_URL_1": "https://old-embedding.example/v1",
            "API_KEY_2": "old-generation-key", "BASE_URL_2": "https://old-generation.example/v1",
        }, clear=True):
            self.assertEqual(resolve_api_endpoint("embedding").api_key, "old-embedding-key")
            self.assertEqual(resolve_api_endpoint("target").api_key, "old-generation-key")
            self.assertEqual(resolve_api_endpoint("aux").api_key, "old-generation-key")

    def test_named_profiles_override_legacy_and_generic_settings(self):
        with patch.dict(os.environ, {
            "TARGET_API_KEY": "new-target", "TARGET_BASE_URL": "https://target.example/v1",
            "AUX_API_KEY": "new-aux", "AUX_BASE_URL": "https://aux.example/v1",
            "API_KEY_2": "old", "BASE_URL_2": "https://old.example/v1",
            "API_KEY": "generic", "BASE_URL": "https://generic.example/v1",
        }, clear=True):
            self.assertEqual(resolve_api_endpoint("target").api_key, "new-target")
            self.assertEqual(resolve_api_endpoint("aux").api_key, "new-aux")

    def test_partial_profile_never_mixes_providers(self):
        with patch.dict(os.environ, {
            "TARGET_API_KEY": "target-key",
            "API_KEY_2": "old-key", "BASE_URL_2": "https://old.example/v1",
        }, clear=True):
            endpoint = resolve_api_endpoint("target")
            self.assertEqual(endpoint.api_key, "target-key")
            self.assertIsNone(endpoint.base_url)

        with patch.dict(os.environ, {
            "TARGET_BASE_URL": "https://target.example/v1", "AUX_API_KEY": "aux-key",
        }, clear=True):
            self.assertIsNone(resolve_api_endpoint("target").api_key)

    def test_missing_target_key_does_not_borrow_auxiliary_credentials(self):
        with patch.dict(os.environ, {"AUX_API_KEY": "aux-key", "AUX_BASE_URL": "https://aux.example/v1"}, clear=True), \
             patch("llm_wrapper.OpenAI") as client:
            target = LLMWrapper("target-model", api_role="target")
            with self.assertRaisesRegex(ValueError, "TARGET_API_KEY"):
                target.generate([{"role": "user", "content": "Fixture request."}])
            client.assert_not_called()

    def test_generation_requests_use_their_own_profile_and_model(self):
        with patch.dict(os.environ, {
            "TARGET_API_KEY": "target-key", "TARGET_BASE_URL": "https://target.example/v1",
            "AUX_API_KEY": "aux-key", "AUX_BASE_URL": "https://aux.example/v1",
        }, clear=True), patch("llm_wrapper.OpenAI") as client:
            response = MagicMock()
            response.choices[0].message.content = "Fixture output."
            client.return_value.chat.completions.create.return_value = response
            for role in ["target", "aux"]:
                wrapper = LLMWrapper(f"{role}-model", api_role=role)
                self.assertEqual(wrapper.generate([{"role": "user", "content": "Fixture request."}]), "Fixture output.")
                client.assert_called_with(api_key=f"{role}-key", base_url=f"https://{role}.example/v1")
                self.assertEqual(client.return_value.chat.completions.create.call_args.kwargs["model"], f"{role}-model")

    def test_explicit_connection_overrides_named_profile_as_a_pair(self):
        with patch.dict(os.environ, {"AUX_API_KEY": "aux-key", "AUX_BASE_URL": "https://aux.example/v1"}, clear=True):
            wrapper = LLMWrapper("custom-model", api_key="explicit-key")
            self.assertEqual(wrapper.api_key, "explicit-key")
            self.assertIsNone(wrapper.base_url)

    def test_embedding_client_uses_the_embedding_override(self):
        with patch.dict(os.environ, {
            "EMBEDDING_API_KEY": "embedding-key", "EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "AUX_API_KEY": "aux-key", "AUX_BASE_URL": "https://aux.example/v1",
        }, clear=True), patch("memory.embedder.OpenAI") as client:
            embedder = OpenAIEmbedder()
            client.assert_called_once_with(api_key="embedding-key", base_url="https://embedding.example/v1")
            self.assertEqual(embedder.model, "text-embedding-3-small")

    def test_single_and_batch_entry_points_assign_model_roles(self):
        class StopBeforeSearch(Exception):
            pass

        for module, function, arguments in [
            (jailbreak_single_query, jailbreak_single_query.jailbreak_attack, {"target_query": "Explain sorting."}),
            (main, main.run_prompthacker_campaign, {"file_path": "fixture.csv"}),
        ]:
            with self.subTest(entry_point=module.__name__), \
                 patch("core.model_config.ModelRegistry") as registry_class, \
                 patch.object(module, "PromptHackerOrchestrator", side_effect=StopBeforeSearch), \
                 redirect_stdout(StringIO()):
                registries = {role: MagicMock() for role in ("aux", "target", "embedding")}
                registry_class.side_effect = lambda role, path: registries[role]
                registries["target"].select.return_value = "selected-target"
                with self.assertRaises(StopBeforeSearch):
                    function(**arguments)
                self.assertEqual([call.args[0] for call in registry_class.call_args_list],
                                 ["aux", "target", "embedding"])
                registries["aux"].create_llm.assert_called_once_with(None)
                registries["target"].create_llm.assert_called_once_with("selected-target")

    def test_unknown_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown API role"):
            resolve_api_endpoint("not-a-role")


if __name__ == "__main__":
    unittest.main()
