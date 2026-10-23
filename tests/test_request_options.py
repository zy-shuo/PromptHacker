"""Offline API request-schema checks for per-model configuration."""

import unittest
from unittest.mock import MagicMock, patch

from llm_wrapper import LLMWrapper


class RequestOptionsTests(unittest.TestCase):
    def request(self, **kwargs):
        with patch("llm_wrapper.OpenAI") as client:
            response = MagicMock()
            response.choices[0].message.content = "Fixture response."
            client.return_value.chat.completions.create.return_value = response
            wrapper = LLMWrapper("fixture-model", api_key="fixture-key", **kwargs)
            wrapper.generate([{"role": "user", "content": "Fixture request."}])
            return client.return_value.chat.completions.create.call_args.kwargs

    def test_completion_token_limit_and_explicit_false_thinking_are_forwarded(self):
        request = self.request(max_tokens=6048, token_parameter="max_completion_tokens",
                               enable_thinking=False, temperature=0.0, top_p=1.0)
        self.assertEqual(request["max_completion_tokens"], 6048)
        self.assertNotIn("max_tokens", request)
        self.assertEqual(request["extra_body"], {"enable_thinking": False})
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["top_p"], 1.0)

    def test_default_length_parameter_and_auxiliary_sampling(self):
        request = self.request()
        self.assertEqual(request["max_tokens"], 2048)
        self.assertNotIn("max_completion_tokens", request)
        self.assertNotIn("extra_body", request)
        self.assertEqual(request["temperature"], 1)
        self.assertEqual(request["top_p"], .9)

    def test_thinking_true_is_preserved(self):
        self.assertEqual(self.request(enable_thinking=True)["extra_body"], {"enable_thinking": True})

    def test_invalid_request_options_fail_before_any_request(self):
        with patch("llm_wrapper.OpenAI") as client:
            with self.assertRaisesRegex(ValueError, "token_parameter"):
                LLMWrapper("fixture", token_parameter="invalid_parameter")
            with self.assertRaisesRegex(ValueError, "boolean"):
                LLMWrapper("fixture", enable_thinking="false")
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
