"""Offline checks for batch-prefix selection and shuffle ordering."""

from contextlib import redirect_stderr, redirect_stdout
import csv
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import main


class BatchLimitTests(unittest.TestCase):
    def test_prefix_selection_precedes_shuffle_and_respects_dataset_size(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "queries.csv"
            with dataset.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["target_query"])
                writer.writeheader()
                writer.writerows({"target_query": f"Fixture query {i}"} for i in range(20))

            for limit, expected in ((15, 15), (1, 1), (100, 20), (None, 11)):
                aux, target, embedder = MagicMock(), MagicMock(), MagicMock()
                aux.call_count = target.call_count = 0
                result = MagicMock()
                result.best_strategy_program = None
                result.to_result_dict.side_effect = lambda **kwargs: {
                    "success_score": 0.0, "is_successful": False, "iterations_used": 1,
                }
                with self.subTest(limit=limit), \
                     patch.object(main, "prepare_models", return_value=(aux, target, embedder)), \
                     patch.object(main, "PromptHackerOrchestrator") as controller, \
                     patch("utils.save_result"), patch("utils.save_final_statistics"), \
                     redirect_stdout(StringIO()):
                    controller.return_value.hack.return_value = result
                    results, stats = main.run_prompthacker_campaign(
                        str(dataset), limit=limit, shuffle_seed=17,
                    )
                self.assertEqual(len(results), expected)
                self.assertEqual(stats["total_targets"], expected)
                self.assertEqual({row["source_row"] for row in results}, set(range(expected)))
                self.assertEqual({call.args[0] for call in controller.return_value.hack.call_args_list},
                                 {f"Fixture query {i}" for i in range(expected)})

    def test_invalid_limits_and_conflicting_ranges_fail_before_model_setup(self):
        for options in ({"limit": 0}, {"limit": -1}, {"limit": True},
                        {"limit": 2, "start_row": 4}, {"limit": 2, "end_row": 4}):
            with self.subTest(options=options), patch.object(main, "prepare_models") as prepare:
                with self.assertRaises(ValueError):
                    main.run_prompthacker_campaign("unused.csv", **options)
                prepare.assert_not_called()

    def test_cli_passes_limit_to_batch_runner(self):
        with patch.object(sys, "argv", ["main.py", "--limit", "100"]), \
             patch.object(main, "run_prompthacker_campaign", return_value=([], {})) as run, \
             redirect_stdout(StringIO()):
            main.cli()
        self.assertEqual(run.call_args.kwargs["limit"], 100)

    def test_cli_rejects_invalid_limits_and_conflicting_ranges(self):
        for args in (["--limit", "0"], ["--limit", "-1"], ["--limit", "abc"],
                     ["--limit", "2", "--start-row", "5"]):
            with self.subTest(args=args), patch.object(sys, "argv", ["main.py", *args]), \
                 patch.object(main, "run_prompthacker_campaign") as run, \
                 redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main.cli()
                self.assertEqual(result.exception.code, 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
