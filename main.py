"""
PromptHacker — batch campaign entry point.
"""

import argparse
import time
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from core.config import PromptHackerConfig
from core.orchestrator import PromptHackerOrchestrator
from core.model_config import add_model_arguments, list_model_profiles, prepare_models
import utils

def run_prompthacker_campaign(
    file_path: str,
    start_row: int = 0,
    end_row: int = 10,
    aux_model: Optional[str] = None,
    target_model: Optional[str] = None,
    budget: int = 5,
    output_file: str = "result/prompthacker_results.jsonl",
    success_threshold: float = 1.0,
    use_local: bool = False,
    shuffle_order: bool = True,
    shuffle_seed: Optional[int] = None,
    config: Optional[PromptHackerConfig] = None,
    target_config_path: Optional[str] = None,
    aux_config_path: Optional[str] = None,
    embedding_config_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer.")
        if start_row != 0 or end_row != 10:
            raise ValueError("--limit cannot be combined with a custom --start-row/--end-row range.")
        start_row, end_row = 0, limit - 1
    cfg = config or PromptHackerConfig(
        budget=budget,
        success_threshold=success_threshold,
        output_file=output_file,
        shuffle_order=shuffle_order,
        shuffle_seed=shuffle_seed,
    )
    overrides = {key: value for key, value in dict(
        target_config_path=target_config_path, aux_config_path=aux_config_path,
        embedding_config_path=embedding_config_path, embedding_model=embedding_model,
        embedding_path=embedding_path,
    ).items() if value is not None}
    cfg = replace(cfg, **overrides)
    aux_llm, target_llm, embedder = prepare_models(
        cfg, aux_model, target_model, use_local=use_local,
    )

    print("=== PromptHacker Jailbreak Attack System ===")
    print(f"Target file: {file_path}")
    print(f"Processing range: {start_row} - {end_row}")
    print(f"Auxiliary model: {aux_llm.model_name}")
    print(f"Target model: {target_llm.model_name}")
    print(f"Embedding model: {embedder.model}")
    print(f"Budget (target-model calls): {cfg.budget}")
    print("=" * 50)

    orchestrator = PromptHackerOrchestrator(
        aux_llm=aux_llm,
        target_llm=target_llm,
        config=cfg,
        embedder=embedder,
    )

    prompts = utils.get_prompts(file_path, start_row, end_row)
    if cfg.shuffle_order:
        prompts = utils.shuffle_prompts(prompts, seed=cfg.shuffle_seed)
        seed_info = cfg.shuffle_seed if cfg.shuffle_seed is not None else "random"
        print(f"Prompt order shuffled (seed={seed_info}) to reduce memory-order bias")
    results: List[Dict] = []
    campaign_stats = {
        "total_targets": len(prompts),
        "successful_attacks": 0,
        "average_score": 0.0,
        "strategy_popularity": defaultdict(int),
        "attack_results": [],
        "shuffle_order": cfg.shuffle_order,
        "shuffle_seed": cfg.shuffle_seed,
    }

    for i, prompt in enumerate(prompts):
        print(f"\n{'=' * 20} Target {i + 1}/{len(prompts)} {'=' * 20}")
        print(f"Target prompt: {prompt['target_query'][:100]}...")
        start_time = time.time()

        try:
            hack_result = orchestrator.hack(prompt["target_query"], budget=cfg.budget)
            execution_time = time.time() - start_time

            if prompt.get("target_source"):
                target_source = "strongreject_" + prompt["target_source"]
            else:
                target_source = "harmbench"

            result = hack_result.to_result_dict(
                target_type=prompt.get("target_type", ""),
                target_source=target_source,
                execution_time=execution_time,
                # aux_llm accumulates every call by every auxiliary role.
                api_call_count=(
                    aux_llm.call_count + target_llm.call_count
                ),
                algorithm=cfg.algorithm_name,
            )
            result["source_row"] = prompt.get("source_row")
            result["processing_order"] = i + 1

            aux_llm.clean_up()
            target_llm.clean_up()

            results.append(result)
            campaign_stats["attack_results"].append(result)

            if result["is_successful"]:
                campaign_stats["successful_attacks"] += 1

            if hack_result.best_strategy_program:
                for s in hack_result.best_strategy_program.strategies:
                    campaign_stats["strategy_popularity"][s] += 1

            utils.save_result(result, cfg.output_file)
            print(
                f"Attack completed! Score: {result['success_score']:.3f}, "
                f"Success: {'Yes' if result['is_successful'] else 'No'}, "
                f"Time: {execution_time:.2f}s"
            )
        except Exception as exc:
            print(f"Error attacking target {i + 1}: {exc}")
            continue

    if results:
        campaign_stats["average_score"] = sum(r["success_score"] for r in results) / len(results)
        campaign_stats["success_rate"] = campaign_stats["successful_attacks"] / len(results)

    print(f"\n{'=' * 20} Campaign Summary {'=' * 20}")
    print(f"Total targets: {campaign_stats['total_targets']}")
    print(f"Successful attacks: {campaign_stats['successful_attacks']}")
    if results:
        print(f"Success rate: {campaign_stats['success_rate']:.2%}")
        print(f"Average score: {campaign_stats['average_score']:.3f}")

    utils.save_final_statistics(campaign_stats, "result/prompthacker_stats.json")
    return results, campaign_stats


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def cli():
    parser = argparse.ArgumentParser(description="Run a PromptHacker batch campaign.")
    add_model_arguments(parser)
    parser.add_argument("--file-path", default="datasets/harmbench.csv")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=10)
    parser.add_argument("--limit", type=positive_int, help="Run the first X dataset rows, selected before shuffling")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--output-file", default="result/prompthacker_results.jsonl")
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--no-shuffle", dest="shuffle_order", action="store_false")
    args = vars(parser.parse_args())
    if args.pop("list_models"):
        list_model_profiles(args)
        return
    if args["limit"] is not None and (args["start_row"] != 0 or args["end_row"] != 10):
        parser.error("--limit cannot be combined with a custom --start-row/--end-row range")
    results, stats = run_prompthacker_campaign(**args)
    print(f"\nDone. Processed {len(results)} targets.")


if __name__ == "__main__":
    cli()
