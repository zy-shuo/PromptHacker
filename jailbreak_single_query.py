"""
PromptHacker — single query jailbreak attack wrapper.
"""

import argparse
import time
from dataclasses import replace
from typing import Dict, Optional

from core.config import PromptHackerConfig
from core.orchestrator import PromptHackerOrchestrator
from core.model_config import add_model_arguments, list_model_profiles, prepare_models

def jailbreak_attack(
    target_query: str,
    aux_model: Optional[str] = None,
    target_model: Optional[str] = None,
    budget: int = 5,
    success_threshold: float = 1.0,
    use_local: bool = False,
    target_id: Optional[str] = None,
    target_source: str = "custom",
    config: Optional[PromptHackerConfig] = None,
    target_config_path: Optional[str] = None,
    aux_config_path: Optional[str] = None,
    embedding_config_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_path: Optional[str] = None,
) -> Dict:
    """
    Perform a single-query jailbreak attack using PromptHacker.
    """
    start_time = time.time()
    cfg = config or PromptHackerConfig(budget=budget, success_threshold=success_threshold)
    overrides = {key: value for key, value in dict(
        target_config_path=target_config_path, aux_config_path=aux_config_path,
        embedding_config_path=embedding_config_path, embedding_model=embedding_model,
        embedding_path=embedding_path,
    ).items() if value is not None}
    cfg = replace(cfg, **overrides)
    aux_llm, target_llm, embedder = prepare_models(
        cfg, aux_model, target_model, use_local=use_local,
    )
    print("=== PromptHacker Single Query Attack ===")
    print(f"Target query: {target_query[:100]}...")
    print(f"Auxiliary model: {aux_llm.model_name}")
    print(f"Target model: {target_llm.model_name}")
    print(f"Embedding model: {embedder.model}")
    print(f"Budget: {cfg.budget}")
    print("=" * 50)

    orchestrator = PromptHackerOrchestrator(
        aux_llm=aux_llm,
        target_llm=target_llm,
        config=cfg,
        embedder=embedder,
    )

    try:
        hack_result = orchestrator.hack(target_query, budget=cfg.budget)
        execution_time = time.time() - start_time

        if target_id is None:
            target_id = f"custom_{int(time.time() * 1000)}"

        result = hack_result.to_result_dict(
            target_source=target_source,
            execution_time=execution_time,
            # Sharing a model does not merge independent role invocations.
            api_call_count=aux_llm.call_count + target_llm.call_count,
            algorithm=cfg.algorithm_name,
        )
        result["target_id"] = target_id

        aux_llm.clean_up()
        target_llm.clean_up()

        print(f"\nAttack completed!")
        print(f"Success score: {result['success_score']:.3f}")
        print(f"Success: {'Yes' if result['is_successful'] else 'No'}")
        print(f"Execution time: {execution_time:.2f}s")
        print(f"Iterations used: {result['iterations_used']}")

        return result

    except Exception as exc:
        print(f"Error during attack: {exc}")
        raise


def cli():
    import utils

    parser = argparse.ArgumentParser(description="Run PromptHacker for one query.")
    add_model_arguments(parser)
    parser.add_argument("--query", dest="target_query", help="Query to evaluate")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--output-file", default="result/single_query_result.jsonl")
    args = vars(parser.parse_args())
    if args.pop("list_models"):
        list_model_profiles(args)
        return
    if not args["target_query"]:
        parser.error("--query is required unless --list-models is used")
    output_file = args.pop("output_file")
    result = jailbreak_attack(**args)
    utils.save_result(result, output_file)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    cli()
