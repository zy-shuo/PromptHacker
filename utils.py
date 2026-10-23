"""
General utility functions for dataset loading and result persistence.
"""

import csv
import json
import os
import random
from typing import Any, Dict, List, Optional


def get_prompts(file_name: str, start_row: int = 0, end_row: Optional[int] = None) -> List[Dict[str, str]]:
    """Load target prompts from a CSV benchmark file."""
    prompt_list = []
    with open(file_name, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        if end_row is None:
            end_row = len(rows) - 1
        for idx, row in enumerate(rows):
            if idx < start_row:
                continue
            if idx > end_row:
                break
            prompt_list.append({
                "source_row": idx,
                "target_type": row.get("target_type", ""),
                "target_query": row.get("target_query", ""),
                "target_source": row.get("target_source", ""),
            })
    return prompt_list


def shuffle_prompts(
    prompts: List[Dict[str, Any]],
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Shuffle prompt order to reduce memory-order bias across batch runs.

    Args:
        prompts: Prompt list loaded from get_prompts()
        seed: Optional RNG seed for reproducible shuffling. If None, uses
              system randomness so each campaign gets a different order.
    """
    shuffled = list(prompts)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def save_result(result: Dict[str, Any], output_file: str = "result/prompthacker_results.jsonl") -> None:
    """Append a single attack result to a JSONL file."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


def save_final_statistics(
    campaign_stats: Dict[str, Any],
    output_file: str = "result/prompthacker_stats.json",
) -> None:
    """Save batch campaign statistics."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(campaign_stats, file, ensure_ascii=False, indent=2)
