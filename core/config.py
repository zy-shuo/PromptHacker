"""
PromptHacker configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptHackerConfig:
    """Framework hyper-parameters aligned with the paper."""

    # Target-model query budget per malicious query (paper: B=5)
    budget: int = 5

    # Success threshold (paper: tau=1.0)
    success_threshold: float = 1.0

    # Query-level memory retrieval
    query_memory_similarity_threshold: float = 0.7
    query_memory_top_k: int = 1

    # Strategy-level memory retrieval
    strategy_memory_similarity_threshold: float = 0.5
    strategy_memory_top_k: int = 3

    # Independent role-specific model files (None uses config/*_models.json).
    target_config_path: Optional[str] = None
    aux_config_path: Optional[str] = None
    embedding_config_path: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_path: Optional[str] = None

    # Agent JSON parse retries
    agent_max_retries: int = 3

    # Memory persistence paths
    query_memory_path: str = "memory/query_memory.json"
    strategy_memory_path: str = "memory/strategy_memory.json"

    # Batch campaign: shuffle order by default to reduce memory-order bias
    shuffle_order: bool = True
    shuffle_seed: Optional[int] = None

    # Output
    output_file: str = "result/prompthacker_results.jsonl"
    algorithm_name: str = "PromptHacker"


DEFAULT_CONFIG = PromptHackerConfig()
