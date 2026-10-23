"""
Strategy taxonomy loader and validation utilities.
Reuses STRATEGY_HIERARCHY_EN with normalized name aliases.
"""

import json
from typing import Dict, List, Set

import STRATEGY_HIERARCHY_EN

# Paper name -> current hierarchy name (backward compatibility)
STRATEGY_NAME_ALIASES = {
    "Role Play": "Role-Playing",
    "Word Order Flip": "Word Order Inversion",
}

REVERSE_ALIASES = {v: k for k, v in STRATEGY_NAME_ALIASES.items()}


def _normalize_hierarchy(raw: Dict) -> Dict:
    """Ensure every atomic strategy has description + example dict."""
    normalized = {}
    for family, tactics in raw.items():
        normalized[family] = {}
        for tactic, strategies in tactics.items():
            normalized[family][tactic] = {}
            for name, details in strategies.items():
                if isinstance(details, dict):
                    normalized[family][tactic][name] = {
                        "description": details.get("description", ""),
                        "example": details.get("example", ""),
                    }
                else:
                    normalized[family][tactic][name] = {
                        "description": str(details),
                        "example": "",
                    }
    return normalized


STRATEGY_HIERARCHY = _normalize_hierarchy(STRATEGY_HIERARCHY_EN.STRATEGY_HIERARCHY)


def resolve_strategy_name(name: str) -> str:
    """Map paper-style names to hierarchy keys."""
    if name in get_all_strategy_names():
        return name
    return STRATEGY_NAME_ALIASES.get(name, name)


def get_all_strategy_names() -> Set[str]:
    names: Set[str] = set()
    for family in STRATEGY_HIERARCHY.values():
        for tactic in family.values():
            names.update(tactic.keys())
    return names


def validate_strategies(strategies: List[str]) -> List[str]:
    """Return valid strategy names; resolve aliases when possible."""
    valid: List[str] = []
    all_names = get_all_strategy_names()
    for s in strategies:
        resolved = resolve_strategy_name(s)
        if resolved in all_names:
            valid.append(resolved)
    return valid


def format_hierarchy_for_prompt(hierarchy: Dict = None) -> str:
    """Serialize hierarchy as JSON for LLM module prompts."""
    h = hierarchy or STRATEGY_HIERARCHY
    return json.dumps(h, ensure_ascii=False, indent=2)


def get_strategy_details(strategies: List[str]) -> Dict[str, Dict[str, str]]:
    details: Dict[str, Dict[str, str]] = {}
    for strategy in strategies:
        resolved = resolve_strategy_name(strategy)
        for family in STRATEGY_HIERARCHY.values():
            for tactic in family.values():
                if resolved in tactic:
                    details[resolved] = tactic[resolved]
                    break
    return details
