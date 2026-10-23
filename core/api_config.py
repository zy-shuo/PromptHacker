"""Resolve API credentials by purpose, without mixing provider profiles."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class APIEndpoint:
    api_key: Optional[str]
    base_url: Optional[str]


_PROFILES = {
    "target": [
        ("TARGET_API_KEY", "TARGET_BASE_URL"),
        ("API_KEY_2", "BASE_URL_2"),
        ("API_KEY", "BASE_URL"),
    ],
    "aux": [
        ("AUX_API_KEY", "AUX_BASE_URL"),
        ("API_KEY_2", "BASE_URL_2"),
        ("API_KEY", "BASE_URL"),
    ],
    "embedding": [
        ("EMBEDDING_API_KEY", "EMBEDDING_BASE_URL"),
        ("API_KEY_1", "BASE_URL_1"),
        ("AUX_API_KEY", "AUX_BASE_URL"),
        ("API_KEY_2", "BASE_URL_2"),
        ("API_KEY", "BASE_URL"),
    ],
}


def resolve_api_endpoint(role: str) -> APIEndpoint:
    """Prefer named roles; retain numbered variables for older configurations.

    A key and URL are selected together from one profile. A partial named
    profile never borrows another provider's key or URL. An absent URL uses
    the API client's default endpoint.
    """
    if role not in _PROFILES:
        raise ValueError(f"Unknown API role: {role}")
    for key_name, url_name in _PROFILES[role]:
        api_key = os.getenv(key_name, "").strip() or None
        base_url = os.getenv(url_name, "").strip() or None
        if api_key is not None or base_url is not None:
            return APIEndpoint(api_key=api_key, base_url=base_url)
    return APIEndpoint(api_key=None, base_url=None)
