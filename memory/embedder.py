"""Interchangeable local and OpenAI-compatible dense embedding clients."""

import hashlib
import math
from pathlib import Path
import time
from typing import List, Optional, Protocol

from openai import OpenAI
from core.api_config import resolve_api_endpoint


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...
    def cosine_similarity(self, a: List[float], b: List[float]) -> float: ...


def _identity(*parts) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).hexdigest()


class OpenAIEmbedder:
    """Compute embeddings via OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        retry_sleep: float = 2.0,
    ):
        self.model = model
        endpoint = resolve_api_endpoint("embedding")
        if api_key is not None or base_url is not None:
            self.api_key = api_key
            self.base_url = base_url
        else:
            self.api_key = endpoint.api_key
            self.base_url = endpoint.base_url
        if not self.api_key:
            raise ValueError("Missing embedding API key; configure EMBEDDING_API_KEY or AUX_API_KEY.")
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.call_count = 0
        self.cache_key = _identity("api", self.model, self.base_url or "https://api.openai.com/v1")

    def embed(self, text: str) -> List[float]:
        text = text.strip()
        if not text:
            return []

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self.call_count += 1
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                return response.data[0].embedding
            except Exception as exc:
                last_error = exc
                print(f"[Embedder] attempt {attempt + 1}/{self.max_retries} failed: {exc}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_sleep)
        raise RuntimeError(f"Embedding failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class LocalEmbedder:
    """Load an existing SentenceTransformers directory on first use only."""

    cosine_similarity = staticmethod(OpenAIEmbedder.cosine_similarity)

    def __init__(self, local_path: str, model: str = "BAAI/bge-m3",
                 device: str = "auto", max_tokens: int = 2048):
        path = Path(local_path).expanduser().resolve()
        if not local_path or not path.is_dir():
            raise FileNotFoundError(f"Supply an existing local embedding model directory: {local_path}")
        if max_tokens <= 0:
            raise ValueError("Embedding max_tokens must be positive.")
        self.local_path = str(path)
        self.model = model
        self.device = device
        self.max_tokens = max_tokens
        self.cache_key = _identity("local", model, str(path), max_tokens)
        self.call_count = 0
        self._encoder = None

    def _load(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("Local embeddings require sentence-transformers in the active environment.") from exc
            encoder = SentenceTransformer(
                self.local_path, device=None if self.device == "auto" else self.device,
                local_files_only=True, trust_remote_code=False,
            )
            encoder.max_seq_length = min(self.max_tokens, encoder.max_seq_length)
            self._encoder = encoder
        return self._encoder

    def embed(self, text: str) -> List[float]:
        text = text.strip()
        if not text:
            return []
        encoder = self._load()
        self.call_count += 1
        return encoder.encode(
            text, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        ).tolist()


def default_embedder() -> Embedder:
    from core.model_config import ModelRegistry
    return ModelRegistry("embedding").create_embedder()
