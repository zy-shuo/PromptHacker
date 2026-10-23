"""Independent target, auxiliary, and embedding model configuration files."""

import argparse
import json
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILES = {
    role: PROJECT_ROOT / "config" / f"{role}_models.json"
    for role in ("target", "aux", "embedding")
}


class ModelRegistry:
    def __init__(self, role: str, path: Optional[str] = None):
        if role not in DEFAULT_CONFIG_FILES:
            raise ValueError(f"Unknown model role: {role}")
        self.role = role
        self.path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_FILES[role]
        if not self.path.is_file():
            raise FileNotFoundError(f"Model configuration not found: {self.path}")
        with self.path.open(encoding="utf-8-sig") as handle:
            self.data = json.load(handle)
        if not isinstance(self.data, dict) or not isinstance(self.data.get("models"), dict):
            raise ValueError(f"{self.path.name} must contain a 'models' object.")

    def select(self, name: Optional[str] = None) -> str:
        name = name or self.data.get("default_model")
        if not name:
            raise ValueError(f"Select a {self.role} model or set default_model in {self.path.name}.")
        if name in self.data["models"]:
            return name
        matches = [alias for alias, spec in self.data["models"].items()
                   if spec.get("model") == name]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Unknown or ambiguous {self.role} model '{name}'. Use --list-models.")

    def profile(self, name: str):
        spec = self.data["models"][name]
        if not isinstance(spec, dict):
            raise ValueError(f"Model '{name}' must be a configuration object.")
        expected_kind = "embedding" if self.role == "embedding" else "chat"
        if spec.get("kind", expected_kind) != expected_kind:
            raise ValueError(f"Model '{name}' is not a {expected_kind} model.")
        if spec.get("backend", "api") not in {"api", "local"}:
            raise ValueError(f"Model '{name}' has an unsupported backend.")
        return spec

    @staticmethod
    def max_tokens(spec) -> int:
        value = spec.get("max_tokens", 2048)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        return value

    def local_path(self, name, spec, override=None):
        value = override or spec.get("local_path")
        if not value:
            raise ValueError(
                f"Set models.{name}.local_path in {self.path.name} "
                "(or --embedding-path for embeddings). No weights will be downloaded."
            )
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.path.parent / path
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {path}")
        return str(path)

    def endpoint(self, spec):
        key = (spec.get("api_key") or "").strip()
        url = (spec.get("base_url") or "").strip() or None
        if not key:
            raise ValueError(f"Configure api_key in the selected model entry in {self.path.name}.")
        return key, url

    def create_llm(self, name: Optional[str] = None):
        from llm_wrapper import LLMWrapper

        if self.role == "embedding":
            raise ValueError("Use create_embedder for embedding models.")
        name = self.select(name)
        spec = self.profile(name)
        local = spec.get("backend", "api") == "local"
        # Role-wide decoding policy takes precedence over individual entries.
        temperature, top_p = (0.0, 1.0) if self.role == "target" else (1.0, 0.9)
        kwargs = dict(
            model_name=spec.get("model", name), max_tokens=self.max_tokens(spec),
            temperature=temperature, top_p=top_p, api_role=self.role, use_local=local,
            token_parameter=spec.get("token_parameter", "max_tokens"),
            enable_thinking=spec.get("enable_thinking"),
        )
        if local:
            import torch
            dtype_name = spec.get("dtype", "bfloat16")
            dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                      "float32": torch.float32, "auto": "auto"}
            if dtype_name not in dtypes:
                raise ValueError(f"Unsupported dtype for '{name}': {dtype_name}")
            kwargs.update(local_model_path=self.local_path(name, spec),
                          device_map=spec.get("device_map", "auto"),
                          torch_dtype=dtypes[dtype_name])
        else:
            kwargs["api_key"], kwargs["base_url"] = self.endpoint(spec)
        return LLMWrapper(**kwargs)

    def create_embedder(self, name: Optional[str] = None, local_path: Optional[str] = None):
        from memory.embedder import LocalEmbedder, OpenAIEmbedder

        if self.role != "embedding":
            raise ValueError("Embedding models must come from embedding_models.json.")
        name = self.select(name)
        spec = self.profile(name)
        if spec.get("backend", "api") == "local":
            return LocalEmbedder(
                local_path=self.local_path(name, spec, local_path),
                model=spec.get("model", name), device=spec.get("device", "auto"),
                max_tokens=self.max_tokens(spec),
            )
        if local_path:
            raise ValueError("--embedding-path requires a local embedding model profile.")
        key, url = self.endpoint(spec)
        return OpenAIEmbedder(model=spec.get("model", name), api_key=key, base_url=url)

    def list_models(self):
        """List model names without credentials, API requests, or weight loading."""
        print(f"[{self.role}] {self.path.name}")
        for name, spec in self.data["models"].items():
            default = " (default)" if name == self.data.get("default_model") else ""
            print(f"  {name:24} {spec.get('backend', 'api')}{default}")


def add_model_arguments(parser: argparse.ArgumentParser):
    for role in DEFAULT_CONFIG_FILES:
        parser.add_argument(f"--{role}-config", dest=f"{role}_config_path",
                            help=f"Override config/{role}_models.json")
    parser.add_argument("--aux-model", help="Shared model profile for all components except the target and embeddings")
    parser.add_argument("--target-model", help="Target model profile")
    parser.add_argument("--embedding-model", help="Embedding model profile (default: bge-m3)")
    parser.add_argument("--embedding-path", help="Override the selected local embedding model directory")
    parser.add_argument("--list-models", action="store_true", help="List model names without loading weights or calling APIs")


def list_model_profiles(arguments):
    for role in DEFAULT_CONFIG_FILES:
        ModelRegistry(role, arguments.get(f"{role}_config_path")).list_models()


def prepare_models(config, aux_model=None, target_model=None, use_local=False):
    """Create two generation models and one embedder from separate files."""
    aux_registry = ModelRegistry("aux", config.aux_config_path)
    target_registry = ModelRegistry("target", config.target_config_path)
    embedding_registry = ModelRegistry("embedding", config.embedding_config_path)
    target = target_registry.select(target_model)
    if use_local and target_registry.profile(target).get("backend", "api") != "local":
        raise ValueError("use_local=True requires a target profile with backend='local'.")
    embedder = embedding_registry.create_embedder(config.embedding_model, config.embedding_path)
    return aux_registry.create_llm(aux_model), target_registry.create_llm(target), embedder
