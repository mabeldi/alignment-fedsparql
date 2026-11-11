"""Loads per-agent model configuration from agent_models.yaml.

Edit agent_models.yaml to change which provider/model each agent uses.
This module just reads it and exposes small helpers for the LLM router.
"""
import os
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent / "agent_models.yaml"


def _load():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Loaded once at import time. Call reload() if you edit the YAML at runtime.
_CONFIG = _load()


def get_agent_config(agent_key: str) -> dict:
    """Return {'provider', 'model', 'temperature', 'reasoning_effort'} for an agent."""
    agents = _CONFIG.get("agents", {})
    if agent_key not in agents:
        raise KeyError(
            f"No model config for agent '{agent_key}'. "
            f"Add an entry under 'agents:' in {_CONFIG_PATH.name}."
        )
    cfg = dict(agents[agent_key])
    defaults = _CONFIG.get("defaults", {})
    cfg.setdefault("temperature", defaults.get("temperature", 0.2))
    cfg.setdefault("reasoning_effort", defaults.get("reasoning_effort"))
    if "provider" not in cfg or "model" not in cfg:
        raise ValueError(
            f"Agent '{agent_key}' must define both 'provider' and 'model' "
            f"in {_CONFIG_PATH.name}."
        )
    return cfg


def get_provider_config(provider: str) -> dict:
    """Return connection settings for a provider (e.g. ollama base_url)."""
    return dict(_CONFIG.get("providers", {}).get(provider, {}))


def get_ollama_base_url() -> str:
    """Ollama endpoint: env var wins, then YAML, then localhost default."""
    return (
        os.getenv("OLLAMA_BASE_URL")
        or get_provider_config("ollama").get("base_url")
        or "http://localhost:11434/v1"
    )
