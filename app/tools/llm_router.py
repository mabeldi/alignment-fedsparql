"""Single entry point the agents use to call an LLM.

An agent calls run_agent("agent1_extractor", messages) and the router looks up
the provider/model from config/agent_models.yaml and dispatches accordingly.
To experiment with a different model, edit that YAML — no agent code changes.
"""
from config.model_config import get_agent_config

from tools.ollama_client import ollama_execute, ollama_async_execute
from tools.open_ai import open_ai_execute, open_ai_async_execute
from tools.anthropic_client import anthropic_execute, anthropic_async_execute
from tools.llma3_client import llma3_execute, llma3_async_execute
from tools.gemini_client import gemini_execute, gemini_async_execute
from tools.vercel_client import vercel_execute, vercel_async_execute
from tools.output_writer import write_prompt_file
from tools.token_usage import agent_scope

# provider name -> (sync_fn, async_fn)
_DISPATCH = {
    "ollama": (ollama_execute, ollama_async_execute),
    "openai": (open_ai_execute, open_ai_async_execute),
    "anthropic": (anthropic_execute, anthropic_async_execute),
    "huggingface": (llma3_execute, llma3_async_execute),
    "gemini": (gemini_execute, gemini_async_execute),
    "vercel": (vercel_execute, vercel_async_execute),
}


# Providers whose clients accept `reasoning_effort`. Passing it to the others
# would be a TypeError, and the setting is meaningless for a non-reasoning model.
_SUPPORTS_REASONING_EFFORT = {"ollama"}


def _resolve(agent_key, temperature):
    cfg = get_agent_config(agent_key)
    provider = cfg["provider"]
    if provider not in _DISPATCH:
        raise ValueError(
            f"Unknown provider '{provider}' for agent '{agent_key}'. "
            f"Valid providers: {', '.join(_DISPATCH)}."
        )
    temp = cfg["temperature"] if temperature is None else temperature
    extra = {}
    if provider in _SUPPORTS_REASONING_EFFORT and cfg.get("reasoning_effort"):
        extra["reasoning_effort"] = cfg["reasoning_effort"]
    return provider, cfg["model"], temp, extra


def run_agent(agent_key: str, messages, temperature=None, json_mode=False) -> str:
    """Synchronous LLM call for the given agent, per YAML config.

    json_mode=True asks the provider to return strict JSON (supported by
    ollama/openai; ignored by providers that don't support response_format).
    """
    provider, model, temp, extra = _resolve(agent_key, temperature)
    sync_fn, _ = _DISPATCH[provider]
    # Dumped here rather than inside each agent: this is the last point the
    # messages are still exactly what the provider receives. Debug output only —
    # write_prompt_file swallows its own failures.
    write_prompt_file(agent_key, messages, provider, model)
    # agent_scope tells the provider client which agent to bill the tokens to;
    # the client is the only place the counts are visible. See tools/token_usage.
    with agent_scope(agent_key):
        return sync_fn(messages, temperature=temp, model=model, json_mode=json_mode, **extra)


async def run_agent_async(agent_key: str, messages, temperature=None, json_mode=False) -> str:
    """Async LLM call for the given agent, per YAML config."""
    provider, model, temp, extra = _resolve(agent_key, temperature)
    _, async_fn = _DISPATCH[provider]
    write_prompt_file(agent_key, messages, provider, model)
    with agent_scope(agent_key):
        return await async_fn(messages, temperature=temp, model=model, json_mode=json_mode, **extra)
