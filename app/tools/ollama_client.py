"""Chat completions against a local Ollama server.

Ollama exposes an OpenAI-compatible API, so we reuse the OpenAI client and
just point it at the Ollama endpoint. No API key is required.

REASONING MODELS AND THE EMPTY ANSWER
-------------------------------------
`gpt-oss` is a reasoning model: it writes its chain of thought into a separate
`reasoning` field and only then writes the real answer into `content`. Ollama
serves a 4096-token context by default, and on the Agent 7 prompt that budget
ran out DURING the reasoning:

    finish_reason : length
    usage         : prompt 2447 + completion 1649 = 4096   (the whole window)
    reasoning     : 7275 chars — including a fully formed SPARQL query
    content       : 0 chars

The model had solved the task and was cut off before it could write the answer
down, and the caller received "". Two things follow, both handled here:

  * `reasoning_effort` (per agent, in agent_models.yaml) caps how much of the
    budget the model may spend thinking. On the same prompt, "low" turns the
    result into finish_reason=stop with a complete query.
  * an empty `content` is now an ERROR, not an empty string. Returning ""
    silently made a truncation look like a model that had nothing to say, and
    the real cause was invisible five call frames away.

`options: {num_ctx: ...}` does NOT work here — Ollama's OpenAI-compatible
endpoint ignores it (verified: total_tokens stayed at 4096). To enlarge the
window, set OLLAMA_CONTEXT_LENGTH on the Ollama SERVER, or bake
`PARAMETER num_ctx` into a Modelfile.
"""
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from config.model_config import get_ollama_base_url
from tools.token_usage import record

load_dotenv()

_BASE_URL = get_ollama_base_url()
_DEFAULT_MODEL = "mistral-nemo:12b-instruct-2407-q8_0"

_client = OpenAI(base_url=_BASE_URL, api_key="ollama")
_aclient = AsyncOpenAI(base_url=_BASE_URL, api_key="ollama")


class EmptyCompletion(RuntimeError):
    """The model returned no answer text. Message says why, so callers can log it."""


def _kwargs(json_mode, reasoning_effort):
    extra = {}
    if json_mode:
        # Ollama honours OpenAI's response_format to force strict JSON output.
        extra["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        # Reasoning models ONLY. Ollama answers HTTP 400 '"<model>" does not
        # support thinking' for anything else, so agent_models.yaml sets this
        # per agent rather than in `defaults`.
        extra["reasoning_effort"] = reasoning_effort
    return extra


def _content(response, model):
    """The answer text, or raise with enough detail to diagnose the failure."""
    # Before the empty-answer check: the truncation described above burned a
    # full context window, and that is precisely what the report should show.
    record("ollama", model, getattr(response, "usage", None))

    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if content:
        return content

    reason = getattr(choice, "finish_reason", None)
    reasoning = getattr(choice.message, "reasoning", None) or ""
    hint = ""
    if reason == "length":
        hint = (" The context window filled up before the answer was written"
                " — lower 'reasoning_effort' for this agent in agent_models.yaml,"
                " raise OLLAMA_CONTEXT_LENGTH on the Ollama server, or shorten"
                " the prompt.")
    raise EmptyCompletion(
        f"{model} returned no content (finish_reason={reason}, "
        f"reasoning={len(reasoning)} chars).{hint}"
    )


def ollama_execute(messages, temperature=0.2, model=_DEFAULT_MODEL,
                   json_mode=False, reasoning_effort=None):
    response = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **_kwargs(json_mode, reasoning_effort),
    )
    return _content(response, model)


async def ollama_async_execute(messages, temperature=0.2, model=_DEFAULT_MODEL,
                               json_mode=False, reasoning_effort=None):
    response = await _aclient.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **_kwargs(json_mode, reasoning_effort),
    )
    return _content(response, model)
