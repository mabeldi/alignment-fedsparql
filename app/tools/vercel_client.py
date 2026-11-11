"""Chat completions through the Vercel AI Gateway.

The gateway speaks the OpenAI Chat Completions API, so this is the same trick as
tools/gemini_client.py and tools/ollama_client.py: reuse the OpenAI client and
point it at a different base URL. The agents send the identical
[{"role": ..., "content": ...}] list wherever they are pointed, and switching is
a one-line change in agent_models.yaml.

Needs VERCEL_AI_API_KEY (or AI_GATEWAY_API_KEY) in app/.env.

WHY THIS PROVIDER EXISTS
------------------------
One key reaches models from every vendor, so the frontier-model ablation does
not need a second account, a second key, or a second client — only a different
`model:` line per agent. Vercel bills provider list price with no markup, so the
cost estimates made against a vendor's own pricing still hold.

MODEL NAMES ARE NAMESPACED
--------------------------
`anthropic/claude-opus-5`, `openai/gpt-5.6-sol` — vendor first, then the model.
A bare `claude-opus-5` is not a valid id here. `list_models()` below prints what
the key can actually reach, which is worth checking before editing the YAML:
availability depends on the account, not on this file.

DIFFERENCES FROM THE OTHER PROVIDERS
------------------------------------
* No `reasoning_effort`. That flag is Ollama's, and llm_router only forwards it
  to providers in _SUPPORTS_REASONING_EFFORT.
* json_mode sends NO `response_format` at all. The gateway refuses OpenAI's
  json_object form and its json_schema alternative needs a shape no caller here
  can supply — see _kwargs, which records what each attempt actually produced.
  Callers state the shape in their prompt and strip fences when parsing.
* Every call sends max_tokens. The gateway's own default is 4096, which a model
  that reasons before answering can spend entirely on thinking — see _MAX_TOKENS
  for the two questions that measured it.
* Model availability is per ACCOUNT, not per this file. On a free-tier key most
  ids answer 403 "Free tier users do not have access to this model", and the
  ones that do answer are rate limited hard. `list_models()` lists what the
  gateway knows, NOT what the key may call — the only way to learn that is to
  make a call.
* An empty answer is an ERROR, not "". Returning "" made a truncated response
  look like a model with nothing to say, and the real cause was invisible
  several call frames away. Same reasoning as gemini_client.
"""
import os

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from tools.token_usage import record

load_dotenv()

# The gateway's OpenAI-compatible endpoint. Overridable via env var for testing.
_BASE_URL = os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")

# Only a fallback — agent_models.yaml names the model for every real call.
_DEFAULT_MODEL = "anthropic/claude-opus-5"

# Either name works. Vercel's docs call it AI_GATEWAY_API_KEY; this project's
# .env calls it VERCEL_AI_API_KEY, which reads better next to GEMINI_API_KEY and
# CLAUDE_API_KEY. Accepting both means neither the docs nor the existing file is
# wrong, and nobody has to rename a key to make an example work.
_API_KEY = os.getenv("VERCEL_AI_API_KEY") or os.getenv("AI_GATEWAY_API_KEY")

# How many tokens a reply may use. Left unset, the gateway allows 4096, and a
# model that reasons before answering can spend all of it thinking.
#
# Measured on "what are the given names of the people who are still studying?"
# and "which workshops belong to a conference, and when is their submission
# deadline?": agent 7 stopped at exactly 4096 completion tokens on both, the
# query was cut off mid-sentence, and the run saved an empty query with no error
# — the same failure ollama_client.py describes for gpt-oss in a 4096 window.
# The four questions that did finish used 673-3582, so the ceiling was the only
# thing separating them.
#
# This is a cap, not a reservation: a short answer still costs short-answer
# money. Raise it further if a harder question truncates again.
_MAX_TOKENS = int(os.getenv("VERCEL_MAX_TOKENS", "25384"))


class MissingApiKey(RuntimeError):
    """No gateway key is set — say so here rather than failing with a 401."""


class EmptyCompletion(RuntimeError):
    """The model returned no answer text. Message says why, so callers can log it."""


def _require_key():
    if not _API_KEY:
        raise MissingApiKey(
            "No gateway key found. Set VERCEL_AI_API_KEY (or "
            "AI_GATEWAY_API_KEY) in app/.env, or point the agent at another "
            "provider in config/agent_models.yaml."
        )
    return _API_KEY


# Clients are built lazily so that merely importing this module (llm_router
# imports every provider at startup) does not fail when the key is absent.
_client = None
_aclient = None


def _sync_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=_require_key(), base_url=_BASE_URL)
    return _client


def _async_client():
    global _aclient
    if _aclient is None:
        _aclient = AsyncOpenAI(api_key=_require_key(), base_url=_BASE_URL)
    return _aclient


def _kwargs(json_mode):
    """No `response_format` — see below. json_mode is honoured by the prompt.

    The gateway rejects OpenAI's `{"type": "json_object"}` with a 400 "Invalid
    input" and offers `json_schema` instead. But callers here never describe
    their reply shape, and a schema vague enough to mean "any object" makes
    strict models produce nonsense rather than freeing them:

      {"type": "object"}                      -> minimax returned `{}`
      {"type": "object", additionalProperties} -> minimax returned arrays as
                                                  STRINGS: {"classes": "[]"}

    Measured on "which papers were withdrawn?", the first form cost the whole
    run: agent 1 extracted nothing, nothing was retrieved, no query was written,
    in 3 seconds, with no error raised anywhere. Claude had masked it by
    ignoring the restriction, so the bug only appeared on a model that obeys.

    Asking for a shape we cannot describe is worse than not asking. Every caller
    already states the shape in its prompt and tolerates fences when parsing, and
    that path parses correctly on 5/5 gold questions with the right types. So
    this provider is one that ignores response_format, which llm_router already
    documents as allowed.
    """
    return {}


def _content(response, model):
    """The answer text, or raise with enough detail to diagnose the failure."""
    # Before the empty-answer check: a truncated call still cost its prompt
    # tokens, and those are exactly the ones worth seeing in the report.
    record("vercel", model, getattr(response, "usage", None))

    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if content:
        return content

    reason = getattr(choice, "finish_reason", None)
    hint = ""
    if reason == "length":
        hint = (" The output limit was reached before the answer finished"
                " — shorten the prompt or raise max_tokens.")
    elif reason == "content_filter":
        hint = " The provider's safety filter blocked the response."
    raise EmptyCompletion(
        f"{model} returned no content (finish_reason={reason}).{hint}"
    )


def vercel_execute(messages, temperature=0.2, model=_DEFAULT_MODEL,
                   json_mode=False):
    response = _sync_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=_MAX_TOKENS,
        **_kwargs(json_mode),
    )
    return _content(response, model)


async def vercel_async_execute(messages, temperature=0.2, model=_DEFAULT_MODEL,
                               json_mode=False):
    response = await _async_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=_MAX_TOKENS,
        **_kwargs(json_mode),
    )
    return _content(response, model)


def list_models():
    """The model ids this key can reach — run before editing the YAML.

    Availability is per account, so a name from the docs is not proof it works
    here. Sorted, because the list is long and grouped by vendor prefix.
    """
    return sorted(m.id for m in _sync_client().models.list().data)


if __name__ == "__main__":
    for name in list_models():
        print(name)
